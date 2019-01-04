import numpy as np, pandas as pd
from copy import deepcopy
from scipy.stats import mode

class WeightedAllPairs:
    """
    Weighted All-Pairs for Cost-Sensitive Classification
    
    Note
    ----
    This implementation also offers the option of weighting each observation
    in a pairwise comparison according to the absolute difference in costs
    between the two labels. Even though such a method might not enjoy theoretical
    bounds on its regret or error, in practice, it can produce better results
    than the weighting schema proposed in [1] and [2]
    
    Parameters
    ----------
    base_classifier : object
        Base binary classification algorithm. Must have:
            * A fit method of the form 'base_classifier.fit(X, y, sample_weights = w)'.
            * A predict method.
    weight_simple_diff : bool
        Whether to weight each sub-problem according to the absolute difference in
        costs between labels, or according to the formula described in [1] (See Note)
    
    Attributes
    ----------
    nclasses : int
        Number of classes on the data in which it was fit.
    classifiers : list of objects
        Classifier that compares each two classes. Classes i and j out of n classes, with i<j,
        are compared by the classifier at index i*(n-(i+1)/2)+j-i-1.
    weight_simple_diff : bool
        Whether each sub-problem was weighted according to the absolute difference in
        costs between labels, or according to the formula described in [1]
    base_classifier : object
        Unfitted base regressor that was originally passed.
        
    References
    ----------
    [1] Beygelzimer, A., Dani, V., Hayes, T., Langford, J., & Zadrozny, B. (2005)
        Error limiting reductions between classification tasks.
    [2] Beygelzimer, A., Langford, J., & Zadrozny, B. (2008).
        Machine learning techniques—reductions between prediction quality metrics.
    """
    def __init__(self, base_classifier, weigh_by_cost_diff=True):
        self.base_classifier=base_classifier
        self.weigh_by_cost_diff=weigh_by_cost_diff
    
    def fit(self, X, C):
        """
        Fit one classifier comparing each pair of classes
        
        Parameters
        ----------
        X : array (n_samples, n_features)
            The data on which to fit a cost-sensitive classifier.
        C : array (n_samples, n_classes)
            The cost of predicting each label for each observation (more means worse).
        """
        X,C = _check_fit_input(X,C)
        self.nclasses=C.shape[1]
        ncombs=int(self.nclasses*(self.nclasses-1)/2)
        self.classifiers=[deepcopy(self.base_classifier) for c in range(ncombs)]
        self.classes_compared=[None for i in range(ncombs)]
        if self.weigh_by_cost_diff:
            V=C
        else:
            V=self._calculate_v(C)
        
        for i in range(self.nclasses-1):
            for j in range(i+1,self.nclasses):
                y=(V[:,i]<V[:,j]).astype('uint8')
                w=np.abs(V[:,i]-V[:,j])
                valid_cases=w>0
                X_take=X[valid_cases,:]
                y_take=y[valid_cases]
                w_take=w[valid_cases]
                w_take=_standardize_weights(w_take)
                ix=self._get_comb_index(i,j)
                self.classes_compared[ix]=(j,i)
                self.classifiers[ix].fit(X_take, y_take, sample_weight=w_take)
        self.classes_compared=np.array(self.classes_compared)
        return self
        
    def decision_function(self, X, method='most-wins'):
        """
        Calculate a 'goodness' distribution over labels
        
        Note
        ----
        Predictions can be calculated either by counting which class wins the most
        pairwise comparisons (as in [1] and [2]), or - for classifiers with a 'predict_proba'
        method - by taking into account also the margins of the prediction difference
        for one class over the other for each comparison.
        
        If passing method = 'most-wins', this 'decision_function' will output the proportion
        of comparisons that each class won. If passing method = 'goodness', it sums the
        outputs from 'predict_proba' from each pairwise comparison and divides it by the
        number of comparisons.
        
        Using method = 'goodness' requires the base classifier to have a 'predict_proba' method.
        
        Parameters
        ----------
        X : array (n_samples, n_features)
            Data for which to predict the cost of each label.
        method : str, either 'most-wins' or 'goodness':
            How to decide the best label (see Note)
        
        Returns
        -------
        pred : array (n_samples, n_classes)
            A goodness score (more is better) for each label and observation.
            If passing method='most-wins', it counts the proportion of comparisons
            that each class won.
            If passing method='goodness', it sums the outputs from 'predict_proba' from
            each pairwise comparison and divides it by the number of comparisons.
            
        References
        ----------
        [1] Beygelzimer, A., Dani, V., Hayes, T., Langford, J., & Zadrozny, B. (2005)
            Error limiting reductions between classification tasks.
        [2] Beygelzimer, A., Langford, J., & Zadrozny, B. (2008).
            Machine learning techniques—reductions between prediction quality metrics.
        """
        X=_check_predict_input(X)
        if len(X.shape)==1:
            X=X.reshape(1,-1)
        if method=='most-wins':
            return self._decision_function_winners(X)
        elif method=='goodness':
            return self._decision_function_goodness(X)
        else:
            raise ValueError("method must be one of 'most-wins' or 'goodness'.")
    
    def predict(self, X, method='most-wins'):
        """
        Predict the less costly class for a given observation
        
        Note
        ----
        Predictions can be calculated either by counting which class wins the most
        pairwise comparisons (as in [1] and [2]), or - for classifiers with a 'predict_proba'
        method - by taking into account also the margins of the prediction difference
        for one class over the other for each comparison.
        
        Using method = 'goodness' requires the base classifier to have a 'predict_proba' method.
        
        Parameters
        ----------
        X : array (n_samples, n_features)
            Data for which to predict minimum cost label.
        method : str, either 'most-wins' or 'goodness':
            How to decide the best label (see Note)
        
        Returns
        -------
        y_hat : array (n_samples,)
            Label with expected minimum cost for each observation.
            
        References
        ----------
        [1] Beygelzimer, A., Dani, V., Hayes, T., Langford, J., & Zadrozny, B. (2005)
            Error limiting reductions between classification tasks.
        [2] Beygelzimer, A., Langford, J., & Zadrozny, B. (2008).
            Machine learning techniques—reductions between prediction quality metrics.
        """
        X=_check_predict_input(X)
        if len(X.shape)==1:
            X=X.reshape(1,-1)
        if method=='most-wins':
            winners=[self.classes_compared[np.repeat(c,X.shape[0]),self.classifiers[c].predict(X).astype('uint8')]\
                         for c in range(len(self.classifiers))]
            winners=np.vstack(winners)
            winners = mode(winners, axis=0)[0].reshape(-1)
            if winners.shape[0]==1:
                return winners[0]
            else:
                return winners
        elif method=='goodness':
            goodness=self._decision_function_goodness(X)
            if (len(goodness.shape)==1) or (goodness.shape[0]==1):
                return np.argmax(goodness)
            else:
                return np.argmax(goodness, axis=1)
        else:
            raise ValueError("method must be one of 'most-wins' or 'largest-goodness'.")
            
    def _decision_function_winners(self, X):
        winners=np.zeros((X.shape[0], self.nclasses))
        for c in range(len(self.classifiers)):
            round_comp=self.classes_compared[np.repeat(c,X.shape[0]),self.classifiers[c].predict(X).astype('uint8')]
            winners[np.arange(X.shape[0]), round_comp]+=1
        winners=winners/len(self.classifiers)
        return winners
    
    def _decision_function_goodness(self, X):
        if 'predict_proba' not in dir(self.classifiers[0]):
            raise Exception("'goodness' method requires a classifier with 'predict_proba' method.")
        goodness=np.zeros((X.shape[0],self.nclasses))
        for c in range(len(self.classifiers)):
            comp=comp=self.classifiers[c].predict_proba(X)
            goodness[:,int(self.classes_compared[c,0])]+=comp[:,0]
            goodness[:,int(self.classes_compared[c,1])]+=comp[:,1]
        return goodness/len(self.classifiers)
    
    def _calculate_v(self,C):
        V=list()
        for cost_vector in C:
            cost=cost_vector.copy()
            out_order=np.argsort(cost)
            cost=cost[out_order]-cost.min()
            n=cost.shape[0]
            v=np.zeros(n)
            rectangle_width=np.diff(cost)
            rectangle_height=1/(np.arange(n-1)+1)
            v[1:]=rectangle_width*rectangle_height
            V.append(np.cumsum(v)[np.argsort(out_order)])
        return np.vstack(V)
    
    def _get_comb_index(self,i,j):
        return int(i*(self.nclasses-(i+1)/2)+j-i-1)

class _BinTree:
    # constructs a balanced binary tree
    # keeps track of which nodes compare which classes
    # node_comparisons -> [all nodes, nodes to the left]
    # childs -> [child left, child right]
        # terminal nodes are negative numbers
        # non-terminal nodes refer to the index in 'node_comparisons' for next comparison
    def __init__(self,n):
        self.n_arr=np.arange(n)
        self.node_comparisons=[[None,None,None] for i in range(n-1)]
        self.node_counter=0
        self.childs=[[None,None] for i in range(n-1)]
        self.parents=[None for i in range(n-1)]
        self.isterminal=set()
        
        split_point=int(np.ceil(self.n_arr.shape[0]/2))
        self.node_comparisons[0][0]=list(self.n_arr)
        self.node_comparisons[0][1]=list(self.n_arr[:split_point])
        self.node_comparisons[0][2]=list(self.n_arr[split_point:])
        self.split_arr(self.n_arr[:split_point],0,True)
        self.split_arr(self.n_arr[split_point:],0,False)
        self.isterminal=list(self.isterminal)
        self.is_at_bottom=[i for i in range(len(self.childs)) if (self.childs[i][0]<=0) and (self.childs[i][1]<=0)]
        
    def split_arr(self,arr,parent_node,direction_left):
        if arr.shape[0]==1:
            if direction_left:
                self.childs[parent_node][0]=-arr[0]
            else:
                self.childs[parent_node][1]=-arr[0]
            self.isterminal.add(parent_node)
            return None
        
        self.node_counter+=1
        curr_node=self.node_counter
        if direction_left:
            self.childs[parent_node][0]=curr_node
        else:
            self.childs[parent_node][1]=curr_node
        self.parents[curr_node]=parent_node
        
        split_point=int(np.ceil(arr.shape[0]/2))
        self.node_comparisons[curr_node][0]=list(arr)
        self.node_comparisons[curr_node][1]=list(arr[:split_point])
        self.node_comparisons[curr_node][2]=list(arr[split_point:])
        self.split_arr(arr[:split_point],curr_node,True)
        self.split_arr(arr[split_point:],curr_node,False)
        return None

class FilterTree:
    """
    Filter-Tree for Cost-Sensitive Multi-Class classification
    
    Parameters
    ----------
    base_classifier : object
        Base binary classification algorithm. Must have:
            * A fit method of the form 'base_classifier.fit(X, y, sample_weights = w)'.
            * A predict method.
    
    Attributes
    ----------
    nclasses : int
        Number of classes on the data in which it was fit.
    classifiers : list of objects
        Classifier that compares each two classes belonging to a node.
    tree : object
        Binary tree with attributes childs and parents.
        Non-negative numbers for children indicate non-terminal nodes,
        while negative and zero indicates a class (terminal node).
        Root is the node zero.
    base_classifier : object
        Unfitted base regressor that was originally passed.
    
    References
    ----------
    [1] Beygelzimer, A., Langford, J., & Ravikumar, P. (2007).
        Multiclass classification with filter trees.
    """
    def __init__(self, base_classifier):
        self.base_classifier=base_classifier
    
    def fit(self, X, C):
        """
        Fit a filter tree classifier
        
        Note
        ----
        Shifting the order of the classes within the cost array will produce different
        results, as it will build a different binary tree comparing different classes
        at each node.
        
        Parameters
        ----------
        X : array (n_samples, n_features)
            The data on which to fit a cost-sensitive classifier.
        C : array (n_samples, n_classes)
            The cost of predicting each label for each observation (more means worse).
        """
        X,C = _check_fit_input(X,C)
        nclasses=C.shape[1]
        self.tree=_BinTree(nclasses)
        self.classifiers=[deepcopy(self.base_classifier) for c in range(nclasses-1)]
        classifier_queue=self.tree.is_at_bottom
        next_round=list()
        already_fitted=set()
        labels_take=-np.ones((X.shape[0],len(self.classifiers)))
        while True:
            for c in classifier_queue:
                if c in already_fitted or (c is None):
                    continue
                child1, child2 = self.tree.childs[c]
                if (child1>0) and (child1 not in already_fitted):
                    continue
                if (child2>0) and (child2 not in already_fitted):
                    continue
                    
                if child1<=0:
                    class1=-np.repeat(child1,X.shape[0]).astype('int64')
                else:
                    class1=labels_take[:, child1].astype('int64')
                if child2<=0:
                    class2=-np.repeat(child2,X.shape[0]).astype('int64')
                else:
                    class2=labels_take[:, child2].astype('int64')


                cost1=C[np.arange(X.shape[0]),np.clip(class1,a_min=0,a_max=None)]
                cost2=C[np.arange(X.shape[0]),np.clip(class2,a_min=0,a_max=None)]
                y=(cost1<cost2).astype('uint8')
                w=np.abs(cost1-cost2)

                valid_obs=w>0
                if child1>0:
                    valid_obs=valid_obs&(labels_take[:,child1]>=0)
                if child2>0:
                    valid_obs=valid_obs&(labels_take[:,child2]>=0)
                
                X_take=X[valid_obs,:]
                y_take=y[valid_obs]
                w_take=w[valid_obs]
                w_take=_standardize_weights(w_take)
                
                self.classifiers[c].fit(X_take,y_take,sample_weight=w_take)
                
                labels_arr=np.c_[class1,class2].astype('int64')
                labels_take[valid_obs,c]=labels_arr[np.repeat(0,X_take.shape[0]),\
                                                    self.classifiers[c].predict(X_take).reshape(-1).astype('uint8')]
                already_fitted.add(c)
                next_round.append(self.tree.parents[c])
                if c==0 or (len(classifier_queue)==0):
                    break
            classifier_queue=list(set(next_round))
            next_round=list()
            if (len(classifier_queue)==0):
                break
        return self
    
    def _predict(self, X):
        curr_node=0
        while True:
            go_right=self.classifiers[curr_node].predict(X)
            if go_right:
                curr_node=self.tree.childs[curr_node][0]
            else:
                curr_node=self.tree.childs[curr_node][1]
                
            if curr_node<=0:
                return -curr_node

            
    def predict(self, X):
        """
        Predict the less costly class for a given observation
        
        Note
        ----
        The implementation here happens in a Python loop rather than in some
        NumPy array operations, thus it will be slower than the other algorithms
        here, even though in theory it implies fewer comparisons.
        
        Parameters
        ----------
        X : array (n_samples, n_features)
            Data for which to predict minimum cost label.
        method : str, either 'most-wins' or 'goodness':
            How to decide the best label (see Note)
        
        Returns
        -------
        y_hat : array (n_samples,)
            Label with expected minimum cost for each observation.
        """
        X=_check_predict_input(X)
        if len(X.shape)==1:
            return self._predict(X.reshape(1, -1))
        elif X.shape[0]==1:
            return self._predict(X)
        else:
            out=list()
            for i in range(X.shape[0]):
                out.append(self._predict(X[i,:].reshape(1, -1)))
            return np.array(out)

class CostProportionateClassifier:
    """
    Cost-Proportionate Rejection Sampling
    
    Turns a binary classifier with no native sample weighting method into a
    binary classifier that supports sample weights.
    
    Parameters
    ----------
    base_classifier : object
        Binary classifier used for predicting in each sample. Must have:
            * A fit method of the form 'base_classifier.fit(X, y)'.
            * A predict method.
    n_samples : int
        Number of samples taken. One classifier is fit per sample.
    
    Attributes
    ----------
    n_samples : int
        Number of samples taken. One classifier is fit per sample.
    classifiers : list of objects
        Classifier that was fit to each sample.
    base_classifier : object
        Unfitted base classifier that was originally passed.
    extra_rej_const : float
        Extra rejection constant used for sampling (see 'fit' method).
    
    References
    ----------
    [1] Beygelzimer, A., Langford, J., & Zadrozny, B. (2008).
        Machine learning techniques—reductions between prediction quality metrics.
    """
    def __init__(self, base_classifier, n_samples=10, extra_rej_const=1e-1):
        self.base_classifier = base_classifier
        self.n_samples = n_samples
        self.extra_rej_const = extra_rej_const
    
    def fit(self, X, y, sample_weight=None):
        """
        Fit a binary classifier with sample weights to data.
        
        Note
        ----
        Examples at each sample are accepted with probability = weight/Z,
        where Z = max(weight) + extra_rej_const.
        Larger values for extra_rej_const ensure that no example gets selected in
        every single sample, but results in smaller sample sizes as more examples are rejected.
        
        Parameters
        ----------
        X : array (n_samples, n_features)
            Data on which to fit the model.
        y : array (n_samples,) or (n_samples, 1)
            Class of each observation.
        sample_weight : array (n_samples,) or (n_samples, 1)
            Weights indicating how important is each observation in the loss function.
        """
        assert self.extra_rej_const >= 0
        if sample_weight is None:
            sample_weight = np.ones(y.shape[0])
        else:
            if isinstance(sample_weight, list):
                sample_weight = np.array(sample_weight)
            if len(sample_weight.shape):
                sample_weight = sample_weight.reshape(-1)
        assert sample_weight.shape[0] == X.shape[0]
        assert sample_weight.min() > 0
        
        Z = sample_weight.max() + self.extra_rej_const
        sample_weight[:] = sample_weight / Z # sample weight is now acceptance prob
        self.classifiers = [deepcopy(self.base_classifier) for c in range(self.n_samples)]
        for c in range(self.n_samples):
            take = np.random.random(size = X.shape[0]) <= sample_weight
            self.classifiers[c].fit(X[take, :], y[take])
        return self
    
    def decision_function(self, X, aggregation='raw'):
        """
        Calculate how preferred is positive class according to classifiers
        
        Note
        ----
        If passing aggregation = 'raw', it will output the proportion of the classifiers
        that voted for the positive class.
        If passing aggregation = 'weighted', it will output the average predicted probability
        for the positive class for each classifier.
        
        Calculating it with aggregation = 'weighted' requires the base classifier to have a
        'predict_proba' method.
        
        Parameters
        ----------
        X : array (n_samples, n_features):
            Observations for which to determine class likelihood.
        aggregation : str, either 'raw' or 'weighted'
            How to compute the 'goodness' of the positive class (see Note)
            
        Returns
        -------
        pred : array (n_samples,)
            Score for the positive class (see Note)
        """
        if aggregation=='weighted':
            if 'predict_proba' not in dir(self.classifiers[0]):
                raise Exception("'aggregation='weighted'' is only available for classifiers with 'predict_proba' method.")
        preds=list()
        for c in self.classifiers:
            if aggregation=='raw':
                preds.append(c.predict(X).reshape(-1))
            elif aggregation=='weighted':
                preds.append(c.predict_proba(X)[:,1].reshape(-1))
            else:
                raise ValueError("'aggregation' must be one of 'raw' or 'weighted'.")
        return np.vstack(preds).mean(axis=0)
    
    def predict(self, X, aggregation='raw'):
        """
        Predict the class of an observation
        
        Note
        ----
        If passing aggregation = 'raw', it will output the class that most classifiers outputted,
        breaking ties by predicting the positive class.
        If passing aggregation = 'weighted', it will weight each vote from a classifier according
        to the probabilities predicted.
        
        Predicting with aggregation = 'weighted' requires the base classifier to have a
        'predict_proba' method.
        
        Parameters
        ----------
        X : array (n_samples, n_features):
            Observations for which to predict their class.
        aggregation : str, either 'raw' or 'weighted'
            How to compute the 'goodness' of the positive class (see Note)
        
        Returns
        -------
        pred : array (n_samples,)
            Predicted class for each observation.
        """
        return (self.decision_function(X,aggregation)>=.5).astype('int64')

class WeightedOneVsRest:
    """
    Weighted One-Vs-Rest Cost-Sensitive Classification
    
    Note
    ----
    This will convert the problem into one sub-problem per class.
    
    If passing weight_simple_diff=True, the observations for each subproblem
    will be weighted according to the difference between the cost of the label being
    predicted and the minimum cost of any other label.
    
    If passing weight_simple_diff=False, they will be weighted according to the formula
    described in [1], originally meant for the All-Pairs variant.
    
    The predictions are taken to be the maximum value of the decision functions of
    each One-Vs-Rest classifier. If the classifier has no method 'decision_function' or
    'predict_proba', it will output the class that whatever classifier considered correct,
    breaking ties by choosing the smallest index.
    
    Parameters
    ----------
    base_classifier : object
        Base binary classification algorithm. Must have:
            * A fit method of the form 'base_classifier.fit(X, y, sample_weights = w)'.
            * A predict method.
    weight_simple_diff : bool
        Whether to weight each sub-problem according to the absolute difference in
        costs between labels, or according to the formula described in [1] (See Note)
    
    Attributes
    ----------
    nclasses : int
        Number of classes on the data in which it was fit.
    classifiers : list of objects
        Classifier that predicts each class.
    weight_simple_diff : bool
        Whether each sub-problem was weighted according to the absolute difference in
        costs between labels, or according to the formula described in [1].
    base_classifier : object
        Unfitted base regressor that was originally passed.
        
    References
    ----------
    [1] Beygelzimer, A., Dani, V., Hayes, T., Langford, J., & Zadrozny, B. (2005, August).
        Error limiting reductions between classification tasks.
    """
    def __init__(self, base_classifier, weight_simple_diff = False):
        self.base_classifier=base_classifier
        self.weight_simple_diff=weight_simple_diff
        
    def fit(self, X, C):
        """
        Fit one weighted classifier per class
        
        Parameters
        ----------
        X : array (n_samples, n_features)
            The data on which to fit a cost-sensitive classifier.
        C : array (n_samples, n_classes)
            The cost of predicting each label for each observation (more means worse).
        """
        X,C = _check_fit_input(X,C)
        self.nclasses=C.shape[1]
        self.classifiers=[deepcopy(self.base_classifier) for i in range(self.nclasses)]
        if not self.weight_simple_diff:
            C=WeightedAllPairs._calculate_v(None,C)
        for c in range(self.nclasses):
            cols_rest=[i for i in range(self.nclasses)]
            del cols_rest[c]
            cost_choice=C[:,c]
            cost_others=C[:,cols_rest].min(axis=1)
            w=np.abs(cost_choice-cost_others)
            y=(cost_choice<cost_others).astype('uint8')
            valid_cases=w>0
            X_take=X[valid_cases,:]
            y_take=y[valid_cases]
            w_take=w[valid_cases]
            w_take=_standardize_weights(w_take)
            
            self.classifiers[c].fit(X_take, y_take, sample_weight=w_take)
        return self
    
    def decision_function(self, X, apply_softmax = True):
        """
        Calculate a 'goodness' distribution over labels
        
        Note
        ----
        This will only work if the base classifiers has a 'predict_proba' method.
        It will output the predicted probabilities of each class being the less costly
        according to each classifier.
        
        If passing apply_softmax = True, it will then apply a softmax transformation so
        that these scores sum up to 1 (per row).
        
        Parameters
        ----------
        X : array (n_samples, n_features)
            Data for which to predict the cost of each label.
        apply_softmax : bool
            Whether to apply a softmax transform to the 'goodness' (see Note).
        
        Returns
        -------
        pred : array (n_samples, n_classes)
            A goodness score (more is better) for each label and observation.
            If passing apply_softmax=True, these are standardized to sum up to 1 (per row).
        """
        X=_check_predict_input(X)
        if len(X.shape)==1:
            X=X.reshape(1,-1)
        preds=np.zeros((X.shape[0],self.nclasses))
        for c in range(self.nclasses):
            try:
                preds[:,c]=self.classifiers[c].decision_function(X)
            except:
                try:
                    preds[:,c]=self.classifiers[c].predict_proba(X)[:,1]
                    apply_softmax=False
                except:
                    preds[:,c]=self.classifiers[c].predict(X)
                    apply_softmax=False
        if apply_softmax:
            preds=np.exp(preds - preds.max(axis=1).reshape(-1,1))
            preds=preds/preds.sum(axis=1).reshape(-1,1)
        return preds
    
    def predict(self, X):
        """
        Predict the less costly class for a given observation
        
        Parameters
        ----------
        X : array (n_samples, n_features)
            Data for which to predict minimum cost label.
        
        Returns
        -------
        y_hat : array (n_samples,)
            Label with expected minimum cost for each observation.
        """
        X=_check_predict_input(X)
        return np.argmax(self.decision_function(X,False), axis=1)
    
class RegressionOneVsRest:
    """
    Regression One-Vs-Rest
    
    Fits one regressor trying to predict the cost of each class.
    Predictions are the class with the minimum predicted cost across regressors.
    
    Parameters
    ----------
    base_regressor : object
        Regressor to be used for the sub-problems. Must have:
            * A fit method of the form 'base_classifier.fit(X, y)'.
            * A predict method.
    
    Attributes
    ----------
    nclasses : int
        Number of classes on the data in which it was fit.
    regressors : list of objects
        Regressor that predicts the cost of each class.
    base_regressor : object
        Unfitted base regressor that was originally passed.
        
    References
    ----------
    [1] Beygelzimer, A., Langford, J., & Zadrozny, B. (2008).
        Machine learning techniques—reductions between prediction quality metrics.
    """
    def __init__(self, base_regressor):
        self.base_regressor=base_regressor
        
    def fit(self, X, C):
        """
        Fit one regressor per class
        
        Parameters
        ----------
        X : array (n_samples, n_features)
            The data on which to fit a cost-sensitive classifier.
        C : array (n_samples, n_classes)
            The cost of predicting each label for each observation (more means worse).
        """
        X,C = _check_fit_input(X,C)
        self.nclasses=C.shape[1]
        self.regressors=[deepcopy(self.base_regressor) for i in range(self.nclasses)]
        for c in range(self.nclasses):
            cost_choice=C[:,c]
            self.regressors[c].fit(X, cost_choice)
        return self
    
    def decision_function(self, X, apply_softmax=True):
        """
        Get cost estimates for each observation
        
        Note
        ----
        If called with apply_softmax = False, this will output the predicted
        COST rather than the 'goodness' - meaning, more is worse.
        
        If called with apply_softmax = True, it will output one minus the softmax on the costs,
        producing a distribution over the choices summing up to 1 where more is better.
        
        
        Parameters
        ----------
        X : array (n_samples, n_features)
            Data for which to predict the cost of each label.
        apply_softmax : bool
            Whether to apply a softmax transform to the costs (see Note).
        
        Returns
        -------
        pred : array (n_samples, n_classes)
            Either predicted cost or a distribution of 'goodness' over the choices,
            according to the apply_softmax argument.
        """
        X=_check_predict_input(X)
        if len(X.shape)==1:
            X=X.reshape(1,-1)
        preds=np.zeros((X.shape[0],self.nclasses))
        for c in range(self.nclasses):
            preds[:,c]=self.regressors[c].predict(X)
        if not apply_softmax:
            return preds
        else:
            preds=np.exp(preds - preds.max(axis=1).reshape(-1,1))
            preds=preds/preds.sum(axis=1).reshape(-1,1)
            return 1-preds
    
    def predict(self, X):
        """
        Predict the less costly class for a given observation
        
        Parameters
        ----------
        X : array (n_samples, n_features)
            Data for which to predict minimum cost labels.
        
        Returns
        -------
        y_hat : array (n_samples,)
            Label with expected minimum cost for each observation.
        """
        X=_check_predict_input(X)
        return np.argmin(self.decision_function(X,False), axis=1)
    
def _check_fit_input(X,C):
    if type(X)==pd.core.frame.DataFrame:
        X=X.as_matrix()
    if type(X)==np.matrixlib.defmatrix.matrix:
        X=np.array(X)
    if type(X)!=np.ndarray:
        raise ValueError("'X' must be a numpy array or pandas data frame.")
        
    if type(C)==pd.core.frame.DataFrame:
        C=C.as_matrix()
    if type(C)==np.matrixlib.defmatrix.matrix:
        C=np.array(C)
    if type(X)!=np.ndarray:
        raise ValueError("'C' must be a numpy array or pandas data frame.")
        
    assert X.shape[0]==C.shape[0]
    assert C.shape[1]>2
    
    return X,C
    
def _check_predict_input(X):
    if type(X)==pd.core.frame.DataFrame:
        X=X.as_matrix()
    if type(X)==np.matrixlib.defmatrix.matrix:
        X=np.array(X)
    if type(X)!=np.ndarray:
        raise ValueError("'X' must be a numpy array or pandas data frame.")
    return X

def _standardize_weights(w):
    return w*w.shape[0]/w.sum()
