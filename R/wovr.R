#' @title Weighted One-Vs-Rest
#' @description Creates a cost-sensitive classifier by creating one classifier per
#' class to predict cost. Takes as input a classifier accepting observation weights.
#' The objective is to create a model that would predict the
#' class with the minimum cost.
#' @param X The data (covariates/features).
#' @param C matrix(n_samples, n_classes) Costs for each class for each observation.
#' @param classifier function(X, y, weights=w, ...) -> object, that would create a classifier with method `predict`.
#' The `y` vector passed to it is of class `integer` with values 0/1 only.
#' @param predict_type_prob argument to pass to method `predict` from the classifier passed to `classifier`
#' in order to output probabilities or numeric scores instead of classes
#' (i.e. `predict(object, newdata, type=predict_type_prob`)).
#' @param wap_weights Whether to use the weighting technique from the `Weighted-All-Pairs` algorithm.
#' @param nthreads Number of parallel threads to use (not available on Windows systems). Note
#' that, unlike the Python version, this is not a shared memory model and each additional thread will
#' require more memory from the system. Not recommended to use when the algorithm is itself parallelized.
#' @param ... Extra arguments to pass to `classifier`.
#' @references Beygelzimer, A., Dani, V., Hayes, T., Langford, J., & Zadrozny, B. (2005, August). Error limiting reductions between classification tasks.
#' @export
#' @examples
#' library(costsensitive)
#' wrapped.logistic <- function(X, y, weights, ...) {
#' 	return(glm(y ~ ., data = X, weights = weights, family = "quasibinomial", ...))
#' }
#' set.seed(1)
#' X <- data.frame(feature1 = rnorm(100), feature2 = rnorm(100), feature3 = runif(100))
#' C <- data.frame(cost1 = rgamma(100, 1), cost2 = rgamma(100, 1), cost3 = rgamma(100, 1))
#' model <- weighted.one.vs.rest(X, C, wrapped.logistic, predict_type_prob = "response")
#' predict(model, X, type = "class")
#' predict(model, X, type = "score")
#' print(model)
#' 
weighted.one.vs.rest <- function(X, C, classifier, predict_type_prob = "prob", wap_weights = FALSE, nthreads = 1, ...) {
	out <- extract.info(C, nthreads)
	nclasses <- length(out[["classes"]])
	out[["predict_type_prob"]] <- predict_type_prob
	out[["wap_weights"]] <- as.logical(wap_weights)
	
	if (wap_weights) {
		C <- calc.V(C)
	}
	if (out[["nthreads"]] == 1) {
		out[["classifiers"]] <- lapply(1:nclasses, fit.single.wovr, X, C, classifier, ...)
	} else {
		out[["classifiers"]] <- parallel::mclapply(1:nclasses, fit.single.wovr, X, C, classifier, ...,
												   mc.cores = out[["nthreads"]])	
	}
	return(structure(out, class = "wovr"))
}

fit.single.wovr <- function(cl, X, C, classifier, ...) {
	cost_choice <- C[, cl]
	cost_others <- C[, 1:NCOL(C) != cl]
	cost_others <- apply(cost_others, 1, min)
	w <- abs(cost_choice - cost_others)
	y <- as.integer(cost_choice < cost_others)
	valid_cases <- w > 0
	X_take <- X[valid_cases, ]
	y_take <- y[valid_cases]
	w_take <- w[valid_cases]
	w_take <- standardize.weights(w_take)
	return(classifier(X_take, y_take, weights = w_take))
}

#' @title Predict method for Weighted One-Vs-Rest
#' @description Predicts either the class with expected minimum cost or scores (more is better) for new data.
#' @param object An object of class `wovr` as output by function `weighted.one.vs.rest`.
#' @param newdata New data on which to make predictions.
#' @param type One of "class" (will output the class with minimum expected cost) or "score"
#' (will output the predicted score for each class, i.e. more is better).
#' @param ... Additional arguments to pass to the predict method of the base classifier.
#' @return When passing `type = "class"`, a vector with class numbers or names (if the cost matrix had them).
#' When passing `type = "score"`, will output a `data.frame` with the same number of columns as `C` (passed to
#' the `weighted.one.vs.rest` function) and the predicted score for each observation and class.
#' @export
predict.wovr <- function(object, newdata, type = "class", ...) {
	check.predict.type(type)
	if (object[["nthreads"]] == 1) {
		pred <- as.data.frame(lapply(object[["classifiers"]],
									 function(m) predict(m, newdata, type = object[["predict_type_prob"]], ...)))
	} else {
		pred <- as.data.frame(parallel::mclapply(object[["classifiers"]],
												 function(m) predict(m, newdata, type = object[["predict_type_prob"]], ...),
												 mc.cores = object[["nthreads"]]))
	}
	if (type == "class") {
		return(object[["classes"]][apply(pred, 1, which.max)])
	} else {
		colnames(pred) <- object[["classes"]]
		return(pred)
	}
}

#' @title Get information about Weighted One-Vs-Rest object
#' @description Prints basic information about a `wovr` object
#' (Number of classes, classifier class).
#' @param x An object of class "wovr".
#' @param ... Extra arguments (not used).
#' @export
print.wovr <- function(x, ...) {
	cat("Weighted One-vs-Rest (cost-sensitive classifier)\n\n")
	cat("Classifier type:", class(x[["classifiers"]][[1]]), "\n")
	cat("WAP weighting technique:", x[["wap_weights"]], "\n")
	cat("Number of classes:", length(x[["classes"]]), "\n")
	cat("Class names:", head(x[["classes"]]), "\n")
}

#' @title Get information about Weighted One-Vs-Rest object
#' @description Prints basic information about a `wovr` object
#' (Number of classes, classifier class). Same as function `print`.
#' @param object An object of class "wovr".
#' @param ... Extra arguments (not used).
#' @export
summary.wovr <- function(object, ...) {
	print.wovr(object)
}
