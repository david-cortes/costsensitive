#' @title Weighted All-Pairs
#' @description Creates a cost-sensitive classifier by creating one classifier per
#' pair of classes to predict cost. Takes as input a classifier accepting observation weights.
#' The objective is to create a model that would predict the
#' class with the minimum cost.
#' @param X The data (covariates/features).
#' @param C matrix(n_samples, n_classes) Costs for each class for each observation.
#' @param classifier function(X, y, weights=w, ...) -> object, that would create a classifier with method `predict`.
#' The `y` vector passed to it is of class `integer` with values 0/1 only.
#' @param predict_type_prob argument to pass to method `predict` from the classifier passed to `classifier`
#' in order to output probabilities (must be between zero and one) instead of classes
#' (i.e. `predict(object, newdata, type=predict_type_prob`)).
#' @param wap_weights Whether to use the weighting technique from the `Weighted-All-Pairs` algorithm.
#' @param nthreads Number of parallel threads to use (not available on Windows systems). Note
#' that, unlike the Python version, this is not a shared memory model and each additional thread will
#' require more memory from the system. Not recommended to use when the algorithm is itself parallelized.
#' @param ... Extra arguments to pass to `classifier`.
#' @references Beygelzimer, A., Langford, J., & Zadrozny, B. (2008). Machine learning techniques-reductions between prediction quality metrics.
#' @examples 
#' library(costsensitive)
#' wrapped.logistic <- function(X, y, weights, ...) {
#' 	return(glm(y ~ ., data = X, weights = weights, family = "quasibinomial", ...))
#' }
#' set.seed(1)
#' X <- data.frame(feature1 = rnorm(100), feature2 = rnorm(100), feature3 = runif(100))
#' C <- data.frame(cost1 = rgamma(100, 1), cost2 = rgamma(100, 1), cost3 = rgamma(100, 1))
#' model <- weighted.all.pairs(X, C, wrapped.logistic, predict_type_prob = "response")
#' predict(model, X, type = "class")
#' predict(model, X, type = "score")
#' print(model)
#' 
weighted.all.pairs <- function(X, C, classifier, predict_type_prob = "prob", wap_weights = TRUE, nthreads = 1, ...) {
	out <- extract.info(C, nthreads)
	nclasses <- length(out[["classes"]])
	out[["ncombs"]] <- nclasses * (nclasses - 1) / 2
	out[["predict_type_prob"]] <- predict_type_prob
	out[["wap_weights"]] <- as.logical(wap_weights)
	
	if (wap_weights) {
		C <- calc.V(C)
	}
	
	out[["lst_comparisons"]] <- as.list(as.data.frame(combn(1:length(out[["classes"]]), 2)))
	names(out[["lst_comparisons"]]) <- NULL
	if (out[["nthreads"]] == 1) {
		out[["classifiers"]] <- lapply(out[["lst_comparisons"]],
									   function(x) fit.single.wap(x[1], x[2], X, C, classifier, ...))
	} else {
		out[["classifiers"]] <- parallel::mclapply(out[["lst_comparisons"]],
												   function(x) fit.single.wap(x[1], x[2], X, C, classifier, ...),
												   mc.cores = out[["nthreads"]])
	}
	return(structure(out, class = "wap"))
}

fit.single.wap <- function(cl1, cl2, X, C, classifier, ...) {
	y <- as.integer(C[, cl1] < C[, cl2])
	w <- abs(C[, cl1] - C[, cl2])
	valid_cases <- w > 0
	X_take <- X[valid_cases, ]
	y_take <- y[valid_cases]
	w_take <- w[valid_cases]
	w_take <- standardize.weights(w_take)
	return(classifier(X_take, y_take, weights = w_take))
}

assign.score.goodness <- function(ix, lst_comparisons, pred, scores) {
	cl1 <- lst_comparisons[[ix]][1]
	cl2 <- lst_comparisons[[ix]][2]
	
	if (NCOL(pred[[ix]]) == 1) {
		scores[, cl1] <- pred[[ix]]
		scores[, cl2] <- 1 - pred[[ix]]
	} else {
		if (is.null(colnames(pred[[ix]]))) {
			scores[, cl1] <- pred[[ix]][, 2]
			scores[, cl2] <- pred[[ix]][, 1]
		} else {
			scores[, cl1] <- pred[[ix]][, "1"]
			scores[, cl2] <- pred[[ix]][, "0"]
		}
	}
	return(scores)
}

assign.score.most.wins <- function(ix, lst_comparisons, scores, winners) {
	cl1 <- lst_comparisons[[ix]][1]
	cl2 <- lst_comparisons[[ix]][2]
	winners_this <- assign.class.most.wins(ix, lst_comparisons, winners)
	slices <- cbind(1:NROW(winners_this), winners_this)
	scores[slices] <- 1
	return(scores)
}

assign.class.most.wins <- function(ix, lst_comparisons, winners) {
	which_class <- c(lst_comparisons[[ix]][1], lst_comparisons[[ix]][2])
	return(which_class[winners[[ix]] + 1])
}

convert.prob.to.winner <- function(p) {
	if (NCOL(p) == 1) {
		return(as.integer(p >= .5))
	} else {
		return(as.integer(p[, "1"] >= .5))
	}
}

#' @title Predict method for Weighted All-Pairs
#' @description Predicts either the class with expected minimum cost or scores (more is better) for new data.
#' @param object An object of class `wap` as output by function `weighted.all.pairs`.
#' @param newdata New data on which to make predictions.
#' @param type One of "class" (will output the class with minimum expected cost) or "score"
#' (will output the predicted score for each class, i.e. more is better).
#' @param criterion One of "goodness" (will use the sum of probabilities output by each classifier) or "most-wins"
#' (will use the predicted class by each classifier).
#' @param ... Additional arguments to pass to the predict method of the base classifier.
#' @return When passing `type = "class"`, a vector with class numbers or names (if the cost matrix had them).
#' When passing `type = "score"`, will output a `matrix` with the same number of columns as `C` (passed to
#' the `weighted.all.pairs` function) and the predicted score for each observation and class.
#' @export
predict.wap <- function(object, newdata, type = "class", criterion = "most-wins", ...) {
	check.predict.type(type)
	if (!(criterion %in% c("goodness", "most-wins"))) {
		stop("'criterion' must be one of 'goodness' or 'most-wins'.")
	}
	
	if (object[["nthreads"]] == 1) {
		pred <- lapply(object[["classifiers"]],
					   function(m) predict(m, newdata, type = object[["predict_type_prob"]], ...))
	} else {
		pred <- parallel::mclapply(object[["classifiers"]],
								   function(m) predict(m, newdata, type = object[["predict_type_prob"]], ...),
								   mc.cores = object[["nthreads"]])
	}
	
	if (criterion == "goodness") {
		scores <- matrix(0, nrow = NROW(newdata), ncol = length(object[["classes"]]))
		if (object[["nthreads"]] == 1) {
			scores <- lapply(1:object[["ncombs"]],
							 function(ix) assign.score.goodness(ix, object[["lst_comparisons"]],
							 								   pred, scores))
		} else {
			scores <- parallel::mclapply(1:object[["ncombs"]],
										 function(ix) assign.score.goodness(ix, object[["lst_comparisons"]],
										 								   pred, scores), mc.cores = object[["nthreads"]])
		}
		scores <- Reduce(function(a, b) {return(a + b)}, scores)
		if (type == "class") {
			return(object[["classes"]][apply(scores, 1, which.max)])
		} else {
			scores <- scores / length(object[["classes"]])
			colnames(scores) <- object[["classes"]]
			return(scores)
		}
	} else {
		winners <- lapply(pred, convert.prob.to.winner)
		
		if (type == "class") {
			if (object[["nthreads"]] == 1) {
				winners <- lapply(1:object[["ncombs"]],
								  function(ix) assign.class.most.wins(ix, object[["lst_comparisons"]], winners))
			} else {
				winners <- parallel::mclapply(1:object[["ncombs"]],
											  function(ix) assign.class.most.wins(ix, object[["lst_comparisons"]], winners),
											  mc.cores = object[["nthreads"]])
			}
			winners <- as.data.frame(winners)
			return(object[["classes"]][apply(winners, 1, function(x) which.max(table(x)))])
		} else {
			scores <- matrix(0, nrow = NROW(newdata), ncol = length(object[["classes"]]))
			if (object[["nthreads"]] == 1) {
				scores <- lapply(1:object[["ncombs"]], function(ix) assign.score.most.wins(ix, object[["lst_comparisons"]],
																						   scores, winners))
			} else {
				scores <- parallel::mclapply(1:object[["ncombs"]],
											 function(ix) assign.score.most.wins(ix, object[["lst_comparisons"]],
											 									scores, winners), mc.cores = object[["nthreads"]])
			}
			scores <- Reduce(function(a, b) {return(a + b)}, scores)
			scores <- scores / length(object[["classes"]])
			colnames(scores) <- object[["classes"]]
			return(scores)
		}
	}
}

#' @title Get information about Weighted All-Pairs object
#' @description Prints basic information about a `wap` object
#' (Number of classes and classifiers, classifier class).
#' @param x An object of class "wap".
#' @param ... Extra arguments (not used).
#' @export
print.wap <- function(x, ...) {
	cat("Weighted All-Pairs (cost-sensitive classifier)\n\n")
	cat("Classifier type:", class(x[["classifiers"]][[1]]), "\n")
	if (!x[["wap_weights"]]) {
		cat("Using simple weight differences\n")
	}
	cat("Number of classes:", length(x[["classes"]]), "\n")
	cat("Number of classifiers:", length(x[["classifiers"]]), "\n")
	cat("Class names:", head(x[["classes"]]), "\n")
}

#' @title Get information about Weighted All-Pairs object
#' @description Prints basic information about a `wap` object
#' (Number of classes and classifiers, classifier class). Same as function `print`.
#' @param object An object of class "wap".
#' @param ... Extra arguments (not used).
#' @export
summary.wap <- function(object, ...) {
	print.wap(object)
}

