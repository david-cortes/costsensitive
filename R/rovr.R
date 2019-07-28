#' @title Regression One-Vs-Rest
#' @description Creates a cost-sensitive classifier by creating one regressor per
#' class to predict cost. Takes as input a regressor rather than a classifier.
#' The objective is to create a model that would predict the
#' class with the minimum cost.
#' @param X The data (covariates/features).
#' @param C matrix(n_samples, n_classes) Costs for each class for each observation.
#' @param regressor function(X, y, ...) -> object, that would create regressor with method `predict`.
#' @param nthreads Number of parallel threads to use (not available on Windows systems). Note
#' that, unlike the Python version, this is not a shared memory model and each additional thread will
#' require more memory from the system. Not recommended to use when the algorithm is itself parallelized.
#' @param ... Extra arguments to pass to `regressor`.
#' @references Beygelzimer, A., Langford, J., & Zadrozny, B. (2008). Machine learning techniques-reductions between prediction quality metrics.
#' @export
#' @examples
#' library(costsensitive)
#' wrapped.lm <- function(X, y, ...) {
#' 	return(lm(y ~ ., data = X, ...))
#' }
#' set.seed(1)
#' X <- data.frame(feature1 = rnorm(100), feature2 = rnorm(100), feature3 = runif(100))
#' C <- data.frame(cost1 = rgamma(100, 1), cost2 = rgamma(100, 1), cost3 = rgamma(100, 1))
#' model <- regression.one.vs.rest(X, C, wrapped.lm)
#' predict(model, X, type = "class")
#' predict(model, X, type = "score")
#' print(model)
regression.one.vs.rest <- function(X, C, regressor, nthreads = 1, ...) {
	out <- extract.info(C, nthreads)
	nclasses <- length(out[["classes"]])
	if (out[["nthreads"]] == 1) {
		out[["regressors"]] <- lapply(1:nclasses, function(cl) regressor(X, C[, cl], ...))
	} else {
		out[["regressors"]] <- parallel::mclapply(1:nclasses, function(cl) regressor(X, C[, cl], ...), mc.cores = out[["nthreads"]])
	}
	return(structure(out, class = "rovr"))
}

#' @title Predict method for Regression One-Vs-Rest
#' @description Predicts either class with expected minimum cost or the expected cost (less is better) for new data.
#' @param object An object of class `rovr` as output by function `regression.one.vs.rest`.
#' @param newdata New data on which to make predictions.
#' @param type One of "class" (will output the class with minimum expected cost) or "score"
#' (will output the predicted cost for each class, i.e. less is better).
#' @param ... Additional arguments to pass to the predict method of the base regressor.
#' @return When passing `type = "class"`, a vector with class numbers or names (if the cost matrix had them).
#' When passing `type = "score"`, will output a `data.frame` with the same number of columns as `C` (passed to
#' the `regression.one.vs.rest` function) and the predicted cost for each observation and class.
#' @export
predict.rovr <- function(object, newdata, type = "class", ...) {
	check.predict.type(type)
	if (object[["nthreads"]] == 1) {
		pred <- as.data.frame(lapply(object[["regressors"]], function(m) predict(m, newdata, ...)))
	} else {
		pred <- as.data.frame(parallel::mclapply(object[["regressors"]], function(m) predict(m, newdata, ...),
												 mc.cores = object[["nthreads"]]))
	}
	if (type == "class") {
		return(object[["classes"]][apply(pred, 1, which.min)])
	} else {
		names(pred) <- object[["classes"]]
		return(pred)
	}
}

#' @title Get information about Regression One-Vs-Rest object
#' @description Prints basic information about a `rovr` object
#' (Number of classes, regressor class).
#' @param x An object of class "rovr".
#' @param ... Extra arguments (not used).
#' @export
print.rovr <- function(x, ...) {
	cat("Regression One-vs-Rest (cost-sensitive classifier)\n\n")
	cat("Regressor type:", class(x[["regressors"]][[1]]), "\n")
	cat("Number of classes:", length(x[["classes"]]), "\n")
	cat("Class names:", head(x[["classes"]]), "\n")
}

#' @title Get information about Regression One-Vs-Rest object
#' @description Prints basic information about a `rovr` object
#' (Number of classes, regressor class). Same as function `print`.
#' @param object An object of class "rovr".
#' @param ... Extra arguments (not used).
#' @export
summary.rovr <- function(object, ...) {
	print.rovr(object)
}

