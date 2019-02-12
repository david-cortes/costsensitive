#' @title Cost-Proportionate Classifier
#' @description Fits a classifier with sample weights by reducing the problem to classification
#' without sample weights through rejection sampling.
#' @param X Features/Covariates for each observation.
#' @param y Class for each observation.
#' @param weights Weights for each observation.
#' @param classifier Base classifier to use.
#' @param nsamples Number of resamples to take.
#' @param extra_rej_const Extra rejection constant - the higher, the smaller each sample ends up being,
#' but the smallest the chance that the highest-weighted observations would end up in each sample.
#' @param nthreads Number of parallel threads to use (not available on Windows systems). Note
#' that, unlike the Python version, this is not a shared memory model and each additional thread will
#' require more memory from the system. Not recommended to use when the algorithm is itself parallelized.
#' @param seed Random seed to use for the random number generation.
#' @param ... Additional arguments to pass to `classifier`.
#' @references Beygelzimer, A., Langford, J., & Zadrozny, B. (2008). Machine learning techniques-reductions between prediction quality metrics.
#' @examples
#' \dontrun{
#' library(costsensitive)
#' data(iris)
#' set.seed(1)
#' X <- iris[, c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width")]
#' y <- factor(iris$Species == "setosa", labels = c("class1", "class2"))
#' weights <- rgamma(100, 1)
#' classifier <- caret::train
#' model <- cost.proportionate.classifier(X, y, weights, classifier,
#'   method = "glm", family = "binomial",
#'   trControl=caret::trainControl(method="none"), tuneLength=1)
#' predict(model, X, aggregation = "raw", type = "raw")
#' predict(model, X, aggregation = "weighted", type = "prob")
#' }
#' @export 
cost.proportionate.classifier <- function(X, y, weights, classifier, nsamples=10, extra_rej_const=1e-1, nthreads=1, seed=1, ...) {
	
	## check inputs
	if (min(weights) <= 0) { stop("'weights' can only have positive entries.") }
	if (extra_rej_const < 0) {stop("'extra_rej_const' must be a positive small number.") }
	nsamples <- as.integer(nsamples)
	if (nsamples < 1) { stop("'nsamples' must be a positive integer.") }
	
	out <- list()
	out[["extra_rej_const"]] <- extra_rej_const
	out[["nsamples"]] <- nsamples
	out[["seed"]] <- seed
	out[["nthreads"]] <- check.nthreads(nthreads, nsamples)
	if ("data.frame" %in% class(y)) {
		out[["classes"]] <- names(y)
	} else {
		out[["classes"]] <- colnames(y)
	}
	
	Z <- max(weights) + extra_rej_const
	weights <- weights / Z #weights is now acceptance prob
	
	fit.single.sample <- function(sample_num, X, y, weights, classifier, seed, ...) {
		set.seed(seed + sample_num)
		rows_take <- runif(NROW(X), 0, 1) <= weights
		return(classifier(X[rows_take, ], y[rows_take], ...))
	}
	if (out[["nthreads"]] == 1) {
		out[["classifiers"]] <- lapply(1:nsamples, fit.single.sample, X, y, weights, classifier, seed, ...)
	} else {
		out[["classifiers"]] <- parallel::mclapply(1:nsamples, fit.single.sample, X, y, weights, classifier, seed, ...,
												   mc.cores = out[["nthreads"]])
	}
	return(structure(out, class = "costprop"))
}

#' @title Predict method for Cost-Proportionate Classifier
#' @description Predicts either the class or score according to predictions from classifiers fit to different
#' resamples each. Be aware that the base classifier with which it was built must provide appropriate outputs
#' that match with the arguments passed here (`type` and `criterion`). This is usually managed through argument
#' `type` that goes to its `predict` method.
#' @param object An object of class `costprop` as output by function `cost.proportionate.classifier`.
#' @param newdata New data on which to make predictions.
#' @param aggregation One of "raw" (will take the class according to votes from each classifier. The predictions from
#' classifiers must in turn be 1-dimensional vectors with the predicted class, not probabilities, scores, or two-dimensional
#' arrays - in package `caret` for example, this corresponds to `type = "raw"`), or "weighted" (will take a weighted
#' vote according to the probabilities or scores predicted by each classifier. The predictions from classifiers must in turn
#' be either 1-dimensional vectors with the predicted probability/score, or two-dimensional matrices with the second
#' column having the probability/score for the positive class = in package `caret` for example, this corresponds to `type = "prob`).
#' @param output_type One of "class" (will output the predicted class) or "score" (will output the predicted score).
#' @param ... Additional arguments to pass to the predict method of the base classifier.
#' @export 
#' @examples
#' \dontrun{
#' library(costsensitive)
#' data(iris)
#' set.seed(1)
#' X <- X <- iris[, c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width")]
#' y <- factor(iris$Species == "setosa", labels = c("class1", "class2"))
#' weights <- rgamma(100, 1)
#' classifier <- caret::train
#' model <- cost.proportionate.classifier(X, y, weights, classifier,
#'   method = "glm", family = "binomial",
#'   trControl=caret::trainControl(method="none"), tuneLength=1)
#' predict(model, X, aggregation = "raw", type = "raw")
#' predict(model, X, aggregation = "weighted", type = "prob")
#' }
predict.costprop <- function(object, newdata, aggregation = "raw", output_type = "score", ...) {
	
	### check inputs
	if (!(output_type %in% c("score", "class"))) {
		stop("'output_type' must be one of 'score' or 'class'.")
	}
	if (!(aggregation %in% c("raw", "weighted"))) {
		stop("'aggregation' must be one of 'raw' or 'weighted'.")
	}
	
	if (aggregation == "raw") {
		
		if (object[["nthreads"]] == 1) {
			pred <- lapply(object[["classifiers"]], predict, newdata, ...)
		} else {
			pred <- parallel::mclapply(object[["classifiers"]], predict, newdata, ..., mc.cores = object[["nthreads"]])
		}
		
		if (NCOL(pred[[1]]) > 1) {
			stop("Outputs from predic classifier of classifiers must be 1-dimensional vectors.")
		} else {
			pred <- as.data.frame(pred)
			if (output_type == "class") {
				get.most.common <- function(x) {
					tbl <- table(x, useNA = "no")
					if (length(tbl) > 2) {
						tbl <- table(x >= .5, useNA = "no")
					}
					return(names(tbl)[which.max(tbl)])
				}
				pred <- apply(pred, 1, get.most.common)
				return(pred)
			} else {
				get.score.pos <- function(x, pos_class) {
					tbl <- table(x, useNA = "no")
					if (length(tbl) > 2) {
						tbl <- table(x >= .5, useNA = "no")
					}
					if (length(tbl) == 1) {
						if (tbl[1]) {
							tbl <- c(0, tbl)
						} else {
							tbl <- c(tbl, 0)
						}
					}
					if (is.null(pos_class)) {
						return(tbl[[2]] / sum(tbl))
					} else {
						return(tbl[[pos_class]] / sum(tbl))
					}
				}
				pred <- apply(pred, 1, get.score.pos, object[["classes"]][[2]])
				return(pred)
			}
		}
		
		
	} else {
		if (object[["nthreads"]] == 1) {
			pred <- lapply(object[["classifiers"]], predict, newdata, ...)
		} else {
			pred <- parallel::mclapply(object[["classifiers"]], predict, newdata, ..., mc.cores = object[["nthreads"]])
		}
		if (NCOL(pred[[1]]) > 1) {
			if (is.null(object[["classes"]])) {
				pred <- lapply( pred, function(x) x[, 2] )
			} else {
				pred <- lapply( pred, function(x) x[, object[["classes"]][[2]]] )
			}
		}
		pred <- Reduce(function(a, b) a + b, pred) / length(object[["classifiers"]])
		
		if (output_type == "class") {
			if (is.null(object[["classes"]])) {
				return(as.integer(pred >= .5))
			} else {
				return(object[["classes"]][1 + as.integer(pred >= .5)])
			}
			
		} else {
			return(pred)
		}
		
	}
}

#' @title Get information about Cost-Proportionate classifier object
#' @description Prints basic information about a `costprop` object
#' (Number of samples, rejection constant, random seed, class names, classifier class).
#' @param x An object of class "costprop".
#' @param ... Extra arguments (not used).
#' @export
print.costprop <- function(x, ...) {
	cat("Cost-Proportionate-Sampled Classifier (weighted classifier)\n\n")
	cat("Classifier type:", class(x[["classifiers"]][[1]]), "\n")
	cat("Classes:", x[["classes"]], "\n")
	cat("Number of samples:", length(x[["classifiers"]]), "\n")
	cat("Extra rejection constant:", x[["extra_rej_const"]], "\n")
	cat("Random seed used:", x[["seed"]], "\n")
}

#' @title Get information about Cost-Proportionate classifier object
#' @description Prints basic information about a `costprop` object
#' (Number of samples, rejection constant, random seed, class names, classifier class).
#' Same as function `print`.
#' @param object An object of class "costprop".
#' @param ... Extra arguments (not used).
#' @export
summary.costprop <- function(object, ...) {
	print.costprop(object)
}
