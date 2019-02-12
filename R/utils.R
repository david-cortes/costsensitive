extract.info <- function(C, nthreads) {
	nclasses <- NCOL(C)
	out <- list()
	if ("data.frame" %in% class(C)) {
		out[["classes"]] <- names(C)
	} else if ("matrix" %in% class(C)) {
		classes <- colnames(C)
		if (is.null(classes)) {
			out[["classes"]] <- 1:nclasses
		} else {
			out[["classes"]] <- classes
		}
	} else {
		out[["classes"]] <- 1:nclasses
	}
	out[["nthreads"]] <- check.nthreads(nthreads, nclasses)
	return(out)
}

standardize.weights <- function(w) {
	return(w * NROW(w) / sum(w))
}

check.predict.type <- function(type) {
	if (!(type %in% c("class", "score"))) {
		stop("'type' must be one of 'class' or 'score'.")
	}
}

check.nthreads <- function(nthreads, nclasses) {
	if (is.null(nthreads)) {
		nthreads <- 1
	} else if (is.na(nthreads)) {
		nthreads <- 1
	} else if (nthreads == "auto") {
		nthreads <- parallel::detectCores()
	} else if (nthreads < 1) {
		nthreads <- 1
	}
	nthreads <- as.integer(min(nthreads, nclasses))
	if (nthreads > 1 && Sys.info()[['sysname']] == "Windows") {
		warning("Multi-threading not available on Windows systems.")
		nthreads <- 1L
	}
	return(nthreads)
}

calc.V <- function(C) {
	C_mat = as.vector(t(as.matrix(C)))
	V_mat = rep(0.0, NROW(C) * NCOL(C))
	.Call("r_calc_v", C_mat, V_mat, NROW(C), NCOL(C))
	return(matrix(V_mat, nrow = NROW(C), ncol = NCOL(C), byrow = TRUE))
}
