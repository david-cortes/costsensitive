#include <R.h>
#include <R_ext/Print.h>
#include <Rinternals.h>
#include <Rmath.h>
#include <stdlib.h>
#include <stddef.h>

int calculate_V(double C[], double V[], size_t nrow, size_t ncol, int nthreads);

SEXP r_calc_v(SEXP C, SEXP V, SEXP nrow, SEXP ncol)
{
	size_t n_row = (size_t) INTEGER(nrow)[0];
	size_t n_col = (size_t) INTEGER(ncol)[0];
	
	int failed = calculate_V(REAL(C), REAL(V), n_row, n_col, 1);
	if (failed) {
		REprintf("Could not allocate enough memory for the procedure");
	}
	return R_NilValue;
}
