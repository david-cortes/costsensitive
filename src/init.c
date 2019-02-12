#include <R_ext/Rdynload.h>
#include <R.h>
#include <Rinternals.h>

extern SEXP r_calc_v(SEXP C, SEXP V, SEXP nrow, SEXP ncol);

static const R_CallMethodDef callMethods [] = {
    { "r_calc_v", (DL_FUNC) &r_calc_v, 4 },
    { NULL, NULL, 0 }
}; 

void
R_init_costsensitive(DllInfo *info)
{
   R_registerRoutines(info, NULL, callMethods, NULL, NULL);
   R_useDynamicSymbols(info, TRUE);
}

