import numpy as np
cimport numpy as np

cdef extern from "../src/calcv.c":
	void calculate_V(double *C, double *V, size_t nrow, size_t ncol, int nthreads)

def c_calc_v(np.ndarray[double, ndim=2] C, int nthreads=1):
	cdef np.ndarray[double, ndim=2] V = np.empty((C.shape[0], C.shape[1]))
	calculate_V(&C[0,0], &V[0,0], C.shape[0], C.shape[1], nthreads)
	return V
