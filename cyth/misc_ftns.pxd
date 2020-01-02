# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=False
# cython: embedsignature=True

ctypedef double DT_D
ctypedef unsigned long DT_UL

cpdef tuple get_asymms_sample(DT_D[:] u, DT_D[:] v) except +

cpdef DT_D get_asymm_1_sample(DT_D[:] u, DT_D[:] v) except +

cpdef DT_D get_asymm_2_sample(DT_D[:] u, DT_D[:] v) except +
