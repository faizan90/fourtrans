# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3

import numpy as np
from libc.stdlib cimport malloc, free
from cython.parallel import prange, parallel

cimport numpy as np
cimport cython


cdef inline void discreteHaarWaveletTransform(
        double[:] x, double *xx, int N, double div, double[:] output) nogil:

    cdef int i, j

    for i in range(N):
        xx[i] = output[i] = x[i]

    while N > 1:
        N >>= 1

        for i in range(N):
            j = i << 1
            output[i] = (xx[j] + xx[j + 1]) / div
            output[N + i] = (xx[j] - xx[j + 1]) / div

        for i in range(N):
            xx[i] = output[i]

    return


cdef inline void inversediscreteHaarWaveletTransform(
        double[:] gx, double *hx, int N, double div, double[:] output) nogil:

    cdef int i, j, k, n = 1

    for i in range(N):
        hx[i] = output[i] = gx[i]

    while n < N:
        for i in range(n):
            j = i << 1
            k = i + n

            hx[j] = (output[i] + output[k]) / div
            hx[j+1] = (output[i] - output[k]) / div

        n <<= 1

        for i in range(n):
            output[i] = hx[i]
    return

def haar_trans_1d(double[::1] in_arr):

    cdef int N = in_arr.shape[0]
    cdef double *xx

    cdef double[::1] output = np.empty_like(in_arr)

    cdef double div = 2.0 ** .5

    xx = <double *> malloc(sizeof(double) * N)

    discreteHaarWaveletTransform(in_arr, xx, N, div, output)
    free(xx)
    return np.asarray(output)


def haar_trans2d(double[:, :] image):

    cdef int i, M, N = len(image)
    cdef double div = 2.0 ** .5
    cdef double[:, ::1] output = np.empty_like(image)
    xx = <double *> malloc(sizeof(double) * N)

    M = image.shape[1]

    assert N == M

    assert np.isclose(np.log2(N), int(np.log2(N)))

    with nogil:
        for i in prange(N):
            discreteHaarWaveletTransform(image[i],
                                         xx,
                                         N,
                                         div,
                                         output[i])

        for i in prange(N):
            discreteHaarWaveletTransform(output[:, i],
                                         xx,
                                         N,
                                         div,
                                         output[:, i])

    free(xx)
    return np.asarray(output)


def haar_inv2d(double[:, :] haar):

    cdef int i, N = len(haar)
    cdef double div = 2.0 ** .5
    cdef double[:, :] output = np.empty_like(haar)
    hx = <double *> malloc(sizeof(double) * N)

    with nogil:
        for i in prange(N):
            inversediscreteHaarWaveletTransform(
                haar[:, i], hx, N, div, output[:, i])

        for i in prange(N):
            inversediscreteHaarWaveletTransform(
                output[i], hx, N, div, output[i])

    free(hx)
    return np.asarray(output)
