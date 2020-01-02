cpdef tuple get_asymms_sample(DT_D[:] u, DT_D[:] v) except +:

    cdef:
        Py_ssize_t i, n_vals
        DT_D asymm_1, asymm_2

    n_vals = u.shape[0]

    asymm_1 = 0.0
    asymm_2 = 0.0

    for i in range(n_vals):
        asymm_1 += (u[i] + v[i] - 1.0)**3
        asymm_2 += (u[i] - v[i])**3

    asymm_1 = asymm_1 / n_vals
    asymm_2 = asymm_2 / n_vals

    return asymm_1, asymm_2


cpdef DT_D get_asymm_1_sample(DT_D[:] u, DT_D[:] v) except +:

    cdef:
        Py_ssize_t i, n_vals
        DT_D asymm_1

    n_vals = u.shape[0]

    asymm_1 = 0.0
    for i in range(n_vals):
        asymm_1 += (u[i] + v[i] - 1.0)**3

    asymm_1 = asymm_1 / n_vals

    return asymm_1


cpdef DT_D get_asymm_2_sample(DT_D[:] u, DT_D[:] v) except +:

    cdef:
        Py_ssize_t i, n_vals
        DT_D asymm_2

    n_vals = u.shape[0]

    asymm_2 = 0.0

    for i in range(n_vals):
        asymm_2 += (u[i] - v[i])**3

    asymm_2 = asymm_2 / n_vals

    return asymm_2