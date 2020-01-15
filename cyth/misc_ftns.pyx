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


cpdef void fill_bi_var_cop_dens(
        DT_D[:] x_probs, DT_D[:] y_probs, DT_D[:, ::1] emp_dens_arr) except +:

    '''Fill the bivariate empirical copula'''

    cdef:
        Py_ssize_t i, j
        Py_ssize_t tot_pts = x_probs.shape[0]
        Py_ssize_t n_cop_bins = emp_dens_arr.shape[0]

        Py_ssize_t tot_sum, i_row, j_col

        DT_D div_cnst, u1, u2

    assert x_probs.size == y_probs.size

    assert emp_dens_arr.shape[0] == emp_dens_arr.shape[1]

    for i in range(n_cop_bins):
        for j in range(n_cop_bins):
            emp_dens_arr[i, j] = 0.0

    tot_sum = 0
    for i in xrange(tot_pts):
        u1 = x_probs[i]
        u2 = y_probs[i]

        i_row = <Py_ssize_t> (u1 * n_cop_bins)
        j_col = <Py_ssize_t> (u2 * n_cop_bins)

        emp_dens_arr[i_row, j_col] += 1
        tot_sum += 1

    assert tot_pts == tot_sum

    div_cnst = (n_cop_bins**2) / float(tot_pts)

    for i in range(n_cop_bins):
        for j in range(n_cop_bins):
            emp_dens_arr[i, j] = emp_dens_arr[i, j] * div_cnst

    return
