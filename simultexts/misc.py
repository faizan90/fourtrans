'''
Created on Feb 4, 2019

@author: Faizan-Uni
'''

import numpy as np
import pandas as pd
from scipy.stats import norm, mvn

print_line_str = 40 * '#'


def print_sl():

    print(2 * '\n', print_line_str, sep='')
    return


def print_el():

    print(print_line_str)
    return


def ret_mp_idxs(n_vals, n_cpus):

    assert n_vals > 0

    idxs = np.linspace(
        0, n_vals, min(n_vals + 1, n_cpus + 1), endpoint=True, dtype=np.int64)

    idxs = np.unique(idxs)

    assert idxs.shape[0]

    if idxs.shape[0] == 1:
        idxs = np.concatenate((np.array([0]), idxs))

    assert (idxs[0] == 0) & (idxs[-1] == n_vals), idxs
    return idxs


def get_daily_annual_cycles_df(in_data_df):

    '''Given full time series dataframe, get daily annual cycle dataframe
    for all columns.
    '''

    annual_cycle_df = pd.DataFrame(
        index=in_data_df.index, columns=in_data_df.columns, dtype=float)

    cat_ser_gen = (in_data_df[col].copy() for col in in_data_df.columns)

    for col_ser in cat_ser_gen:
        annual_cycle_df.update(get_daily_annual_cycle(col_ser))

    assert np.all(np.isfinite(annual_cycle_df.values))
    return annual_cycle_df


def get_daily_annual_cycle(col_ser):

    '''Given full time series series, get daily annual cycle series.
    '''

    assert isinstance(col_ser, pd.Series), 'Expected a pd.Series object!'
    col_ser.dropna(inplace=True)

    # For each day of a year, get the days for all year and average them
    # the annual cycle is the average value and is used for every doy of
    # all years
    for month in range(1, 13):
        month_idxs = col_ser.index.month == month
        for dom in range(1, 32):
            dom_idxs = col_ser.index.day == dom
            idxs_intersect = np.logical_and(month_idxs, dom_idxs)
            curr_day_vals = col_ser.values[idxs_intersect]

            if not curr_day_vals.shape[0]:
                continue

            curr_day_avg_val = curr_day_vals.mean()
            col_ser.loc[idxs_intersect] = curr_day_avg_val
    return col_ser


def get_mvn_cdf_val(corr, ep):

    mu_mat = np.array([0., 0.])

    corr_mat = np.array(
        [[1., corr],
         [corr, 1.]])

    p_max = 0.999999999
    assert ep < p_max

    low = np.full((2,), norm.ppf(ep))
    upp = np.full(low.shape, norm.ppf(p_max))

    gau_cdf_val, _ = mvn.mvnun(low, upp, mu_mat, corr_mat)
    return gau_cdf_val


def get_mvn_corr(pab, ep, n_tries, thresh):

    cr1 = -0.9999999
    cr2 = +0.9999999

    for _ in range(n_tries):
        cr3 = (cr1 + cr2) / 2.0

        p = get_mvn_cdf_val(cr3, ep)

        if p > pab:
            cr2 = cr3

        else:
            cr1 = cr3

        if abs(cr1 - cr2) <= thresh:
            break

    return cr3


def get_mvn_corr_mat_for_indic_corr_mat(ind_corrs, ep):

    n_tries = 200
    thresh = 1e-8

    corr_mat = np.ones_like(ind_corrs)

    upp_idxs = []
    low_idxs = []
    for i in range(ind_corrs.shape[0]):
        for j in range(ind_corrs.shape[1]):
            if i > j:
                low_idxs.append((i, j))

            if j > i:
                upp_idxs.append((i, j))

    for i in range(len(upp_idxs)):

        ind_corr = ind_corrs[upp_idxs[i][0], upp_idxs[i][1]]

        # Prob of same value of bivariate binomial random variable.
        pab = (ep * (1 - ep) * ind_corr) + ((1 - ep) ** 2)

        mvn_corr = get_mvn_corr(pab, ep, n_tries, thresh)

        corr_mat[upp_idxs[i][0], upp_idxs[i][1]] = mvn_corr
        corr_mat[low_idxs[i][0], low_idxs[i][1]] = mvn_corr

    return corr_mat

