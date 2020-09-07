'''
Created on Feb 4, 2019

@author: Faizan-Uni
'''

import numpy as np
import pandas as pd

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

