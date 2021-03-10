'''
@author: Faizan-Uni-Stuttgart

Feb 25, 2021

4:16:28 PM

'''
import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path

DEBUG_FLAG = False

import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm
import matplotlib.pyplot as plt

from phsann.misc import roll_real_2arrs
from phsann.cyth import get_asymms_sample, fill_bi_var_cop_dens

asymms_exp = 3.0

plt.ioff()


def get_asymm_1_max(scorr):

    a_max = (
        0.5 * (1 - scorr)) * (1 - ((0.5 * (1 - scorr)) ** (1.0 / asymms_exp)))

    return a_max


def get_asymm_2_max(scorr):

    a_max = (
        0.5 * (1 + scorr)) * (1 - ((0.5 * (1 + scorr)) ** (1.0 / asymms_exp)))

    return a_max


def get_etpy_min(n_bins):

#     dens = 1 / n_bins
#
#     etpy = -np.log(dens)

    etpy = 0

    return etpy


def get_etpy_max(n_bins):

    dens = (1 / (n_bins ** 2))

    etpy = -np.log(dens)

    return etpy


def get_probs_copula_props(probs, probs_rolled, n_ecop_bins):

    scorr = np.corrcoef(probs, probs_rolled)[0, 1]

    asymm_1, asymm_2 = get_asymms_sample(probs, probs_rolled)
    asymm_1 /= get_asymm_1_max(scorr)
    asymm_2 /= get_asymm_2_max(scorr)

#     plt.scatter(probs, probs_rolled, alpha=0.5)
#     plt.grid()
#     plt.show()
#     plt.close()

    ecop_dens_arr = np.full(
        (n_ecop_bins, n_ecop_bins),
        np.nan,
        dtype=np.float64)

    fill_bi_var_cop_dens(probs, probs_rolled, ecop_dens_arr)

    non_zero_idxs = ecop_dens_arr > 0

    dens = ecop_dens_arr[non_zero_idxs]

    etpy_arr = -(dens * np.log(dens))

    etpy = etpy_arr.sum()

    etpy_min = get_etpy_min(n_ecop_bins)
    etpy_max = get_etpy_max(n_ecop_bins)

    etpy = (etpy - etpy_min) / (etpy_max - etpy_min)

    return asymm_1, asymm_2, etpy, scorr


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Testings\fourtrans_practice')
    os.chdir(main_dir)

    data_file = Path(
        r'P:\Synchronize\IWS\Testings\fourtrans_practice\phsann\neckar_norm_cop_infill_discharge_1961_2015_20190118.csv')

    beg_time = '1961-01-01'
    end_time = '2015-12-31'

    lag_steps = np.arange(1, 60, dtype=np.int64)
    ecop_bins = 20

    data_ser = pd.read_csv(
        data_file, sep=';', index_col=0).loc[beg_time:end_time, '420']

    data = data_ser.values

    if (data.size % 2):
        data = data[:-1]

    probs = rankdata(data) / (data.size + 1.0)

    old_props = []
    for lag_step in lag_steps:
        probs_i, probs_i_rolled = roll_real_2arrs(
                probs, probs, lag_step)

        old_props.append(
            get_probs_copula_props(probs_i, probs_i_rolled, ecop_bins))

    old_props = np.array(old_props)

    new_props = []
    for lag_step in lag_steps:
        probs_i, probs_i_rolled = roll_real_2arrs(
                probs, probs, lag_step)

        probs_i = rankdata(probs_i) / (probs_i.size + 1.0)
        probs_i_rolled = rankdata(probs_i_rolled) / (probs_i_rolled.size + 1.0)

        new_props.append(
            get_probs_copula_props(probs_i, probs_i_rolled, ecop_bins))

    new_props = np.array(new_props)

    axs = plt.subplots(2, 2, figsize=(10, 10), sharex=True)[1]

    prop_labs = ['$AO$', '$AD$', '$H$', '$\\rho_s$']
    for i in range(4):
        ax = axs.ravel()[i]

        ax.plot(lag_steps, old_props[:, i], c='r', alpha=0.7, label='old')
        ax.plot(lag_steps, new_props[:, i], c='b', alpha=0.7, label='new')

        ax.set_ylabel(prop_labs[i])

        ax.grid()
        ax.set_axisbelow(True)

        ax.legend()

    plt.tight_layout()
    plt.savefig('roll_real_props_cmpr.png', bbox_inches='tight')

    plt.close()
    return


if __name__ == '__main__':
    print('#### Started on %s ####\n' % time.asctime())
    START = timeit.default_timer()

    #==========================================================================
    # When in post_mortem:
    # 1. "where" to show the stack
    # 2. "up" move the stack up to an older frame
    # 3. "down" move the stack down to a newer frame
    # 4. "interact" start an interactive interpreter
    #==========================================================================

    if DEBUG_FLAG:
        try:
            main()

        except:
            pre_stack = tb.format_stack()[:-1]

            err_tb = list(tb.TracebackException(*sys.exc_info()).format())

            lines = [err_tb[0]] + pre_stack + err_tb[2:]

            for line in lines:
                print(line, file=sys.stderr, end='')

            import pdb
            pdb.post_mortem()
    else:
        main()

    STOP = timeit.default_timer()
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))
