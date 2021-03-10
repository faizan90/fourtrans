'''
@author: Faizan-Uni-Stuttgart

Mar 3, 2021

5:16:18 PM

'''
import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path

import numpy as np
import pandas as pd
from KDEpy import FFTKDE
import matplotlib.pyplot as plt
from scipy.stats import rankdata

asymms_exp = 3.0

np.seterr(divide='raise')

plt.ioff()

DEBUG_FLAG = True


def get_mag_and_phs_spec(data):

    ft = np.fft.rfft(data)

    mag_spec = np.abs(ft)
    phs_spec = np.angle(ft)

    return mag_spec, phs_spec


def get_irfted_vals(mag_spec, phs_spec):

    ft = np.full(mag_spec.size, np.nan, dtype=complex)

    ft.real = mag_spec * np.cos(phs_spec)
    ft.imag = mag_spec * np.sin(phs_spec)

    ift = np.fft.irfft(ft)

    return ift


def roll_real_2arrs(arr1, arr2, lag, rerank_flag=False):

    assert isinstance(arr1, np.ndarray)
    assert isinstance(arr2, np.ndarray)

    assert arr1.ndim == 1
    assert arr2.ndim == 1

    assert arr1.size == arr2.size

    assert isinstance(lag, (int, np.int64))
    assert abs(lag) < arr1.size

    if lag > 0:
        # arr2 is shifted ahead
        arr1 = arr1[:-lag].copy()
        arr2 = arr2[+lag:].copy()

    elif lag < 0:
        # arr1 is shifted ahead
        arr1 = arr1[+lag:].copy()
        arr2 = arr2[:-lag].copy()

    else:
        pass

    assert arr1.size == arr2.size

    if rerank_flag:
#         assert np.all(arr1 > 0) and np.all(arr2 > 0)
#         assert np.all(arr1 < 1) and np.all(arr2 < 1)

        arr1 = rankdata(arr1) / (arr1.size + 1.0)
        arr2 = rankdata(arr2) / (arr2.size + 1.0)

    return arr1, arr2


def get_fit_kde(data):

#     rand_noise = -1e-10 + (2 * 1e-10 * np.random.random(data.shape))

    kde = FFTKDE().fit(data)

    new_data, dens = kde.evaluate(100)

    return new_data, dens


def get_binned_ts(probs, n_bins):

    assert np.all(probs > 0) and np.all(probs < 1)

    assert n_bins > 1

    bin_idxs_ts = (probs * n_bins).astype(int)

    assert np.all(bin_idxs_ts >= 0) and np.all(bin_idxs_ts < n_bins)

    return bin_idxs_ts


def get_binned_dens_ftn_1d(bin_idxs_ts):

#     bin_freqs = np.unique(bin_idxs_ts, return_counts=True)[1]

#     bin_dens = bin_freqs / bin_idxs_ts.size

    bin_idxs, bin_freqs = np.unique(bin_idxs_ts, return_counts=True)
    bin_dens = bin_freqs * (1 / bin_idxs.size)

    return bin_dens


def get_binned_dens_ftn_2d(probs_1, probs_2, n_bins):

    bins = np.linspace(0.0, 1.0, n_bins + 1)

    bin_freqs_12 = np.histogram2d(probs_1, probs_2, bins=bins)[0]

#     bin_dens_12 = bin_freqs_12 / probs_1.size

    bin_dens_12 = bin_freqs_12 * ((1 / n_bins) ** 2)

    return bin_dens_12


def get_local_entropy(probs_1, probs_2, n_bins):

    bin_idxs_ts_1 = get_binned_ts(probs_1, n_bins)
    bin_idxs_ts_2 = get_binned_ts(probs_2, n_bins)

    bin_dens_1 = get_binned_dens_ftn_1d(bin_idxs_ts_1)
    bin_dens_2 = get_binned_dens_ftn_1d(bin_idxs_ts_2)

    bin_dens_12 = get_binned_dens_ftn_2d(probs_1, probs_2, n_bins)

#     etpy_local = np.empty_like(bin_idxs_ts_1, dtype=float)
#     for i in range(bin_idxs_ts_1.shape[0]):
#
#         dens = bin_dens_12[bin_idxs_ts_1[i], bin_idxs_ts_2[i]]
#
#         if not dens:
#             etpy_local[i] = 0
#
#         else:
#             prod = bin_dens_1[bin_idxs_ts_1[i]] * bin_dens_2[bin_idxs_ts_2[i]]
#             etpy_local[i] = -(dens * np.log(dens / prod))

    dens = bin_dens_12[bin_idxs_ts_1, bin_idxs_ts_2]
    prods = bin_dens_1[bin_idxs_ts_1] * bin_dens_2[bin_idxs_ts_2]

    dens_idxs = dens.astype(bool)

    etpy_local = np.zeros_like(bin_idxs_ts_1, dtype=float)

    etpy_local[dens_idxs] = -dens[dens_idxs] * np.log(
        dens[dens_idxs] / prods[dens_idxs])

    return etpy_local


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Testings\fourtrans_practice\phsann')
    os.chdir(main_dir)

    data_file = Path(
        r'P:\Synchronize\IWS\Testings\fourtrans_practice\phsann\neckar_norm_cop_infill_discharge_1961_2015_20190118.csv')

    beg_time = '2000-01-01'
    end_time = '2010-12-31'

    col = '427'

    max_lags = 2

#     n_sims = 1

    n_bins = 100

    data = pd.read_csv(
        data_file, sep=';', index_col=0).loc[beg_time:end_time, col].values

    if data.size % 2:
        data = data[:-1]

    assert np.all(np.isfinite(data))

#     lags = []
#     etpys = []
#     for lag in range(1, max_lags + 1):
#
#         probs_2, probs_1 = roll_real_2arrs(data, data, lag, True)
#
#         etpy_lcl = get_local_entropy(probs_1, probs_2, n_bins)
#
#         lags.append(lag)
#         etpys.append(etpy_lcl.sum())
#
#     plt.plot(lags, etpys)
#
#     plt.show()

    probs_2, probs_1 = roll_real_2arrs(data, data, max_lags, True)

    etpy_lcl = get_local_entropy(probs_1, probs_2, n_bins)

    ax1 = plt.subplots(1, 1, figsize=(10, 10))[1]

    ax1.plot(data, label='data', c='r', alpha=0.7)

    ax1.legend(loc=1)
    ax1.grid()
    ax1.set_axisbelow(True)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Discharge')

    ax2 = ax1.twinx()
    ax2.plot(etpy_lcl, label='etpy', c='k', alpha=0.7)

    ax2.legend(loc=2)
    ax2.set_ylabel('Entropy')

    plt.show()

#     data_1, data_2 = roll_real_2arrs(data, data, lags, False)
#     marg_1, dens_1 = get_fit_kde(data_1)
#     marg_2, dens_2 = get_fit_kde(data_2)
#
#     marg_12, dens_12 = get_fit_kde(np.concatenate(
#         [data_1.reshape(-1, 1), data_2.reshape(-1, 1)], axis=1))
#
#     dens_1_rep = np.repeat(dens_1, dens_12.shape[0] // dens_1.size)
#     dens_2_til = np.tile(dens_2, dens_12.shape[0] // dens_2.size)
#
#     dens_1_rep[dens_1_rep < min_dens] = 0
#     dens_2_til[dens_2_til < min_dens] = 0
#
#     dens_12[dens_12 < min_dens] = 0
#
#     take_idxs = np.ones_like(dens_1_rep, dtype=bool)
#     take_idxs &= dens_1_rep.astype(bool)
#     take_idxs &= dens_2_til.astype(bool)
#     take_idxs &= dens_12.astype(bool)
#
#     dens_1_rep = dens_1_rep[take_idxs]
#     dens_2_til = dens_2_til[take_idxs]
#     dens_12 = dens_12[take_idxs]
#
#     etpy_arr = (dens_12 * np.log2(dens_12 / (dens_1_rep * dens_2_til))).sum()
#
#     x = 0

#     mag_spec_1, _ = get_mag_and_phs_spec(data_1)
#     mag_spec_2, _ = get_mag_and_phs_spec(probs_2)
#
#     data_mag_spec, data_phs_spec = get_mag_and_phs_spec(data)
#
#     data_sorted = np.sort(data)
#
#     data_ai = pim.active_info(data, k=k, local=True)[0]
#
#     data_ai_mag_spec, data_ai_phs_spec = get_mag_and_phs_spec(data_ai)
#
#     periods = (data_ai.size) / (
#         np.arange(1, data_ai_mag_spec.size - 1))
#
#     periods = np.concatenate(([data_ai.size * 2], periods))
#
#     plt.figure(figsize=(10, 6))
#
#     for i in range(n_sims):
#         rand_phs_spec = (
#             -np.pi + (2 * np.pi * np.random.random(data_phs_spec.size)))
#
#         rand_phs_spec[+0] = data_phs_spec[+0]
#         rand_phs_spec[-1] = data_phs_spec[-1]
#
#         rand_data = get_irfted_vals(data_mag_spec, rand_phs_spec)
#
#         rand_data = data_sorted[np.argsort(np.argsort(rand_data))]
#
#         rand_ai = pim.active_info(rand_data, k=k, local=True)[0]
#
#         rand_ai_mag_spec, rand_ai_phs_spec = get_mag_and_phs_spec(rand_ai)
#
#         plt.semilogx(
#             periods, rand_ai_mag_spec[1:].cumsum(), alpha=0.5, lw=1, c='b')
#
#     plt.semilogx(
#         periods, data_ai_mag_spec[1:].cumsum(), alpha=0.75, lw=2, c='r')
#
#     plt.grid()
#     plt.gca().set_axisbelow(True)
#
#     plt.xlabel('Time step')
#     plt.ylabel('Mag spec')
#
#     plt.xlim(plt.xlim()[::-1])
#
#     plt.show()

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
