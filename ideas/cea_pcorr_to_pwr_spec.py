'''
@author: Faizan-Uni-Stuttgart

Aug 4, 2022

9:25:05 AM

'''
import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm
import matplotlib.pyplot as plt; plt.ioff()

DEBUG_FLAG = False


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Testings\fourtrans_practice\iaaftsa')
    os.chdir(main_dir)

    in_data_file = Path(r'neckar_q_data_combined_20180713.csv')

    sep = ';'

    col = '420'

    beg_time = '1962-01-01'
    end_time = '1963-01-31'
    #==========================================================================

    df_data = pd.read_csv(in_data_file, sep=sep, index_col=0)

    data = df_data.loc[beg_time:end_time, col].values

    if data.shape[0] % 2:
        data = data[:-1]

    assert np.all(np.isfinite(data))

    lags = np.arange(data.size, dtype=np.int64)

    if False:
        probs = rankdata(data) / (data.size + 1.0)

        data = norm.ppf(probs)

    data_cens = data.copy()

    for i in range(10):
        rand_int_a = np.random.randint(0, 20)
        rand_int_b = np.random.randint(0, data.size - 20)

        data_cens[rand_int_b:rand_int_b + rand_int_a] = np.nan

    ref_pcorrs = []
    for lag in lags:

        # if lag <= 50:
        #
        #     pcorr = np.corrcoef(
        #         *roll_real_2arrs_with_nan(data, data, lag, False))[0, 1]
        #
        # else:
        #     pcorr = 0.0

        if True:
            pcorr = np.corrcoef(data, np.roll(data, lag))[0, 1]

        else:
            data_cens_roll = np.roll(data_cens, lag)

            finite_idxs = np.isfinite(data_cens_roll) & np.isfinite(data_cens)

            pcorr = np.corrcoef(
                data_cens[finite_idxs], data_cens_roll[finite_idxs])[0, 1]

        ref_pcorrs.append(pcorr)

    ref_pcorrs = np.array(ref_pcorrs)

    ref_ft = np.fft.fft(data)
    ref_mag = np.abs(ref_ft)[1:1 + data.size // 2]
    ref_pwr = ref_mag ** 2

    # ref_pwr /= ref_pwr.sum()

    ref_pwr = ref_pwr.cumsum()
    ref_pwr /= ref_pwr[-1]
    #==========================================================================

    wk_ft = np.zeros(lags.size, dtype=complex)

    cnst = 2 * np.pi * 1j

    # PCorr to WK.
    for i in range(wk_ft.size):
        for j in range(wk_ft.size):
            wk_ft[i] += ref_pcorrs[j] * np.exp((-cnst * lags[j] * i) / wk_ft.size)

    # WK to PCorr.
    wk_pcorrs = []
    for i in range(wk_ft.size):
        wk_pcorr = 0.0j

        for j in range(wk_ft.size):
            wk_pcorr += wk_ft[j] * np.exp((cnst * lags[i] * j) / wk_ft.size)

        # The sign is important to get the direction of correlation.
        if wk_pcorr.real >= 0:
            sign = +1

        else:
            sign = -1

        wk_pcorrs.append(sign * abs(wk_pcorr) / wk_ft.size)

    wk_pcorrs = np.array(wk_pcorrs)
    #==============================================================================

    print(np.corrcoef(rankdata(ref_pcorrs), rankdata(wk_pcorrs))[0, 0])

    if True:
        periods = (ref_pwr.size * 2) / np.arange(1, ref_pwr.size + 1)

        assert periods.size == ref_pwr.shape[0]

        wk_pwr = np.abs(wk_ft[1:1 + wk_ft.size // 2])

        wk_pwr = wk_pwr.cumsum()
        wk_pwr /= wk_pwr[-1]

        print(np.corrcoef(rankdata(ref_pwr), rankdata(wk_pwr))[0, 0])

        plt.semilogx(
            periods,
            wk_pwr,
            alpha=0.5,
            color='b',
            label='wk',
            lw=1.5,
            zorder=2)

        plt.semilogx(
            periods,
            ref_pwr,
            alpha=0.75,
            color='r',
            label='ref',
            lw=3.0,
            zorder=1)

        plt.xlim(plt.xlim()[::-1])

        plt.xlabel('Period')
        plt.ylabel('Cummulative power')

    else:
        plt.plot(lags, ref_pcorrs, label='pc', zorder=1, lw=2)
        plt.plot(lags, wk_pcorrs, label='wk', zorder=2, lw=1)

        plt.xlabel('Lag steps')
        plt.ylabel('Pearson correlation')

    plt.legend()

    plt.grid()
    plt.gca().set_axisbelow(True)

    plt.show()
    return


def roll_real_2arrs_with_nan(arr1, arr2, lag, rerank_flag=False):

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
        arr1 = arr1[-lag:].copy()
        arr2 = arr2[:+lag].copy()

    else:
        pass

    assert arr1.size == arr2.size

    not_nan_idxs = np.isfinite(arr1) & np.isfinite(arr2)

    arr1, arr2 = arr1[not_nan_idxs], arr2[not_nan_idxs]

    if rerank_flag:
#         assert np.all(arr1 > 0) and np.all(arr2 > 0)
#         assert np.all(arr1 < 1) and np.all(arr2 < 1)

        arr1 = rankdata(arr1) / (arr1.size + 1.0)
        arr2 = rankdata(arr2) / (arr2.size + 1.0)

    return arr1, arr2


if __name__ == '__main__':
    print('#### Started on %s ####\n' % time.asctime())
    START = timeit.default_timer()

    #==========================================================================
    # When in post_mortem:
    # 1. "where" to show the stack,
    # 2. "up" move the stack up to an older frame,
    # 3. "down" move the stack down to a newer frame, and
    # 4. "interact" start an interactive interpreter.
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
