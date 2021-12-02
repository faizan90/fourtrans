'''
@author: Faizan-Uni-Stuttgart

Nov 11, 2021

9:22:22 AM

'''
import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
import matplotlib.pyplot as plt; plt.ioff()

DEBUG_FLAG = False


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
        arr1 = arr1[-lag:].copy()
        arr2 = arr2[:+lag].copy()

    else:
        pass

    assert arr1.size == arr2.size

    if rerank_flag:
#         assert np.all(arr1 > 0) and np.all(arr2 > 0)
#         assert np.all(arr1 < 1) and np.all(arr2 < 1)

        arr1 = rankdata(arr1) / (arr1.size + 1.0)
        arr2 = rankdata(arr2) / (arr2.size + 1.0)

    return arr1, arr2


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Testings\fourtrans_practice\wiener_khintchin')
    os.chdir(main_dir)

    # in_data_file = Path(
    #     r'neckar_full_neckar_avg_temp_kriging_1961-01-01_to_2015-12-31_1km_all__EDK.csv')

    in_data_file = Path(
        r'neckar_norm_cop_infill_discharge_1961_2015_20190118.csv')

    stn = '420'

    beg_time = '1991-01-01'
    end_time = '2000-12-31'

    lags = np.arange(0, 21, dtype=np.int64)
    #==========================================================================

    in_data = pd.read_csv(
        in_data_file, sep=';', index_col=0).loc[beg_time:end_time, stn].values

    ref_pcorrs = []
    for lag in lags:
        pcorr = np.corrcoef(
            *roll_real_2arrs(in_data, in_data, lag, False))[0, 1]

        # pcorr = np.cov(
        #     *roll_real_2arrs(in_data, in_data, lag, False))[0, 1]

        ref_pcorrs.append(round(pcorr, 6))

    ref_pcorrs = np.array(ref_pcorrs)

    print(ref_pcorrs)

    # in_data = np.concatenate((in_data, np.zeros(in_data.size)))
    # in_data = np.concatenate((in_data, in_data[::-1]))

    ft = np.fft.fft(in_data)
    # ft = np.fft.rfft(in_data)

    mag = np.abs(ft)

    pwr = mag ** 2

    pwr = pwr[::-1]

    # pwr /= pwr.sum()

    cnst = 2 * np.pi * 1j
    # cnst = 2 * np.pi
    wk_pcorrs = []
    for lag in lags:
        pcorr = 0.0j
        for i in range(pwr.size):
            pcorr += np.exp((cnst * lag * i) / pwr.size) * pwr[i]
            # pcorr += np.cos((cnst * lag * i) / pwr.size) * pwr[i]

        wk_pcorrs.append(round(abs(pcorr) / pwr.sum(), 6))
        # wk_pcorrs.append(round(abs(pcorr), 6))

    wk_pcorrs = np.array(wk_pcorrs)
    print(wk_pcorrs)

    print(
        'ref and wk scorr:',
        np.corrcoef(rankdata(ref_pcorrs), rankdata(wk_pcorrs))[0, 1])

    # print(rankdata(ref_pcorrs), rankdata(wk_pcorrs))

    r_xx = np.abs(np.fft.ifft(np.abs(np.fft.fft(in_data)) ** 2))

    plt.plot(lags, ref_pcorrs, label='pc')
    plt.plot(lags, wk_pcorrs, label='wk')
    plt.plot(lags, r_xx[1:lags.size + 1] / r_xx[1], label='r_xx')

    plt.grid()
    plt.legend()
    plt.show()
    return


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
