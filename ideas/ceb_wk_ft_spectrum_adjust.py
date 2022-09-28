'''
@author: Faizan-Uni-Stuttgart

Sep 18, 2022

6:10:31 PM

'''
import os

# Numpy sneakily uses multiple threads sometimes. I don't want that.
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MPI_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

import sys
import time
import timeit
import traceback as tb
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt; plt.ioff()

DEBUG_FLAG = False


def main():

    main_dir = Path(os.getcwd())
    os.chdir(main_dir)

    sims_file = Path(r"P:\Synchronize\IWS\Testings\fourtrans_practice\iaaft\test_asymmetrize_ms_19\sim_files\auto_sims_420.csv")

    sep = ';'

    ref_col = 'ref'

    sim_col = 'S00000'
    #==========================================================================

    sim_df = pd.read_csv(
        sims_file, index_col=0, sep=sep).loc['1961-08-01':'1963-08-01']

    ref_ser = sim_df[ref_col].values.copy()

    if ref_ser.size % 2:
        ref_ser = ref_ser[:-1].copy()

    assert np.all(np.isfinite(ref_ser))

    sim_ser = sim_df[sim_col].values.copy()

    if sim_ser.size % 2:
        sim_ser = sim_ser[:-1].copy()

    assert np.all(np.isfinite(sim_ser))

    assert ref_ser.shape == sim_ser.shape

    ref_srtd = np.sort(ref_ser)

    lags = np.arange(1, ref_ser.size, dtype=np.int64)

    ref_pcorrs = []
    sim_pcorrs = []
    for lag in lags:
        ref_pcorr = np.corrcoef(ref_ser, np.roll(ref_ser, lag))[0, 1]
        ref_pcorrs.append(ref_pcorr)

        sim_pcorr = np.corrcoef(sim_ser, np.roll(sim_ser, lag))[0, 1]
        sim_pcorrs.append(sim_pcorr)

    ref_pcorrs = np.array(ref_pcorrs)
    sim_pcorrs = np.array(sim_pcorrs)

    ref_sim_pcorrs_diff = 2 * (ref_pcorrs - sim_pcorrs)

    # wk_ft = np.zeros(lags.size, dtype=complex)
    #
    # cnst = 2 * np.pi * 1j
    #
    # PCorr to WK.
    # for i in range(wk_ft.size):
    #     for j in range(wk_ft.size):
    #         wk_ft[i] += ref_sim_pcorrs_diff[j] * np.exp((-cnst * lags[j] * i) / wk_ft.size)
    #         wk_ft[i] += sim_pcorrs[j] * np.exp((-cnst * lags[j] * i) / wk_ft.size)

    wk_ft = np.fft.fft(ref_sim_pcorrs_diff + sim_pcorrs)

    # # WK to PCorr.
    # wk_pcorrs = []
    # for i in range(wk_ft.size):
    #     wk_pcorr = 0.0j
    #
    #     for j in range(wk_ft.size):
    #         wk_pcorr += wk_ft[j] * np.exp((cnst * lags[i] * j) / wk_ft.size)
    #
    #     # The sign is important to get the direction of correlation.
    #     if wk_pcorr.real >= 0:
    #         sign = +1
    #
    #     else:
    #         sign = -1
    #
    #     wk_pcorrs.append(sign * abs(wk_pcorr) / wk_ft.size)
    #
    # wk_pcorrs = np.array(wk_pcorrs)

    wk_pcorrs = ref_sim_pcorrs_diff + sim_pcorrs

    wk_ifft = np.fft.ifft(wk_ft)

    wk_ser = ref_srtd[np.argsort(np.argsort(wk_ifft))]

    wk_pcorrs_ifft = []
    for lag in lags:
        wk_pcorr_ifft = np.corrcoef(wk_ser, np.roll(wk_ser, lag))[0, 1]
        wk_pcorrs_ifft.append(wk_pcorr_ifft)

    wk_pcorrs_ifft = np.array(wk_pcorrs_ifft)

    ref_pwr = np.abs(np.fft.rfft(ref_ser)) ** 2

    # This is very very important.
    ref_pwr[0] = 0

    cftn_ft = np.fft.irfft(ref_pwr)
    cftn_ft /= cftn_ft[0]

    ref_pwr = ref_pwr.cumsum()

    ref_pwr /= ref_pwr[-1]

    if False:
        # pass
        periods = (ref_pwr.size * 2) / np.arange(1, ref_pwr.size + 1)

        assert periods.size == ref_pwr.shape[0]

        wk_pwr = np.abs(wk_ft[1:2 + wk_ft.size // 2])

        wk_pwr = wk_pwr.cumsum()
        wk_pwr /= wk_pwr[-1]

        # print(np.corrcoef(rankdata(ref_pwr), rankdata(wk_pwr))[0, 0])

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
        plt.plot(lags, ref_pcorrs, label='ref', zorder=1, lw=2)
        plt.plot(lags, sim_pcorrs, label='sim', zorder=2, lw=1)
        plt.plot(lags, wk_pcorrs, label='wk', zorder=2, lw=1)
        # plt.plot(lags, wk_pcorrs_ifft, label='wk_irfft', zorder=2, lw=1)

        # plt.plot(lags, cftn_ft, label='cftn_ft', zorder=2, lw=1)

        plt.xlabel('Lag steps')
        plt.ylabel('Pearson correlation')

    plt.legend()

    plt.grid()
    plt.gca().set_axisbelow(True)

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
