# -*- coding: utf-8 -*-

'''
@author: Faizan-Uni-Stuttgart

Jan 9, 2023

2:46:40 PM

Keywords: Distribution of cross phase spectrum differnerces.

'''
import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DEBUG_FLAG = False


def main():

    main_dir = Path(
        r'P:\Synchronize\IWS\Testings\fourtrans_practice\cross_spec_dists')

    os.chdir(main_dir)

    in_tss_file = Path(
        r'neckar_norm_cop_infill_discharge_1961_2015_20190118.csv')

    stn_a, stn_b = '3465', '3470'

    ts_time_beg, ts_time_end = '1990-06-01', '2015-06-30'
    #==========================================================================

    tss_df = pd.read_csv(in_tss_file, sep=';', index_col=0)

    tss_df = tss_df.loc[ts_time_beg:ts_time_end, [stn_a, stn_b]].copy()

    assert np.all(np.isfinite(tss_df.values))

    tss = tss_df.values.copy()

    ft = np.fft.rfft(tss, axis=0)[1:-1]

    # mg = np.abs(ft)

    pa = np.angle(ft)

    ps = pa[:, 0] - pa[:, 1]

    pb = np.arange(1.0, ps.shape[0] + 1.0) / (ps.shape[0] + 1.0)

    pa0 = np.sort(pa[:, 0])

    plt.figure(figsize=(10, 10))

    # plt.scatter(ps, pb, alpha=0.75, edgecolors='none', c='r', zorder=2)
    plt.scatter(pa0, pb, alpha=0.75, edgecolors='none', c='r', zorder=2)

    for _ in range(100):

        rns = -np.pi + (2 * np.pi) * np.random.random(size=ps.shape)

        pa1 = rns - ps

        for j in range(pa1.shape[0]):

            phs = pa1[j]

            if phs > +np.pi:
                ratio = (phs / +np.pi) - 1
                phs = -np.pi * (1 - ratio)

            elif phs < -np.pi:
                ratio = (phs / -np.pi) - 1
                phs = +np.pi * (1 - ratio)

            pa1[j] = phs

        pa1 = np.sort(pa1)

        plt.plot(pa1, pb, alpha=0.2, c='k', zorder=1)

    plt.grid()
    plt.gca().set_axisbelow(True)

    plt.show()

    plt.close()
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
