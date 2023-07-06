# -*- coding: utf-8 -*-

'''
@author: Faizan-Uni-Stuttgart

Mar 14, 2023

9:05:58 AM

Keywords:

'''
import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

DEBUG_FLAG = False


def main():

    main_dir = Path(os.getcwd())
    os.chdir(main_dir)

    mean = -2.0

    len_ts = 10000

    mags = np.array([0.0, mean * len_ts * 0.25, 0.0, ])
    phss = np.array([0.0, np.pi, 0.0])
    #==========================================================================

    prm_ts = sim_prm_ts(mean, mags, phss, len_ts)

    print(prm_ts)
    print(prm_ts.min(), prm_ts.mean(), prm_ts.max())

    plt.plot(prm_ts)

    plt.grid()
    plt.gca().set_axisbelow(True)

    plt.show()
    return


def sim_prm_ts(mean, mags, phss, len_ts):

    if len_ts % 2:
        adj_len = 2

    else:
        adj_len = 1

    prm_ft = np.zeros(adj_len + (len_ts // 2), dtype=np.complex128)

    prm_ft[0] = mean * len_ts

    n_mags = mags.size

    prm_ft[1:n_mags + 1].real = mags * np.cos(phss)
    prm_ft[1:n_mags + 1].imag = mags * np.sin(phss)

    prm_ts = np.fft.irfft(prm_ft, len_ts)

    assert len_ts == prm_ts.shape[0], (len_ts, prm_ts.shape[0])

    return prm_ts


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
