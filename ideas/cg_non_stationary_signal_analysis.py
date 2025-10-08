# -*- coding: utf-8 -*-

'''
@author: Faizan-TU Munich

Nov 3, 2023

10:54:25 AM

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

from cc_cumm_ft_corr import get_cumm_ft_corr_auto

DEBUG_FLAG = False


def main():

    main_dir = Path(os.getcwd())
    os.chdir(main_dir)

    ser_fst = np.sin(np.tile(np.linspace(-np.pi, +np.pi, 20, endpoint=False), 25)) * 100
    ser_slw = np.sin(np.tile(np.linspace(-np.pi, +np.pi, 100, endpoint=False), 5)) * 100

    wts = np.ones(ser_fst.size, dtype=np.float64)

    assert ser_fst.size == ser_slw.size, (ser_fst.size, ser_slw.size)

    # tns = np.linspace(1, 0, num=0, endpoint=False)
    #
    # wts[int(0.5 * ser_fst.size - np.round(tns.size / 2)):
    #     int(0.5 * ser_fst.size + np.round(tns.size / 2))] = tns
    #
    # wts[int(0.5 * ser_fst.size + np.round(tns.size / 2)):] = 0.0

    wts[int(0.5 * ser_fst.size):] = 0.0

    #==========================================================================

    ser_mix = (wts * ser_fst) + ((1 - wts) * ser_slw)
    # ser_mix = (wts * ser_slw) + ((1 - wts) * ser_fst)

    cum_ft_fst, periods_fst = get_cumm_ft_corr_auto(ser_fst)
    cum_ft_slw, periods_slw = get_cumm_ft_corr_auto(ser_slw)
    cum_ft_mix, periods_mix = get_cumm_ft_corr_auto(ser_mix)
    #==========================================================================

    plt.figure()

    plt.plot(ser_fst, label='FST', c='b')
    plt.plot(ser_slw, label='SLW', c='g')
    plt.plot(ser_mix, label='MIX', c='r')

    plt.legend()

    plt.grid()
    plt.gca().set_axisbelow(True)

    plt.xlabel('Time step')
    plt.ylabel('Magnitude')

    plt.show(block=False)
    #==========================================================================

    plt.figure()

    plt.semilogx(
        periods_fst,
        cum_ft_fst,
        color='b',
        label='FST')

    plt.semilogx(
        periods_slw,
        cum_ft_slw,
        color='g',
        label='SLW')

    plt.semilogx(
        periods_mix,
        cum_ft_mix,
        color='r',
        label='MIX')

    plt.legend()

    plt.grid()
    plt.gca().set_axisbelow(True)

    plt.xlim(plt.xlim()[::-1])

    plt.xlabel('Period')
    plt.ylabel('Cummulative power')

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
