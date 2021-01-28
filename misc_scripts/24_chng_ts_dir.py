'''
@author: Faizan-Uni-Stuttgart

Jan 27, 2021

3:27:35 PM

'''
import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt

DEBUG_FLAG = True


def main():

    main_dir = Path(os.getcwd())
    os.chdir(main_dir)

    # Partly from stack overflow.

    n = 128

    # mult ft[1:-1] with -1 == Asymm1, t_shift = 0.0
    # Conjugate ft == Asymm2, t_shift = -0.5

    t_shift = -0.5

    x = np.arange(n)
    # y = np.sin(2 * np.pi * x / n * 4 + np.pi / 3)

    y = np.concatenate((x, x[::-1]))
    # y = x.copy()

    y_rev = y[::-1]

    fft_y = np.fft.rfft(y)

    y_fft = np.conjugate(fft_y)
    # y_fft[1:-1] *= -1

    y_rev_fft = np.fft.irfft(y_fft)

    ft_shift = y_fft.copy()
    ft_shift *= (
        np.exp(-1j * 2 * np.pi *
               (np.arange(float(y_fft.shape[0])) / (y_fft.shape[0])) * t_shift))

    ft_shift_i = np.fft.irfft(ft_shift)

    plt.plot(y, label="Original Signal", alpha=0.75)
    plt.plot(y_rev, label="Straight Reverse", alpha=0.75)
    plt.plot(y_rev_fft, label="FFT Reverse", alpha=0.75)
    plt.plot(ft_shift_i, label="ft_shift_i", alpha=0.75)

    plt.legend()

    plt.show()

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
