'''
@author: Faizan-Uni-Stuttgart

Aug 2, 2022

3:36:02 PM

'''
import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt; plt.ioff()

DEBUG_FLAG = False


def main():

    main_dir = Path(os.getcwd())
    os.chdir(main_dir)

    #==========================================================================

    max_shift = +3

    n_levels = 100

    levels = np.arange(n_levels, dtype=float)

    shift_exps = np.array([0.5, 1.0, 2.0, 5.0, 10.0])

    plot_shift_cnst = 0.05
    plot_shift = plot_shift_cnst * (len(shift_exps) - 1)
    for shift_exp in shift_exps:
        shifts = max_shift - (max_shift * ((levels / n_levels) ** shift_exp))

        shifts = np.round(shifts)

        shifts = shifts.astype(int)

        plt.plot(levels, (shifts - plot_shift), label=shift_exp, alpha=0.75)

        plot_shift -= plot_shift_cnst

    plt.grid()
    plt.gca().set_axisbelow(True)

    plt.legend()

    plt.xlabel('Level')
    plt.ylabel('Shift')

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
