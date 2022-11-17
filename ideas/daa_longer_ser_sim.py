'''
@author: Faizan-Uni-Stuttgart

Nov 13, 2022

1:34:32 PM

'''
import os
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

    main_dir = Path(
        r'P:\Synchronize\IWS\Testings\fourtrans_practice\longer_series')

    os.chdir(main_dir)

    in_data_file = Path(
        r'neckar_norm_cop_infill_discharge_1961_2015_20190118.csv')

    stn = '420'

    beg_time = '1990-01-01'
    end_time = '1990-12-31'
    #==========================================================================

    in_data = pd.read_csv(
        in_data_file, sep=';', index_col=0).loc[beg_time:end_time, stn].values

    if in_data.size % 2:
        in_data = in_data[:-1].copy()

    ft = np.fft.rfft(in_data)[1:]

    mag = np.abs(ft)

    phs = np.angle(ft)

    data_val = in_data.copy()

    data_idxs = np.ones_like(data_val, dtype=bool)

    data_idxs[10] = False

    data_val[data_idxs] = 0.0

    ft_val = np.fft.rfft(data_val)[1:]

    mag_val = np.abs(ft_val)

    phs_val = np.angle(ft_val)

    coss = np.cos(phs - phs_val).cumsum()

    periods = (ft.shape[0] * 2) / np.arange(1, ft.shape[0] + 1)

    # plt.semilogx(periods, coss, alpha=0.8, label='coss')
    plt.semilogx(periods, mag, alpha=0.8, label='mag')
    plt.semilogx(periods, mag_val, alpha=0.8, label='mag_val')

    plt.xlim(plt.xlim()[::-1])

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
