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
import pyinform as pim
import matplotlib.pyplot as plt

asymms_exp = 3.0

plt.ioff()

DEBUG_FLAG = False


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


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Testings\fourtrans_practice\phsann')
    os.chdir(main_dir)

    data_file = Path(
        r'P:\Synchronize\IWS\Testings\fourtrans_practice\phsann\neckar_norm_cop_infill_discharge_1961_2015_20190118.csv')

    beg_time = '2000-01-01'
    end_time = '2010-12-31'

    col = '427'

    n_sims = 1000

    k = 2

    data = pd.read_csv(
        data_file, sep=';', index_col=0).loc[beg_time:end_time, col].values

    if data.size % 2:
        data = data[:-1]

    assert np.all(np.isfinite(data))

    data_mag_spec, data_phs_spec = get_mag_and_phs_spec(data)

    data_sorted = np.sort(data)

    data_ai = pim.active_info(data, k=k, local=True)[0]

    data_ai_mag_spec, data_ai_phs_spec = get_mag_and_phs_spec(data_ai)

    periods = (data_ai.size) / (
        np.arange(1, data_ai_mag_spec.size - 1))

    periods = np.concatenate(([data_ai.size * 2], periods))

    plt.figure(figsize=(10, 6))

    for i in range(n_sims):
        rand_phs_spec = (
            -np.pi + (2 * np.pi * np.random.random(data_phs_spec.size)))

        rand_phs_spec[+0] = data_phs_spec[+0]
        rand_phs_spec[-1] = data_phs_spec[-1]

        rand_data = get_irfted_vals(data_mag_spec, rand_phs_spec)

        rand_data = data_sorted[np.argsort(np.argsort(rand_data))]

        rand_ai = pim.active_info(rand_data, k=k, local=True)[0]

        rand_ai_mag_spec, rand_ai_phs_spec = get_mag_and_phs_spec(rand_ai)

        plt.semilogx(
            periods, rand_ai_mag_spec[1:].cumsum(), alpha=0.5, lw=1, c='b')

    plt.semilogx(
        periods, data_ai_mag_spec[1:].cumsum(), alpha=0.75, lw=2, c='r')

    plt.grid()
    plt.gca().set_axisbelow(True)

    plt.xlabel('Time step')
    plt.ylabel('Mag spec')

    plt.xlim(plt.xlim()[::-1])

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
