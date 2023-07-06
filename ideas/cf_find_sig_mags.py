'''
@author: Faizan-Uni-Stuttgart

Oct 11, 2022

11:49:14 AM

Keywords: Significance of magnitudes.

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

    # sims_file = Path(
    #     r'P:\Synchronize\IWS\Testings\fourtrans_practice\iaaft\neckar_q_data_combined_20180713_10cps.csv')
    #
    # ref_col = '420'

    sims_file = Path(
        r'P:\Synchronize\IWS\Testings\fourtrans_practice\iaaft\precipitation_bw_1961_2015_10cps.csv')

    ref_col = 'P1162'

    sep = ';'

    lim_alpha = 0.01

    n_sims = 10000
    #==========================================================================

    ref_ser = pd.read_csv(
        sims_file, index_col=0, sep=sep).loc[
            '1961-08-01':'1970-08-01', ref_col].values.copy()

    if ref_ser.size % 2:
        ref_ser = ref_ser[:-1].copy()

    assert np.all(np.isfinite(ref_ser))

    ref_mags = np.abs(np.fft.rfft(ref_ser)[1:])

    sim_magss = np.empty((n_sims, ref_mags.shape[0]))

    for i in range(n_sims):
        sim_ser = ref_ser.copy()

        np.random.shuffle(sim_ser)

        sim_mags = np.abs(np.fft.rfft(sim_ser)[1:])

        sim_magss[i,:] = sim_mags

    sim_magss.sort(axis=0)

    axis_mags, axis_lims = plt.subplots(
        nrows=2, ncols=1, height_ratios=[4, 1], sharex=True)[1]

    leg_flag = True
    for i in range(n_sims):
        if leg_flag:
            label = 'sim'
            leg_flag = False

        else:
            label = None

        axis_mags.plot(
            sim_magss[i,:], c='k', label=label, lw=0.5, alpha=0.1, zorder=1)

    max_lim_idx = int((1.0 - lim_alpha) * n_sims)
    min_lim_idx = int(lim_alpha * n_sims)

    axis_mags.plot(
        sim_magss[max_lim_idx,:],
        c='b',
        label='lim_max',
        lw=1.5,
        alpha=0.8,
        zorder=2)

    axis_mags.plot(
        sim_magss[min_lim_idx,:],
        c='g',
        label='lim_min',
        lw=1.5,
        alpha=0.8,
        zorder=3)

    axis_mags.plot(ref_mags, c='r', label='ref', lw=2.0, alpha=0.8, zorder=2)

    axis_mags.grid()
    axis_mags.set_axisbelow(True)

    axis_mags.legend()

    within_lim_idxs = (
        (sim_magss[min_lim_idx,:] <= ref_mags) &
        ((sim_magss[max_lim_idx,:] >= ref_mags)))

    within_lim_idxs = within_lim_idxs.astype(int)

    axis_lims.plot(
        within_lim_idxs,
        c='grey',
        label='lim',
        lw=2.0,
        alpha=0.8,
        zorder=2)

    axis_lims.set_yticks([0, 1])
    axis_lims.set_yticklabels(['Out', 'Within'])

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
