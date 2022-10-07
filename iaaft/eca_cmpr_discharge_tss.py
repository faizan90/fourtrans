'''
@author: Faizan-Uni-Stuttgart

Jul 22, 2022

2:32:56 PM

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

    main_dir = Path(r'P:\Synchronize\IWS\Testings\hydmod\iaaftsa_sims')
    os.chdir(main_dir)

    hydmod_dir = Path(r'hydmod/test_hbv_all_03__phs_swap')

    sim_files = hydmod_dir.glob(
        './sim_data_*/02_hydrographs/calib_kfold_01__cats_outflow.csv')

    # ref_file = r'P:\Synchronize\IWS\Testings\fourtrans_practice\iaaft\neckar_q_data_combined_20180713_10cps.csv'
    ref_file = hydmod_dir / Path(r'ref_data/02_hydrographs/calib_kfold_01__cats_outflow.csv')

    warmup_steps = 0

    out_dir = Path(f'cmprs/discharge_tss/{hydmod_dir.name}')
    #==========================================================================

    out_dir.mkdir(exist_ok=True, parents=True)

    ref_df = pd.read_csv(ref_file, sep=';', index_col=0).iloc[warmup_steps:,:]

    ref_df.index = pd.to_datetime(ref_df.index, format='%Y-%m-%d')

    cols = set([])
    sim_dfs = []
    for sim_file in sim_files:
        sim_df = pd.read_csv(
            sim_file, sep=';', index_col=0).iloc[warmup_steps:,:]

        sim_df.index = pd.to_datetime(sim_df.index, format='%Y-%m-%d')

        sim_dfs.append(sim_df)

        cols.update(sim_df.columns.tolist())

    assert sim_dfs
    #==========================================================================

    fig = plt.figure(figsize=(10, 6))

    ts_ax = plt.subplot2grid((5, 1), (0, 0), 4, 1, fig=fig)
    bd_ax = plt.subplot2grid((5, 1), (4, 0), 1, 1, fig=fig, sharex=ts_ax)

    ref_df = ref_df.loc[sim_dfs[0].index,:]

    for col in cols:

        sim_mins = np.full(ref_df.shape[0], +np.inf)
        sim_maxs = np.full(ref_df.shape[0], -np.inf)

        leg_flag = True
        for sim_df in sim_dfs:

            if leg_flag:
                label = 'sim'
                leg_flag = False

            else:
                label = None

            ts_ax.plot(
                sim_df.index,
                sim_df[col].values,
                c='k',
                alpha=0.1,
                zorder=1,
                label=label)

            sim_mins = np.minimum(sim_mins, sim_df[col].values)
            sim_maxs = np.maximum(sim_maxs, sim_df[col].values)

        ts_ax.plot(
            ref_df.index,
            ref_df[col].values,
            c='r',
            alpha=0.75,
            zorder=2,
            label='ref')

        bd_arr = np.zeros(ref_df.shape[0])

        bd_arr[ref_df[col].values < sim_mins] = -1
        bd_arr[ref_df[col].values > sim_maxs] = +1

        bd_ax.plot(ref_df.index, bd_arr, c='b', alpha=0.8)

        ts_ax.grid()
        ts_ax.set_axisbelow(True)

        bd_ax.set_xlabel('Time (day)')

        ts_ax.set_ylabel('Discharge ($m^3.s^{-1}$)')
        bd_ax.set_ylabel('Containtment')

        plt.draw()

        bd_ax.set_yticks([-1, 0, 1], ['Below', 'Within', 'Above'])

        plt.setp(ts_ax.get_xticklabels(), visible=False)

        plt.savefig(
            out_dir / f'discharge_ts.png', bbox_inches='tight', dpi=150)

        plt.clf()

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
