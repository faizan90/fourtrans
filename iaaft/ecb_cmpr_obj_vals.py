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

    hydmod_dir = Path(r'hydmod/test_hbv_all_06__mult_prms_search')

    sim_files = hydmod_dir.glob(
        './sim_data_*/03_hbv_figs/kf_01_calib_HBV_model_params_420.csv')

    ref_file = hydmod_dir / Path(r'ref_data/03_hbv_figs/kf_01_calib_HBV_model_params_420.csv')

    obj_val_labels = ['ns', 'ln_ns', 'kge', 'p_corr']

    out_dir = Path(f'cmprs/obj_vals/{hydmod_dir.name}')
    #==========================================================================

    out_dir.mkdir(exist_ok=True, parents=True)

    ref_df = pd.read_csv(ref_file, sep=';', index_col=0).loc[obj_val_labels]
    ref_df = ref_df.iloc[:, 0].astype(float).copy()

    sim_dfs = []
    for sim_file in sim_files:
        sim_df = pd.read_csv(sim_file, sep=';', index_col=0).loc[obj_val_labels]
        sim_df = sim_df.iloc[:, 0].astype(float).copy()

        sim_dfs.append(sim_df)

    assert sim_dfs
    #==========================================================================

    plt.figure(figsize=(7, 6))

    for obj_val_label in obj_val_labels:

        sim_obj_vals = np.sort(
            [sim_df.loc[obj_val_label] for sim_df in sim_dfs])

        sim_probs = np.arange(1.0, sim_obj_vals.size + 1.0)
        sim_probs /= (sim_probs.size + 1.0)

        plt.plot(
            sim_obj_vals,
            sim_probs,
            c='k',
            alpha=0.75,
            zorder=2,
            label='sim')

        plt.vlines(
            ref_df.loc[obj_val_label],
            0,
            1,
            colors='r',
            lw=2.5,
            alpha=0.9,
            label='ref')

        plt.legend(loc='upper left')

        plt.grid()
        plt.gca().set_axisbelow(True)

        plt.xlabel(f'Objective function value [{obj_val_label}]')

        plt.ylabel('Non-exceedence probability [-]')

        plt.draw()

        plt.ylim(-0.05, 1.05)

        plt.savefig(
            out_dir / f'obj_val_cmpr__{obj_val_label}.png',
            bbox_inches='tight',
            dpi=150)

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
