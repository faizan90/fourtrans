'''
@author: Faizan-Uni-Stuttgart

Jun 24, 2022

12:12:19 PM

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

from zb_cmn_ftns_plot import set_mpl_prms

DEBUG_FLAG = False


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Testings\fourtrans_practice\iaaft')

    main_dir /= r'test_wk_182'

    os.chdir(main_dir)

    data_dir = main_dir / 'sim_files'

    sep = ';'

    prms_dict = {
        'figure.figsize': (15, 10),
        'figure.dpi': 150,
        'font.size': 16,
        }

    sim_alpha = 0.2

    out_dir = main_dir / 'figures'
    #==========================================================================

    out_dir.mkdir(exist_ok=True)

    set_mpl_prms(prms_dict)

    plt.figure()
    for opt_file, y_label in zip(
        ['all_order_sdiffs.csv', 'all_obj_vals.csv'],
        ['order_sdiff', 'obj_val']):

        if not (data_dir / opt_file).exists():

            print(f'File: {opt_file} does not exist for opt plot!')
            continue

        opt_df = pd.read_csv(data_dir / opt_file, sep=sep, index_col=0)

        if y_label == 'obj_val':

            best_sim_label = opt_df.columns[
                np.argmin(opt_df.iloc[:,:].values) % opt_df.shape[1]]

            print(f'Sim {best_sim_label} has the best objective value.')

            plt.scatter(
                np.arange(opt_df.shape[1]),
                np.sort(opt_df.values.min(axis=0)),
                alpha=0.05,
                color='k',
                lw=2)

            plt.yscale('log')

            plt.xlabel('Sorted simulations')

            plt.ylabel(y_label)

            plt.grid()

            plt.gca().set_axisbelow(True)

            plt.savefig(
                str(out_dir / f'opt__{y_label}__last_sorted.png'),
                bbox_inches='tight')

            plt.clf()

        for sim_lab in opt_df.columns:
            plt.semilogy(
                opt_df[sim_lab][:],
                alpha=sim_alpha,
                color='k',
                lw=2)

            plt.xlabel('Iteration')

        plt.ylabel(y_label)

        plt.grid()

        plt.gca().set_axisbelow(True)

        plt.savefig(
            str(out_dir / f'opt__{y_label}.png'), bbox_inches='tight')

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
