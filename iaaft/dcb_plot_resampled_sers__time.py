'''
@author: Faizan-Uni-Stuttgart

Apr 7, 2022

5:04:46 PM

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

    main_dir = Path(
        r'P:\Synchronize\IWS\Testings\fourtrans_practice\iaaft')

    main_dir /= r'test_wk_33'

    os.chdir(main_dir)

    data_dir = main_dir / 'sim_files/resampled_series__time'

    resamp_res = 'W'

    out_fig_pref = f'RR{resamp_res}_RTsum'

    data_patt = f'auto_sims_*__{out_fig_pref}.csv'

    fig_x_label = f'{resamp_res} sum [-]'
    fig_y_label = '1 - F(x) [-]'

    prms_dict = {
        'figure.figsize': (7, 7),
        'figure.dpi': 150,
        'font.size': 16,
        }

    sim_alpha = 0.2

    show_best_flag = True
    # show_best_flag = False

    obj_vals_file_path = main_dir / 'sim_files/all_obj_vals.csv'

    out_dir = main_dir / 'figures'
    #==========================================================================

    out_dir.mkdir(exist_ok=True)

    if show_best_flag and obj_vals_file_path.exists():
        obj_vals_df = pd.read_csv(obj_vals_file_path, sep=';', index_col=0)

        best_sim_label = obj_vals_df.columns[
            np.argmin(obj_vals_df.iloc[:,:].values) % obj_vals_df.shape[1]]

    else:
        best_sim_label = None

    set_mpl_prms(prms_dict)
    for data_file in data_dir.glob(data_patt):
        # print('Going through:', data_file)

        data_df = pd.read_csv(data_file, sep=';', index_col=0)

        assert isinstance(data_df, pd.DataFrame)

        sim_leg_not_shown_flag = True
        for i, col in enumerate(data_df.columns):
            if i == 0:
                label = 'ref'
                clr = 'r'
                alpha = 0.75
                lw = 2.0
                zorder = 3

            elif col == best_sim_label:
                label = 'best'
                clr = 'b'
                alpha = 0.75
                lw = 2.0
                zorder = 2

            elif sim_leg_not_shown_flag:
                label = 'sim'
                clr = 'k'
                alpha = (3 / 256)
                lw = 1.5
                zorder = 1
                sim_leg_not_shown_flag = False

            else:
                label = None
                clr = 'k'
                alpha = sim_alpha
                lw = 1.5
                zorder = 1

            data = data_df[col].sort_values()
            data.dropna(inplace=True)

            sim_probs = data.rank().values / (data.shape[0] + 1.0)

            plt.semilogy(
                data.values,
                1 - sim_probs,
                c=clr,
                alpha=alpha,
                lw=lw,
                label=label,
                zorder=zorder)

        plt.grid(which='both')
        plt.gca().set_axisbelow(True)

        plt.legend()

        plt.xlabel(fig_x_label)
        plt.ylabel(fig_y_label)

        plt.savefig(
            out_dir / f'{data_file.stem}.png',
            dpi=150,
            bbox_inches='tight')

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
