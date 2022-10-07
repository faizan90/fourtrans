'''
@author: Faizan-Uni-Stuttgart

Apr 7, 2022

5:03:04 PM

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

    main_dir /= r'test_wk_182'

    os.chdir(main_dir)

    data_dir = main_dir / 'sim_files/resampled_series__space'

    out_fig_name = 'daily_space_sums.png'

    sims_lab = 'S'

    out_fig_pref = '*__RTsum'
    ref_data_patt = f'cross_sims_ref{out_fig_pref}.csv'
    sim_data_patt = f'cross_sims_{sims_lab}{out_fig_pref}.csv'

    fig_x_label = 'Stations\' sum [-]'
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

    read_ref_flag = False
    for ref_data_file in data_dir.glob(ref_data_patt):
        read_ref_flag = True
        break

    assert read_ref_flag, 'Didn\'t find the reference file!'

    ref_data_ser = pd.read_csv(
        ref_data_file, sep=';', index_col=0).squeeze("columns")

    assert isinstance(ref_data_ser, pd.Series)

    sim_data_sers = []
    for sim_data_file in data_dir.glob(sim_data_patt):
        sim_data_ser = pd.read_csv(
            sim_data_file, sep=';', index_col=0).squeeze("columns")

        assert isinstance(sim_data_ser, pd.Series)

        sim_label = sim_data_file.stem.split('_')[2]

        sim_data_sers.append((sim_label, sim_data_ser))

    assert sim_data_sers, 'Didn\'t find the simulation file(s)!'

    set_mpl_prms(prms_dict)

    if show_best_flag and obj_vals_file_path.exists():
        obj_vals_df = pd.read_csv(obj_vals_file_path, sep=';', index_col=0)

        best_sim_label = obj_vals_df.columns[
            np.argmin(obj_vals_df.iloc[:,:].values) % obj_vals_df.shape[1]]

    else:
        best_sim_label = None

    leg_flag = True
    best_leg_flag = True
    for sim_label, sim_data_ser in sim_data_sers:
        if leg_flag:
            label = 'sim'
            leg_flag = False

        else:
            label = None

        if best_sim_label == sim_label:

            old_label = label

            c = 'b'

            zorder = 3

            if best_leg_flag:
                label = 'best'

                best_leg_flag = False

                alpha = 0.4

            else:
                label = old_label

        else:
            c = 'k'
            zorder = 2

            alpha = sim_alpha

        sim_data_ser.dropna(inplace=True)

        # sim_data_ser.values[:] += 1e-3 * np.random.random(
        #     size=sim_data_ser.size)

        sim_ser = sim_data_ser.sort_values()
        sim_probs = sim_ser.rank().values / (sim_ser.shape[0] + 1.0)

        # print('Unique values in sim_probs:', np.unique(sim_probs).size)

        plt.semilogy(
            sim_ser.values,
            1 - sim_probs,
            c=c,
            alpha=alpha,
            lw=1.5,
            label=label,
            zorder=zorder)

    ref_data_ser.dropna(inplace=True)

    # ref_data_ser.values[:] += 1e-2 * np.random.random(
    #     size=ref_data_ser.size)

    ref_ser = ref_data_ser.sort_values()
    ref_probs = ref_ser.rank().values / (ref_ser.shape[0] + 1.0)

    # print('Unique values in ref_probs:', np.unique(ref_probs).size)

    plt.semilogy(
        ref_ser.values,
        1 - ref_probs,
        c='r',
        alpha=0.8,
        lw=2,
        label='ref',
        zorder=1)

    plt.grid(which='both')
    plt.gca().set_axisbelow(True)

    plt.legend()

    plt.xlabel(fig_x_label)
    plt.ylabel(fig_y_label)

    plt.savefig(
        out_dir / out_fig_name,
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
