'''
@author: Faizan-Uni-Stuttgart

Feb 9, 2022

2:38:20 PM

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

from fcopulas import (
    get_asymm_1_sample, get_asymm_2_sample, get_asymm_1_max, get_asymm_2_max)

from zb_cmn_ftns_plot import set_mpl_prms, roll_real_2arrs_with_nan

DEBUG_FLAG = False


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Testings\fourtrans_practice\iaaft')

    main_dir /= r'test_asymm23_dis_16_03'

    os.chdir(main_dir)

    data_dir = main_dir

    sims_lab = 'S'

    sep = ';'

    prms_dict = {
        'figure.figsize': (15, 10),
        'figure.dpi': 150,
        'font.size': 16,
        }

    lags = np.array([-1, 0, 1], dtype=np.int64)

    show_best_flag = True
    # show_best_flag = False

    obj_vals_file_path = Path(r'all_obj_vals.csv')

    out_dir = main_dir
    #==========================================================================

    out_dir.mkdir(exist_ok=True)

    ref_df = pd.read_csv(
        data_dir / r'cross_sims_ref.csv', sep=sep, index_col=0)

    if ref_df.shape[1] < 2:
        print('No cross asymmetries for only one time series!')
        return

    if show_best_flag and obj_vals_file_path.exists():
        obj_vals_df = pd.read_csv(obj_vals_file_path, sep=sep, index_col=0)

        best_sim_label = obj_vals_df.columns[
            np.argmin(obj_vals_df.iloc[:,:].values) % obj_vals_df.shape[1]]

    else:
        best_sim_label = None

    sim_dfs = []
    for sim_file in data_dir.glob(rf'./cross_sims_{sims_lab}*.csv'):
        # print('Going through:', sim_file)

        sim_df = pd.read_csv(sim_file, sep=sep, index_col=0)

        sim_label = sim_file.stem.rsplit('_', 1)[1]

        sim_dfs.append((sim_label, sim_df))

    sim_df = None

    for asymmm_type in ('order', 'directional'):
        out_fig_path = out_dir / f'cross_asymms_{asymmm_type}.png'

        args = (
            ref_df,
            sim_dfs,
            lags,
            asymmm_type,
            out_fig_path,
            prms_dict,
            best_sim_label)

        plot_cross_asymms(args)

    return


def plot_cross_asymms(args):

    (ref_df,
     sim_dfs,
     lags,
     asymm_type,
     out_fig_path,
     prms_dict,
     best_sim_label) = args

    set_mpl_prms(prms_dict)

    ref_cross_asymms_df_lags = {}
    ref_asymms_lags = {}

    for lag in lags:
        ref_cross_asymms_df, ref_asymms = get_lagged_asymms(
            ref_df, lag, asymm_type)

        ref_cross_asymms_df_lags[lag] = ref_cross_asymms_df
        ref_asymms_lags[lag] = ref_asymms

        if False:
            print(ref_cross_asymms_df)
            print(ref_asymms)

    fig_scatt, ax_scatt = plt.subplots(1, 1,)

    scatt_min, scatt_max = np.inf, -np.inf

    leg_flag = True
    best_leg_flag = True

    all_refs = []
    all_sims = []
    for sim_label, sim_df in sim_dfs:
        for i, lag in enumerate(lags):

            ref_asymms = ref_asymms_lags[lag]
            sim_asymms = get_lagged_asymms(sim_df, lag, asymm_type)[1]

            if leg_flag:
                label = f'lag: {lag:+d}'

            else:
                label = None

            if best_sim_label == sim_label:

                old_label = label

                c = 'r'
                zorder = 2

                if best_leg_flag:
                    label = 'best'

                    best_leg_flag = False

                else:
                    label = old_label

            else:
                c = f'C{i}'
                zorder = 1

            # Scatter.
            plt.figure(fig_scatt)

            ax_scatt.scatter(
                ref_asymms,
                sim_asymms,
                alpha=0.5,
                c=c,
                label=label,
                zorder=zorder)

            scatt_min = min([min(ref_asymms), min(sim_asymms), scatt_min])
            scatt_max = max([max(ref_asymms), max(sim_asymms), scatt_max])

            all_refs.extend(ref_asymms.tolist())
            all_sims.extend(sim_asymms.tolist())

        leg_flag = False

    plt.figure(fig_scatt)

    ax_scatt.set_xlabel('Reference')
    ax_scatt.set_ylabel('Simulated')

    scatt_min -= 0.05 * (scatt_max - scatt_min)
    scatt_max += 0.05 * (scatt_max - scatt_min)

    plt.plot(
        [scatt_min, scatt_max],
        [scatt_min, scatt_max],
        alpha=0.25,
        ls='--',
        c='r',
        label='ideal')
    #==========================================================================

    all_refs = np.array(all_refs)
    all_sims = np.array(all_sims)

    poly_coeffs = np.polyfit(
        all_refs,
        all_sims,
        deg=1)

    poly_ftn = np.poly1d(poly_coeffs)

    scatt_min_sim, scatt_max_sim = poly_ftn([scatt_min, scatt_max])

    plt.plot(
        [scatt_min, scatt_max],
        [scatt_min_sim, scatt_max_sim],
        alpha=0.25,
        ls='--',
        c='k',
        label='fit')
    #==========================================================================

    ax_scatt.set_xlim(scatt_min, scatt_max)
    ax_scatt.set_ylim(scatt_min, scatt_max)

    ax_scatt.grid()
    ax_scatt.set_axisbelow(True)

    ax_scatt.legend()

    ax_scatt.set_aspect('equal')

    plt.savefig(out_fig_path, dpi=150, bbox_inches='tight')

    plt.close('all')

    # plt.show()
    return


def get_lagged_asymms(in_df, lag, asymm_type):

    asymms_df = pd.DataFrame(
        index=in_df.columns,
        columns=in_df.columns,
        data=np.zeros((in_df.shape[1], in_df.shape[1])))

    # All values above the diagonal.
    asymms = []

    if asymm_type == 'order':
        asymm_ftn = get_asymm_1_sample
        asymm_norm = get_asymm_1_max

    elif asymm_type == 'directional':
        asymm_ftn = get_asymm_2_sample
        asymm_norm = get_asymm_2_max

    else:
        raise NotImplementedError(asymm_type)

    for i, stn_i in enumerate(in_df.columns):
        arr_i = in_df[stn_i].values.copy()
        for j, stn_j in enumerate(in_df.columns):

            if i > j:
                asymms_df.iloc[i, j] = asymms_df.iloc[j, i]

            elif i == j:
                asymms_df.iloc[i, j] = 1.0

            else:
                arr_j = in_df[stn_j].values

                arr_i_lag, arr_j_lag = roll_real_2arrs_with_nan(
                    arr_i, arr_j, lag, True)

                scorr = np.corrcoef(arr_i_lag, arr_j_lag)[0, 1]

                asymm = asymm_ftn(arr_i_lag, arr_j_lag)

                asymm /= asymm_norm(scorr)

                asymms_df.iloc[i, j] = asymm

                asymms.append(asymm)

    asymms = np.array(asymms)
    return asymms_df, asymms


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
