'''
@author: Faizan-Uni-Stuttgart

Apr 7, 2022

5:13:21 PM

'''
import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path
from fnmatch import fnmatch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt; plt.ioff()

from kde.kernels import triangular_kern, gaussian_kern
from zb_cmn_ftns_plot import set_mpl_prms

DEBUG_FLAG = False


def main():

    '''
    Based on day of year regardless of the time resolution of the data.
    '''

    main_dir = Path(r'P:\Synchronize\IWS\Testings\fourtrans_practice\iaaft')

    main_dir /= r'test_spcorr_47'

    os.chdir(main_dir)

    in_files = main_dir.glob('./auto_sims_*.csv')

    prms_dict = {
        'figure.figsize': (15, 10),
        'figure.dpi': 80,
        'font.size': 16,
        }

    sep = ';'

    # For columns only.
    # So, files can have the same ref but different sims.
    patt_ref = 'ref'
    patt_sim = 'S*'

    # time_fmt = '%Y-%m-%d %H:%M:%S'
    time_fmt = '%Y-%m-%d'

    half_window_size = 30  # Julian days.

    max_n_sims = 5

    out_dir = main_dir
    #==========================================================================

    set_mpl_prms(prms_dict)

    clrs = ['r', 'k']

    doys = np.arange(1, 367)

    for in_file in in_files:

        print('Going through:', in_file)

        in_df = pd.read_csv(in_file, sep=sep, index_col=0)

        in_df.index = pd.to_datetime(in_df.index, format=time_fmt)

        leg_flag = True
        sim_ctr = 0
        for i in range(in_df.shape[1]):
            data = in_df.iloc[:, i].values.copy()

            print(in_df.columns[i])

            if (fnmatch(in_df.columns[i], patt_ref) or
                fnmatch(in_df.columns[i], patt_sim)):

                pass

            else:
                continue

            if fnmatch(in_df.columns[i], patt_ref):
                clr = clrs[0]

                lab = 'ref'

                zorder = 2

                plt_alpha = 0.6
                lw = 3.0

            else:
                clr = clrs[1]

                if leg_flag and fnmatch(in_df.columns[i], patt_sim):
                    leg_flag = False
                    lab = 'sim'

                else:
                    lab = None

                plt_alpha = 0.35
                lw = 2.0

                zorder = 1

            ann_cyc_arr = np.zeros_like(doys, dtype=float)
            for j, doy in enumerate(doys):
                doy_idxs = in_df.index.dayofyear == doy

                n_doy_idxs = doy_idxs.sum()

                # assert doy_idxs.sum(), (j, doy)

                if not n_doy_idxs:
                    continue

                doy_data = data[doy_idxs].mean()

                ann_cyc_arr[j] = doy_data

            ann_cyc_arr = get_smoothed_array(ann_cyc_arr, half_window_size)

            plt.plot(
                doys,
                ann_cyc_arr,
                alpha=plt_alpha,
                color=clr,
                label=lab,
                lw=lw,
                zorder=zorder)

            sim_ctr += 1

            if sim_ctr == max_n_sims:
                break

        plt.grid()
        plt.gca().set_axisbelow(True)

        plt.legend()

        plt.xlabel('Time')
        plt.ylabel('Smoothed Precipitation')

        out_fig_path = str(
            out_dir /
            f'auto_ann_cyc_{in_file.stem}_WS{half_window_size}JD.png')

        plt.savefig(out_fig_path, bbox_inches='tight')

        plt.cla()

    plt.close()
    return


def get_smoothed_array(data, half_window_size):

    n_vals = data.size

    smoothed_arr = np.empty(n_vals, dtype=float)

    data_padded = np.concatenate((
        data[::-1][:half_window_size], data, data[:half_window_size]))

    rel_dists = np.concatenate((
        np.arange(half_window_size, -1, -1.0,),
        np.arange(1.0, half_window_size + 1.0)))

    rel_dists /= rel_dists.max() + 1.0

    if True:
        kern_ftn = np.vectorize(triangular_kern)

    elif False:
        kern_ftn = np.vectorize(gaussian_kern)

    else:
        raise Exception

    window_wts = kern_ftn(rel_dists)

    window_wts /= window_wts.sum()

    for i in range(n_vals):
        smoothed_arr[i] = np.nansum(
            data_padded[i: i + 2 * half_window_size + 1] * window_wts)

    return smoothed_arr


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
