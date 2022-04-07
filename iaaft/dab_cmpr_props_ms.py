'''
@author: Faizan-Uni-Stuttgart

Dec 6, 2021

2:35:27 PM

'''
import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path
from fnmatch import fnmatch
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import rankdata
import matplotlib.pyplot as plt; plt.ioff()

DEBUG_FLAG = True


def set_mpl_prms(prms_dict):

    plt.rcParams.update(prms_dict)

    return


def get_mag_phs_spec(arr):

    assert isinstance(arr, np.ndarray)
    assert arr.ndim == 1
    assert np.all(np.isfinite(arr))
    assert arr.size >= 3

    ft = np.fft.rfft(arr)

    mag_spec = np.abs(ft)
    phs_spec = np.angle(ft)

    return mag_spec[1:], phs_spec[1:]


def get_ft_cum_corr(arr_1, arr_2):

    mag_spec_1, phs_spec_1 = get_mag_phs_spec(arr_1)
    mag_spec_2, phs_spec_2 = get_mag_phs_spec(arr_2)

    assert mag_spec_1.size == mag_spec_2.size
    assert phs_spec_1.size == phs_spec_2.size

    mag_specs_prod = mag_spec_1 * mag_spec_2

    denom_corrs = (
        ((mag_spec_1 ** 2).sum() ** 0.5) *
        ((mag_spec_2 ** 2).sum() ** 0.5))

    corr = (mag_specs_prod * np.cos(phs_spec_1 - phs_spec_2)).cumsum()

    return corr, denom_corrs


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Testings\fourtrans_practice\iaaft')

    main_dir /= r'iaaft_discharge_04_no_cps_ranks_only_daily'
    os.chdir(main_dir)

    data_dir = main_dir

    prms_dict = {
        'figure.figsize': (15, 10),
        'figure.dpi': 150,
        'font.size': 16,
        }

    sep = ';'

    patt_ref = 'ref'
    patt_sim = 'S*'

    out_dir = main_dir
    #==========================================================================

    out_dir.mkdir(exist_ok=True)

    set_mpl_prms(prms_dict)

    clrs = ['r', 'k']

    cols = None
    in_dfs = {}
    file_labs = []
    for in_file in data_dir.glob('./auto_sims_*.csv'):
        in_df = pd.read_csv(in_file, sep=sep, index_col=0)

        if in_df.shape[0] % 2:
            in_df = in_df.iloc[:-1,:]

        if cols is None:
            cols = in_df.columns

        file_labs.append(in_file.stem.split('auto_sims_')[1])
        in_dfs[file_labs[-1]] = in_df

    assert in_dfs
    #==========================================================================

    ref_pwr = None

    combs = combinations(file_labs, 2)
    for comb in combs:
        leg_flag = True
        for col in cols:
            print(col, comb)

            data_a = in_dfs[comb[0]].loc[:, col].values.copy()
            data_b = in_dfs[comb[1]].loc[:, col].values.copy()

            if fnmatch(col, patt_ref):
                clr = clrs[0]

                lab = 'ref'

                zorder = 2

                plt_alpha = 0.6
                lw = 3.0

            else:
                clr = clrs[1]

                if leg_flag and fnmatch(col, patt_sim):
                    leg_flag = False
                    lab = 'sim'

                else:
                    lab = None

                plt_alpha = 0.35
                lw = 2.0

                zorder = 1

            ft_corr, pwr_denom = get_ft_cum_corr(data_a, data_b)

            if fnmatch(col, patt_ref):
                ref_pwr = pwr_denom

            ft_corr /= ref_pwr

            periods = (ft_corr.size * 2) / (
                np.arange(1, ft_corr.size + 1))

            assert periods.size == ft_corr.shape[0]

            plt.semilogx(
                periods,
                ft_corr,
                alpha=plt_alpha,
                color=clr,
                label=lab,
                lw=lw,
                zorder=zorder)

        plt.legend()

        plt.grid()
        plt.gca().set_axisbelow(True)

        plt.xlabel('Period')
        plt.ylabel('Cummulative correlation')

        plt.xlim(plt.xlim()[::-1])

        plt.savefig(
            str(out_dir / f'cross_cumm_pwr_margs__{comb[0]}__{comb[1]}.png'),
            bbox_inches='tight')

        plt.close()
    #==========================================================================
    ref_pwr = None
    # leg_flag = True

    combs = combinations(file_labs, 2)
    for comb in combs:
        leg_flag = True
        for col in cols:
            print(col, comb)

            data_a = rankdata(in_dfs[comb[0]].loc[:, col].values)
            data_b = rankdata(in_dfs[comb[1]].loc[:, col].values)

            if fnmatch(col, patt_ref):
                clr = clrs[0]

                lab = 'ref'

                zorder = 2

                plt_alpha = 0.6
                lw = 3.0

            else:
                clr = clrs[1]

                if leg_flag and fnmatch(col, patt_sim):
                    leg_flag = False
                    lab = 'sim'

                else:
                    lab = None

                plt_alpha = 0.35
                lw = 2.0

                zorder = 1

            ft_corr, pwr_denom = get_ft_cum_corr(data_a, data_b)

            if fnmatch(col, patt_ref):
                ref_pwr = pwr_denom

            ft_corr /= ref_pwr

            periods = (ft_corr.size * 2) / (
                np.arange(1, ft_corr.size + 1))

            assert periods.size == ft_corr.shape[0]

            plt.semilogx(
                periods,
                ft_corr,
                alpha=plt_alpha,
                color=clr,
                label=lab,
                lw=lw,
                zorder=zorder)

        plt.legend()

        plt.grid()
        plt.gca().set_axisbelow(True)

        plt.xlabel('Period')
        plt.ylabel('Cummulative correlation')

        plt.xlim(plt.xlim()[::-1])

        plt.savefig(
            str(out_dir / f'cross_cumm_pwr_ranks__{comb[0]}__{comb[1]}.png'),
            bbox_inches='tight')

        plt.close()
    #==========================================================================
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
