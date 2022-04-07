'''
@author: Faizan-Uni-Stuttgart

Apr 7, 2022

5:04:46 PM

'''
import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt; plt.ioff()

DEBUG_FLAG = False


def main():

    main_dir = Path(
        r'P:\Synchronize\IWS\Testings\fourtrans_practice\iaaft')

    main_dir /= r'iaaft_discharge_04_no_cps_ranks_only_daily'

    os.chdir(main_dir)

    data_dir = Path(r'resampled_series__time')

    resamp_res = 'W'

    out_fig_pref = f'RR{resamp_res}_RTsum'

    data_patt = f'auto_sims_*__{out_fig_pref}.csv'

    fig_x_label = f'{resamp_res} sum [-]'
    fig_y_label = '1 - F(x) [-]'

    out_dir = data_dir
    #==========================================================================

    out_dir.mkdir(exist_ok=True)

    for data_file in data_dir.glob(data_patt):
        print('Going through:', data_file)

        data_df = pd.read_csv(data_file, sep=';', index_col=0)

        assert isinstance(data_df, pd.DataFrame)

        plt.figure(figsize=(7, 7))
        for i, col in enumerate(data_df.columns):
            if i == 0:
                label = 'ref'
                clr = 'r'
                alpha = 0.75
                lw = 2.0

            elif i == 1:
                label = 'sim'
                clr = 'k'
                alpha = 0.5
                lw = 1.5

            else:
                label = None
                clr = 'k'
                alpha = 0.5
                lw = 1.5

            data = data_df[col].sort_values()
            data.dropna(inplace=True)

            sim_probs = data.rank().values / (data.shape[0] + 1.0)

            plt.semilogy(
                data.values,
                1 - sim_probs,
                c=clr,
                alpha=alpha,
                lw=lw,
                label=label)

        plt.grid(which='both')
        plt.gca().set_axisbelow(True)

        plt.legend()

        plt.xlabel(fig_x_label)
        plt.ylabel(fig_y_label)

        plt.savefig(
            out_dir / f'{out_fig_pref}_{data_file.stem}.png',
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
