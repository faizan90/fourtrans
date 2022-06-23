'''
@author: Faizan-Uni-Stuttgart

Jun 3, 2022

2:13:49 PM

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

from cc_cumm_ft_corr import get_cumm_ft_corr_auto

DEBUG_FLAG = False


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Testings\fourtrans_practice\iaaft')
    os.chdir(main_dir)

    in_data_file = Path(r'neckar_norm_cop_infill_discharge_1961_2015_20190118.csv')

    sep = ';'

    short_beg_time = '1980-01-01'
    short_end_time = '1984-12-30'
    short_col = '420'

    long_beg_time = '1986-01-01'
    long_end_time = '2010-12-30'
    long_col = '427'
    #==========================================================================

    if in_data_file.suffix == '.csv':
        df_data = pd.read_csv(in_data_file, sep=sep, index_col=0)

    elif in_data_file.suffix == '.pkl':
        df_data = pd.read_pickle(in_data_file)

    else:
        raise NotImplementedError(
            f'Unknown extension of in_data_file: '
            f'{in_data_file.suffix}!')

    short_corr, short_periods = get_cumm_ft_corr_auto(
        df_data.loc[short_beg_time:short_end_time, short_col].values)

    long_corr, long_periods = get_cumm_ft_corr_auto(
        df_data.loc[long_beg_time:long_end_time, long_col].values)

    print('Time period length factor:', long_periods[0] / short_periods[0])

    assert (long_periods[0] % short_periods[0]) == 0, (
        long_periods[0], short_periods[0])

    # Best method is not to have a transform of the periods apparently.
    # The difference of the periods between short and long has to be estimated
    # for simulation.
    if False:
        # Scaling like this does not work.
        short_periods = short_periods - short_periods.min()
        short_periods /= (short_periods.max() - short_periods.min())

        long_periods = long_periods - long_periods.min()
        long_periods /= (long_periods.max() - long_periods.min())

    elif False:
        short_periods = short_periods[::-1].cumsum()
        short_periods /= short_periods[-1]
        short_periods = short_periods[::-1]

        long_periods = long_periods[::-1].cumsum()
        long_periods /= long_periods[-1]
        long_periods = long_periods[::-1]

    elif False:
        short_periods = short_periods.cumsum()
        short_periods /= short_periods[-1]

        long_periods = long_periods.cumsum()
        long_periods /= long_periods[-1]

    elif False:
        short_periods /= short_periods.sum()
        long_periods /= long_periods.sum()

    elif False:
        short_periods = np.arange(1.0, short_corr.size + 1.0)[::-1] / short_corr.size
        long_periods = np.arange(1.0, long_corr.size + 1.0)[::-1] / long_corr.size

    elif True:
        pass

    else:
        raise Exception

    plt.semilogx(
        short_periods,
        short_corr,
        alpha=0.85,
        color='r',
        label='SHORT',
        lw=4,
        zorder=1,
        # marker='o',
        )

    plt.semilogx(
        long_periods,
        long_corr,
        alpha=0.75,
        color='g',
        label='LONG',
        lw=3,
        zorder=2,
        # marker='o',
        )

    plt.legend()

    plt.grid()
    plt.gca().set_axisbelow(True)

    plt.xlabel('Relative period')
    plt.ylabel('Cummulative power')

    plt.xlim(plt.xlim()[::-1])

    plt.show()

    # plt.savefig(f'{in_file.stem}_{out_fig_name_pwr}', bbox_inches='tight')
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
