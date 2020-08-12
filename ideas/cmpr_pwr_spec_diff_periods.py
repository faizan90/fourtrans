'''
@author: Faizan-Uni-Stuttgart

12 August 2020

11:33:43
'''

import os
import time
import timeit
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.ioff()

DEBUG_FLAG = True


def get_mag_and_phs_spec(data):

    ft = np.fft.rfft(data)

    mag_spec = np.abs(ft)
    phs_spec = np.angle(ft)

    return mag_spec, phs_spec


def get_cumm_ft_corr(mag_spec):

    mag_spec_sq = mag_spec ** 2

    cumm_corr = np.cumsum(mag_spec_sq)
    cumm_corr /= cumm_corr[-1]

    return cumm_corr


def plot_ser_cumm_corr(ser, label):

    mag_spec, _ = get_mag_and_phs_spec(ser.values)

    cumm_corr = get_cumm_ft_corr(mag_spec[1:])

    periods = (mag_spec.size * 2) / (np.arange(1, mag_spec.size))

    plt.semilogx(periods, cumm_corr, alpha=0.7, label=label)

    return cumm_corr


def main():

    main_dir = Path(
        r'P:\Synchronize\IWS\Testings\fourtrans_practice\moving_window_statistic')

    os.chdir(main_dir)

    data_file = Path(
        r'neckar_norm_cop_infill_discharge_1961_2015_20190118.csv')

    prd_1 = '1970-01-01', '1974-12-31'
    prd_2 = '1999-01-01', '2003-12-31'
    prd_3 = '1962-01-01', '1966-12-31'
    prd_4 = '1989-01-01', '1993-12-31'

    col = '3470'

    fig_size = (15, 7)

    rank_flag = True

    x_label = 'Period (days)'
    y_label = 'Cummulative rank correlation contribution'

    out_fig_name = f'cumm_rank_corr_cmpr_{col}.png'

    data_ser = pd.read_csv(data_file, sep=';', index_col=0)[col]

    prds = [prd_1, prd_2, prd_3, prd_4]

    plt.figure(figsize=fig_size)
    for prd in prds:

        if rank_flag:
            sub_data_ser = data_ser.loc[prd[0]:prd[1]].rank()

        else:
            sub_data_ser = data_ser.loc[prd[0]:prd[1]]

        plot_ser_cumm_corr(sub_data_ser, f'{prd[0]}--{prd[1]}')

    plt.xlim(plt.xlim()[::-1])

    plt.legend()
    plt.grid()

    plt.xlabel(x_label)
    plt.ylabel(y_label)

#     plt.show()

    plt.savefig(out_fig_name, bbox_inches='tight')

    plt.close()

    return


if __name__ == '__main__':

    _save_log_ = False
    if _save_log_:
        from datetime import datetime
        from std_logger import StdFileLoggerCtrl

        # save all console activity to out_log_file
        out_log_file = os.path.join(
            r'P:\Synchronize\python_script_logs\\%s_log_%s.log' % (
            os.path.basename(__file__),
            datetime.now().strftime('%Y%m%d%H%M%S')))

        log_link = StdFileLoggerCtrl(out_log_file)

    print('#### Started on %s ####\n' % time.asctime())
    START = timeit.default_timer()

    #==========================================================================
    # When in post_mortem:
    # 1. "where" to show the stack
    # 2. "up" move the stack up to an older frame
    # 3. "down" move the stack down to a newer frame
    # 4. "interact" start an interactive interpreter
    #==========================================================================

    if DEBUG_FLAG:
        try:
            main()

        except:
            import pdb
            pdb.post_mortem()

    else:
        main()

    STOP = timeit.default_timer()
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))

    if _save_log_:
        log_link.stop()
