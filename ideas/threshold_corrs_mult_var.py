'''
@author: Faizan

Aug 18, 2020

10:41:02 AM
'''
import os
import time
import timeit
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DEBUG_FLAG = True


def get_thresh_scorrs(vals, other_vals, threshs):

    probs_orig = vals / (vals.shape[0] + 1)

    other_probs = other_vals / (other_vals.shape[0] + 1)

    scorrs = []
    for thresh in threshs:
        probs = probs_orig.copy()
        probs[probs < thresh] = np.nan

        take_idxs = np.isfinite(probs)

        # High ppt will happen a day before
        op1, op2 = other_probs[1:, 0][take_idxs[:-1]], other_probs[1:, 1][take_idxs[:-1]]

        scorrs.append(np.corrcoef(op1, op2)[0, 1])

    return np.array(scorrs)


def main():

    main_dir = Path(
        r'P:\Synchronize\IWS\Testings\fourtrans_practice\moving_window_statistic')

    os.chdir(main_dir)

    data_file = Path(
        r'neckar_norm_cop_infill_discharge_1961_2015_20190118.csv')

    other_data_file = Path(r'precipitation_bw_1961_2015.csv')

    sep = ';'

    time_fmt = '%Y-%m-%d'

    col = '454'

    other_cols = ['P1727', 'P5229']

    # Value greater than prob_ge_thresh are considered
    prob_ge_threshs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

    prd_1 = '1970-01-01', '1974-12-31'
    prd_2 = '1999-01-01', '2003-12-31'
    prd_3 = '1962-01-01', '1966-12-31'
    prd_4 = '1989-01-01', '1993-12-31'

    fig_size = (10, 10)

    x_label = 'Threshold Prob (-)'
    y_label = 'Threshold Scorr.'

    out_fig_name = (
        f'thresh_scorr_cmpr_{other_cols[0]}_{other_cols[1]}_{col}.png')

    data_ser = pd.read_csv(data_file, sep=sep, index_col=0)[col]

    data_ser.index = pd.to_datetime(data_ser.index, format=time_fmt)

    other_data_ser = pd.read_csv(
        other_data_file, sep=sep, index_col=0)[other_cols]

    other_data_ser.index = pd.to_datetime(
        other_data_ser.index, format=time_fmt)

    prds = [prd_1, prd_2, prd_3, prd_4]

    plt.figure(figsize=fig_size)
    for prd in prds:
        sub_data_vals = data_ser.loc[prd[0]:prd[1]].rank(axis=0).values
        other_sub_data_vals = other_data_ser.loc[prd[0]:prd[1]].rank(axis=0).values

        scorrs = get_thresh_scorrs(
            sub_data_vals, other_sub_data_vals, prob_ge_threshs)

        label = f'{prd[0]}--{prd[1]}'

        plt.plot(prob_ge_threshs, scorrs, alpha=0.8, label=label)

    plt.grid()
    plt.legend()

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
