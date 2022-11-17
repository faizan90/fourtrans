'''
@author: Faizan-Uni-Stuttgart

Aug 12, 2020

2:43:02 PM

'''
import os
import time
import timeit
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata

DEBUG_FLAG = True


def roll_real_2arrs(arr1, arr2, lag):

    assert isinstance(arr1, np.ndarray)
    assert isinstance(arr2, np.ndarray)

    assert arr1.ndim == 1
    assert arr2.ndim == 1

    assert arr1.size == arr2.size

    assert isinstance(lag, (int, np.int64))
    assert abs(lag) < arr1.size

    if lag > 0:
        # arr2 is shifted ahead
        arr1 = arr1[:-lag].copy()
        arr2 = arr2[+lag:].copy()

    elif lag < 0:
        # arr1 is shifted ahead
        arr1 = arr1[+lag:].copy()
        arr2 = arr2[:-lag].copy()

    else:
        pass

    assert arr1.size == arr2.size

    return arr1, arr2


def get_lagged_thresh_scorrs(vals, lags, thresh):

    probs = rankdata(vals) / (vals.size + 1)

    probs[probs < thresh] = np.nan

    scorrs = []
    for lag in lags:
        p1, p2 = roll_real_2arrs(probs, probs, lag)

        take_idxs = np.isfinite(p1) & np.isfinite(p2)

        p1, p2 = p1[take_idxs], p2[take_idxs]

        scorrs.append(np.corrcoef(p1, p2)[0, 1])

    return np.array(scorrs)


def main():

    main_dir = Path(
        r'P:\Synchronize\IWS\Testings\fourtrans_practice\moving_window_statistic')

    os.chdir(main_dir)

    data_file = Path(
        r'neckar_norm_cop_infill_discharge_1961_2015_20190118.csv')

    sep = ';'

    time_fmt = '%Y-%m-%d'

    col = '454'

    lags = np.arange(1, 11, dtype=np.int64)

    # Value greater than prob_ge_thresh are considered
    prob_ge_thresh = 0.0

    prd_1 = '1970-01-01', '1974-12-31'
    prd_2 = '1999-01-01', '2003-12-31'
    prd_3 = '1962-01-01', '1966-12-31'
    prd_4 = '1989-01-01', '1993-12-31'

    fig_size = (10, 10)

    x_label = 'Lag (days)'
    y_label = 'Threshold Scorr.'

    out_fig_name = f'thresh_scorr_cmpr_{col}_tp{prob_ge_thresh:0.2f}.png'

    data_ser = pd.read_csv(data_file, sep=sep, index_col=0)[col]

    data_ser.index = pd.to_datetime(data_ser.index, format=time_fmt)

    prds = [prd_1, prd_2, prd_3, prd_4]

    plt.figure(figsize=fig_size)
    for prd in prds:
        sub_data_vals = data_ser.loc[prd[0]:prd[1]].values

        scorrs = get_lagged_thresh_scorrs(sub_data_vals, lags, prob_ge_thresh)

        label = f'{prd[0]}--{prd[1]}'

        plt.plot(lags, scorrs, alpha=0.8, label=label)

    plt.grid()
    plt.legend()

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.title(f'Thresh: {prob_ge_thresh}')

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
