'''
@author: Faizan-Uni-Stuttgart

Aug 12, 2020

9:03:08 AM

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


def main():

    main_dir = Path(
        r'P:\Synchronize\IWS\Testings\fourtrans_practice\moving_window_statistic')

    os.chdir(main_dir)

    data_file = Path(
        r'neckar_norm_cop_infill_discharge_1961_2015_20190118.csv')

    sep = ';'

    time_fmt = '%Y-%m-%d'

    col = '454'

    rank_flag = True

    x_label = 'Window middle (days)'
    y_label = 'Mean rank discharge (m$^3$/s)'
    out_fig_name = f'moving_window_rank_mean_cmpr_{col}.png'

    window_size = 1 + (365 * 4)  # Clipped to be odd.

    fig_size = (15, 7)

    data_ser = pd.read_csv(data_file, sep=sep, index_col=0)[col]

    data_ser.index = pd.to_datetime(data_ser.index, format=time_fmt)

    if rank_flag:
        data_ser = data_ser.rank()

    n_vals = data_ser.shape[0]

    assert np.all(np.isfinite(data_ser.values))

    if not (window_size % 2):
        window_size -= 1

    assert n_vals > window_size

    half_win = (window_size // 2)

    n_out_vals = n_vals - window_size - 1

    out_vals = []

    data_vals = data_ser.values.copy()

    print('window_size:', window_size)
    for i in range((window_size // 2), n_out_vals):
        subset_vals = data_vals[i - half_win:i + half_win + 1]

        n_subset_vals = subset_vals.size

        assert n_subset_vals == window_size

#         print('n_subset_vals:', subset_vals.size)

        out_vals.append(subset_vals.mean())

    out_ser = pd.Series(
        data=out_vals,
        index=data_ser.index[(window_size // 2): n_out_vals],
        name=col)

    win_min = out_ser.idxmin()
    win_max = out_ser.idxmax()

    plt.figure(figsize=fig_size)

    plt.plot(out_ser.index, out_ser.values, alpha=0.7, color='C0')

    plt.scatter(
        win_min, out_ser.loc[win_min], color='C1', alpha=0.8, label='Min.')

    plt.scatter(
        win_max, out_ser.loc[win_max], color='C2', alpha=0.8, label='Max.')

    plt.title(f'Moving window size: {window_size} steps')

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
