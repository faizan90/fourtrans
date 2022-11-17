'''
@author: Faizan-Uni-Stuttgart

22 May 2020

11:15:56

'''
import os
import time
import timeit
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
# from scipy.stats import norm

DEBUG_FLAG = True


def main():

    main_dir = Path(os.getcwd())
    os.chdir(main_dir)

    sep = ';'
    stn = '427'

    beg_time, end_time = '2009-01-01', '2009-12-30'

    in_daily_file = r"P:\Synchronize\IWS\Testings\fourtrans_practice\phsann\neckar_norm_cop_infill_discharge_1961_2015_20190118.csv"

    in_daily_ser = pd.read_csv(in_daily_file, sep=sep, index_col=0)[stn]

    in_daily_ser.index = pd.to_datetime(in_daily_ser.index, format='%Y-%m-%d')

    in_daily_ser = in_daily_ser.loc[beg_time:end_time]
    in_daily_vals = in_daily_ser.values

    # in_daily_ser.values[:] = norm.ppf(in_daily_ser.rank() / (in_daily_ser.size + 1.0))
    #
    # in_daily_ft = np.fft.rfft(in_daily_ser.values)

    n_pts = in_daily_ser.size
    ts_ft = np.zeros(n_pts, dtype=np.complex)
    for i in range(n_pts):
        for j in range(n_pts):
            ts_ft[i] += in_daily_vals[j] * np.exp((-2 * np.pi * i * j * 1j) / n_pts)

    ft_ts = np.zeros(n_pts, dtype=np.complex)
    for i in range(n_pts):
        for j in range(n_pts):
            ft_ts[j] += (ts_ft[i] * (1 / n_pts)) / np.exp((-2 * np.pi * i * j * 1j) / n_pts)

    sclr = 24
    ft_ts_hi = np.zeros(n_pts * sclr, dtype=np.complex)
    for i in range(n_pts):
    # for i in range(0, 360):
        for j in range(sclr // 2, n_pts * sclr - (sclr // 2)):
            ft_ts_hi[j] += (ts_ft[i] * (1 / n_pts)) / np.exp((-2 * np.pi * i * j * 1j) / (n_pts * sclr))

    # for i in range(n_pts):
    #     print(in_daily_vals[i], abs(ft_ts[i]))

    plt.plot(in_daily_vals, alpha=0.8, lw=4)
    plt.plot(np.abs(ft_ts), alpha=0.8, lw=2)
    plt.plot(np.linspace(0, n_pts, n_pts * sclr), np.abs(ft_ts_hi), alpha=0.8, lw=1)

    plt.show()

    tre = 1

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
