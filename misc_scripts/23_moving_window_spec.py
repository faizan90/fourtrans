'''
@author: Faizan-Uni-Stuttgart

Jan 6, 2021

2:35:44 PM

'''
import os
import time
import timeit
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, rankdata

plt.ioff()

DEBUG_FLAG = False


def get_mag_and_phs_spec(data):

    ft = np.fft.rfft(data)

    mag_spec = np.abs(ft)
    phs_spec = np.angle(ft)

    return mag_spec, phs_spec


def get_smoothed_arr(in_arr, win_size, smooth_ftn_type):

    n_vals = in_arr.shape[0]

    smooth_ftn = getattr(np, smooth_ftn_type)

    smoothed_arr = np.zeros(n_vals - win_size + 1)
    for i in range(smoothed_arr.size):
        smoothed_arr[i] = smooth_ftn(in_arr[i:i + win_size])

    return smoothed_arr


def main():

    main_dir = Path(os.getcwd())
    os.chdir(main_dir)

    data_file = Path(
        r'P:\Synchronize\IWS\Testings\fourtrans_practice\phsann\neckar_norm_cop_infill_discharge_1961_2015_20190118.csv')

    beg_time = '1961-01-01'
    end_time = '2015-12-31'

    n_sims = 100

    data_ser = pd.read_csv(
        data_file, sep=';', index_col=0).loc[beg_time:end_time, '420']

    data = data_ser.values

    if (data.size % 2):
        data = data[:-1]

    probs = rankdata(data) / (data.size + 1.0)
    norms = norm.ppf(probs)

    mag_spec, phs_spec = get_mag_and_phs_spec(norms)

    phs_spec_smoothed = get_smoothed_arr(np.cos(phs_spec[1:-1]) * mag_spec[1:-1], 500, 'mean')

    periods = np.arange(phs_spec_smoothed.size, 2)

    leg_flag = True
    for _ in range(n_sims):
        phs_spec_rand = -np.pi + (2 * np.pi * np.random.random(phs_spec.size))
        phs_spec_rand_smoothed = get_smoothed_arr(np.cos(phs_spec_rand[1:-1]) * mag_spec[1:-1], 500, 'mean')

        if leg_flag:
            leg_flag = False
            label = 'sim'

        else:
            label = None

        plt.plot(phs_spec_rand_smoothed, alpha=0.3, lw=1, c='k', label=label)

    plt.plot(phs_spec_smoothed, alpha=0.7, label='ref', lw=2, c='r')

    plt.legend()

    plt.grid()
    plt.gca().set_axisbelow(True)

    plt.show()

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
