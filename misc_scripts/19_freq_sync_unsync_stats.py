'''
@author: Faizan-Uni-Stuttgart

Dec 21, 2020

11:45:30 AM

'''
import os
import time
import timeit
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import rankdata, norm, skew, kurtosis

from phsann.misc import roll_real_2arrs
from phsann.cyth import get_asymms_sample, fill_bi_var_cop_dens

asymms_exp = 3.0

plt.ioff()

DEBUG_FLAG = True


def get_mag_and_phs_spec(data):

    ft = np.fft.rfft(data)

    mag_spec = np.abs(ft)
    phs_spec = np.angle(ft)

    return mag_spec, phs_spec


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Testings\fourtrans_practice\phsann')
    os.chdir(main_dir)

    data_file = Path(
        r'neckar_norm_cop_infill_discharge_1961_2015_20190118.csv')

    beg_time = '1961-01-01'
    end_time = '2015-12-31'

    col = '427'

    data_ser = pd.read_csv(
        data_file, sep=';', index_col=0).loc[beg_time:end_time, col]

    data = data_ser.values

    if (data.size % 2):
        data = data[:-1]

    data_probs = rankdata(data) / (data.size + 1.0)
    data_norms = norm.ppf(data_probs)

    data_mag_spec, data_phs_spec = get_mag_and_phs_spec(data_norms)

    rand_phs_spec = np.full_like(data_phs_spec, np.nan)
    rand_phs_spec[1:-1] = -np.pi + (2 * np.pi * np.random.random(rand_phs_spec.size - 2))
    rand_phs_spec[+0] = data_phs_spec[+0]
    rand_phs_spec[-1] = data_phs_spec[-1]

    ift_data = np.full((data_mag_spec.size - 2, data.size), np.nan, dtype=float)
    ift_rand = ift_data.copy()

    for i in range(ift_data.shape[0]):

        ft_sub = np.zeros_like(data_mag_spec, dtype=complex)

        ft_sub.real[i + 1] = data_mag_spec[i + 1] * np.cos(data_phs_spec[i + 1])
        ft_sub.imag[i + 1] = data_mag_spec[i + 1] * np.sin(data_phs_spec[i + 1])

#         ft_sub.real[i + 1] = np.cos(data_phs_spec[i + 1])
#         ft_sub.imag[i + 1] = np.sin(data_phs_spec[i + 1])

        ift_sub = np.fft.irfft(ft_sub)

        ift_data[i] = ift_sub

        ft_sub_rand = np.zeros_like(data_mag_spec, dtype=complex)

        ft_sub_rand.real[i] = data_mag_spec[i + 1] * np.cos(rand_phs_spec[i + 1])
        ft_sub_rand.imag[i] = data_mag_spec[i + 1] * np.sin(rand_phs_spec[i + 1])

#         ft_sub_rand.real[i] = np.cos(rand_phs_spec[i + 1])
#         ft_sub_rand.imag[i] = np.sin(rand_phs_spec[i + 1])

        ift_sub_rand = np.fft.irfft(ft_sub_rand)

        ift_rand[i] = ift_sub_rand

    ift_data_means = ift_data.mean(axis=0)
#     ift_data_stds = ift_data.std(axis=0)

    ift_rand_means = ift_rand.mean(axis=0)
#     ift_rand_stds = ift_rand.std(axis=0)

    probs = np.arange(
        1.0, ift_data_means.size + 1.0) / (ift_data_means.size + 1.0)

    plt.plot(np.sort(ift_data_means), probs, alpha=0.5, label='data_mean')
#     plt.plot(np.sort(ift_data_stds), probs, alpha=0.5, label='data_std')
    plt.plot(np.sort(ift_rand_means), probs, alpha=0.5, label='rand_mean')
#     plt.plot(np.sort(ift_rand_stds), probs, alpha=0.5, label='rand_std')

    plt.scatter([0.0], [0.5], c='k', alpha=0.75)

    plt.grid()
    plt.gca().set_axisbelow(True)

    plt.legend()

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
