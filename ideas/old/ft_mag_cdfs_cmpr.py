'''
@author: Faizan-Uni-Stuttgart

13 Feb 2020

13:51:36

'''
import os
import time
import timeit
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, expon, beta

plt.ioff()

DEBUG_FLAG = False


def get_cumm_ft_corr(in_vals):

    ft = np.fft.rfft(in_vals)[1:-1]

    ft_mags = np.abs(ft)

    ft_cov = ft_mags ** 2

    ft_corr_denom = (
        (ft_mags ** 2).sum() *
        (ft_mags ** 2).sum()) ** 0.5

    ft_corr = np.cumsum(ft_cov) / ft_corr_denom

    return ft_corr


def get_mag_cdf(in_vals):

    ft = np.fft.rfft(in_vals)[1:-1]

    ft_mags = np.abs(ft)

    ft_mags.sort()

    probs = np.arange(1, ft_mags.size + 1) / (ft_mags.size + 1)

    return ft_mags, probs


def get_mags(in_vals):

    ft = np.fft.rfft(in_vals)[1:-1]

    ft_mags = np.abs(ft)

    return ft_mags


def get_norm_mag_cdf(in_vals):

    probs_ser = (1 + np.argsort(np.argsort(in_vals))) / (in_vals.size + 1)

    norms_ser = norm.ppf(probs_ser)

    ft = np.fft.rfft(norms_ser)[1:-1]

    ft_mags = np.abs(ft)

    ft_mags.sort()

    probs = np.arange(1, ft_mags.size + 1) / (ft_mags.size + 1)

    return ft_mags, probs


def get_norm_mag_sin_cdf(in_vals):

    probs_ser = (1 + np.argsort(np.argsort(in_vals))) / (in_vals.size + 1)

    norms_ser = norm.ppf(probs_ser)

    ft = np.fft.rfft(norms_ser)[1:-1]

    ft_mags = np.abs(ft)
    ft_phs = np.angle(ft)

    ft_mags = ft_mags * np.sin(ft_phs)

    ft_mags.sort()

    probs = np.arange(1, ft_mags.size + 1) / (ft_mags.size + 1)

    return ft_mags, probs


def get_norm_mag_cos_cdf(in_vals):

    probs_ser = (1 + np.argsort(np.argsort(in_vals))) / (in_vals.size + 1)

    norms_ser = norm.ppf(probs_ser)

    ft = np.fft.rfft(norms_ser)[1:-1]

    ft_mags = np.abs(ft)
    ft_phs = np.angle(ft)

    ft_mags = ft_mags * np.cos(ft_phs)

    ft_mags.sort()

    probs = np.arange(1, ft_mags.size + 1) / (ft_mags.size + 1)

    return ft_mags, probs


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Testings\fourtrans_practice\phsann')
    os.chdir(main_dir)

    in_file_path = r'neckar_norm_cop_infill_discharge_1961_2015_20190118.csv'

    stn_no = '427'

    time_fmt = '%Y-%m-%d'

    sep = ';'

    beg_time_1 = '1999-01-01'
    end_time_1 = '1999-12-31'

    beg_time_2 = '1999-01-01'
    end_time_2 = '2000-12-31'

    in_df = pd.read_csv(in_file_path, index_col=0, sep=sep)
    in_df.index = pd.to_datetime(in_df.index, format=time_fmt)

    in_ser_1 = in_df.loc[beg_time_1:end_time_1, stn_no]
    in_vals_1 = in_ser_1.values

    in_ft_corr_1 = get_cumm_ft_corr(in_vals_1)

    ser_1_periods = 2 * in_ft_corr_1.size / np.arange(1, in_ft_corr_1.size + 1)

    in_ser_2 = in_df.loc[beg_time_2:end_time_2, stn_no]
    in_vals_2 = in_ser_2.values

    in_ft_corr_2 = get_cumm_ft_corr(in_vals_2)

    ser_2_periods = 2 * in_ft_corr_2.size / np.arange(1, in_ft_corr_2.size + 1)

    mags_1 = get_mags(in_vals_1)
    mags_2 = get_mags(in_vals_2)

    plt.plot(ser_1_periods, mags_1, alpha=0.6, label='S1')
    plt.plot(ser_2_periods, mags_2, alpha=0.6, label='S2')

#     plt.plot(ser_1_periods, in_ft_corr_1, alpha=0.6, label='S1')
#     plt.plot(ser_2_periods, in_ft_corr_2, alpha=0.6, label='S2')

#     plt.plot(in_ft_corr_1, alpha=0.6, label='S1')
#     plt.plot(in_ft_corr_2, alpha=0.6, label='S2')
#
#     plt.xlabel('period')
#
#     plt.ylabel('Cummulative correlation')

#     plt.plot(*get_mag_cdf(in_vals_1), alpha=0.6, label='S1')
#     plt.plot(*get_mag_cdf(in_vals_2), alpha=0.6, label='S2')
#
#     plt.xlabel('Magnitude')
#
#     plt.ylabel('Probability')

#     mag_dist_1, probs_1 = get_norm_mag_cdf(in_vals_1)
#     mag_dist_2, probs_2 = get_norm_mag_cdf(in_vals_2)
#
#     plt.plot(mag_dist_1, probs_1, alpha=0.6, label='S1')
#     plt.plot(mag_dist_2, probs_2, alpha=0.6, label='S2')

#     plt.plot(beta.ppf(probs_1, * beta.fit(mag_dist_1)), probs_1, alpha=0.6, label='ES1')
#     plt.plot(expon.ppf(probs_1, scale=mag_dist_1.mean()), probs_1, alpha=0.6, label='ES2')

#     plt.plot(expon.ppf(probs_1, scale=mag_dist_1.mean()), probs_1, alpha=0.6, label='ES1')
#     plt.plot(expon.ppf(probs_2, scale=mag_dist_2.mean()), probs_2, alpha=0.6, label='ES2')
#
#     plt.xlabel('Normal magnitude')
#
#     plt.ylabel('Probability')

#     mag_cos_dist_1, probs_cos_1 = get_norm_mag_cos_cdf(in_vals_1)
#     mag_cos_dist_2, probs_cos_2 = get_norm_mag_cos_cdf(in_vals_2)
#
#     plt.plot(mag_cos_dist_1, probs_cos_1, alpha=0.6, label='S1')
#     plt.plot(mag_cos_dist_2, probs_cos_2, alpha=0.6, label='S2')

    plt.grid()
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
