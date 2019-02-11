'''
@author: Faizan-Uni-Stuttgart

'''
import os
import timeit
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata, norm

plt.ioff()


def sim_surr_trunc(ref_arr, ref_lt_val, niters=1):

    if ref_arr.shape[0] % 2:
        ref_arr = ref_arr[:-1]

    ref_n_vals = ref_arr.shape[0]

    ref_n_lt_vals = (ref_arr < ref_lt_val).sum()
    ref_lt_prob = ref_n_lt_vals / (ref_n_vals + 1.0)

    ref_ranks = rankdata(ref_arr)
    ref_probs = ref_ranks / (ref_n_vals + 1.0)
    ref_probs[ref_probs < ref_lt_prob] = ref_lt_prob

    # truncated
    ref_norms = norm.ppf(ref_probs)

    ref_norm_ft = np.fft.fft(ref_norms)
    ref_norm_mags = np.abs(ref_norm_ft)
    ref_norm_angs = np.angle(ref_norm_ft)

    rands = np.random.random((ref_n_vals // 2) - 1)
    rand_angls = -np.pi + (2 * np.pi * rands)

    rand_ph_angs = np.concatenate((
        [ref_norm_angs[0]],
        rand_angls,
        [ref_norm_angs[ref_n_vals // 2]],
        rand_angls[::-1] * -1))

    # just for consistent naming, N sim and ref vals are equal
    sim_n_vals = ref_n_vals
    sim_lt_prob = ref_lt_prob

    min_diff_iter = 0
    diff_sq_sum_min = np.inf
    old_mags = ref_norm_mags.copy()

    for i in range(niters):
        sim_norm_ft = np.full(sim_n_vals, np.nan, dtype=complex)
        sim_norm_ft.real = old_mags * np.cos(rand_ph_angs)
        sim_norm_ft.imag = old_mags * np.sin(rand_ph_angs)

        sim_norm_ift = np.fft.ifft(sim_norm_ft).real

        sim_norm_ift_sort = np.sort(sim_norm_ift)

        sim_norm_lt_val = sim_norm_ift_sort[int(sim_lt_prob * sim_n_vals)]

        sim_norm_trnc = sim_norm_ift.copy()
        sim_norm_trnc[
            sim_norm_trnc < sim_norm_lt_val] = sim_norm_lt_val

        sim_norm_trnc_ft = np.fft.fft(sim_norm_trnc)
        sim_norm_trnc_ft_mags = np.abs(sim_norm_trnc_ft)

        ref_sim_mags_diff = sim_norm_trnc_ft_mags[1:] - ref_norm_mags[1:]
        ref_sim_mags_diff_sq_sum = (ref_sim_mags_diff ** 2).sum()

        # ref_sim_mags_diff[:sim_n_vals // 2] = 0

        if ref_sim_mags_diff_sq_sum < diff_sq_sum_min:
            min_diff_iter = i
            diff_sq_sum_min = ref_sim_mags_diff_sq_sum
            opt_mags = sim_norm_trnc_ft_mags.copy()

        old_mags[1:] = old_mags[1:] - ref_sim_mags_diff
        old_mags[old_mags < 0] = ref_norm_mags[old_mags < 0]

        print(
            f'{i:03d}, ',
            f'{min_diff_iter:03d}, ',
            f'{ref_sim_mags_diff_sq_sum:+13.5f}, ',
            f'{diff_sq_sum_min:+13.5f}')

    print('\n')

    sim_norm_opt_ft = np.zeros(sim_n_vals, dtype=complex)
    sim_norm_opt_ft.real = opt_mags * np.cos(rand_ph_angs)
    sim_norm_opt_ft.imag = opt_mags * np.sin(rand_ph_angs)

    sim_norm_opt_ift = np.fft.ifft(sim_norm_opt_ft).real

    sim_norm_opt_ift_idxs = (rankdata(sim_norm_opt_ift) - 1).astype(int)

    ref_sort = np.sort(ref_arr)
    sim_trnc_arr = ref_sort[sim_norm_opt_ift_idxs]

    ref_corrs = []
    sim_corrs = []
    for i in range(30):
        ref_corr = np.corrcoef(ref_arr, np.roll(ref_arr, i))[0, 1]
        sim_trnc_corr = np.corrcoef(
            sim_trnc_arr, np.roll(sim_trnc_arr, i))[0, 1]

        ref_corrs.append(ref_corr)
        sim_corrs.append(sim_trnc_corr)
        print(f'{i:03d}, ', f'{ref_corr:+0.5f}, ', f'{sim_trnc_corr:+0.5f}')

    ref_corrs = np.array(ref_corrs)
    sim_corrs = np.array(sim_corrs)

    plt.figure(figsize=(15, 7))
    plt.plot(ref_corrs, alpha=0.4, label='Ref.')
    plt.plot(sim_corrs, alpha=0.4, label='Sim.')

    plt.grid()
    plt.legend()

    plt.show()

    plt.close()

    return sim_trnc_arr


def main():

    main_dir = Path(r'P:\Synchronize\IWS\fourtrans_practice')
    os.chdir(main_dir)

    in_file = Path(r'P:\Synchronize\IWS\QGIS_Neckar\hydmod\input_hyd_data\neckar_daily_discharge_1961_2015.csv')
    stn = '454'
    thresh = 70

#     in_file = Path(r'P:\Synchronize\IWS\DWD_meteo_hist_pres\full_neckar_ppt_norm_cop_infill_1961_to_2015_20190117\02_combined_station_outputs\infilled_var_df.csv')
#     stn = 'P1028'
#     thresh = 1e-1

    beg_date = '1961-01-01'
    end_date = '2015-12-31'

    n_vals = 365 * 20
#     n_corr_lags = 30
#
#     figs_dir = 'test_surr_gen'
    fig_size = (15, 7)

    data = pd.read_csv(in_file, sep=';', index_col=0)[stn].loc[beg_date:end_date].values[:n_vals]

    assert not np.isnan(data).sum()

    sim_arr = sim_surr_trunc(data, thresh, 15)

    plt.figure(figsize=fig_size)
    plt.plot(data, alpha=0.4, label='ref.')
    plt.plot(sim_arr, alpha=0.4, label='sim.')

    plt.grid()
    plt.legend()

    plt.show()

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
    START = timeit.default_timer()  # to get the runtime of the program

    main()

    STOP = timeit.default_timer()  # Ending time
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))

    if _save_log_:
        log_link.stop()
