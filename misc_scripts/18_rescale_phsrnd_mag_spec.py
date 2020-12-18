'''
@author: Faizan-Uni-Stuttgart

Dec 17, 2020

5:21:07 PM

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


def get_max_ft_corr(ref_mag, sim_mag):

    numr = (ref_mag[1:-1] * sim_mag[1:-1])

    demr = (
        ((ref_mag[1:-1] ** 2).sum() ** 0.5) *
        ((sim_mag[1:-1] ** 2).sum() ** 0.5))

    return np.cumsum(numr) / demr


def get_mag_and_phs_spec(data):

    ft = np.fft.rfft(data)

    mag_spec = np.abs(ft)
    phs_spec = np.angle(ft)

    return mag_spec, phs_spec


def get_asymm_1_max(scorr):

    a_max = (
        0.5 * (1 - scorr)) * (1 - ((0.5 * (1 - scorr)) ** (1.0 / asymms_exp)))

    return a_max


def get_asymm_2_max(scorr):

    a_max = (
        0.5 * (1 + scorr)) * (1 - ((0.5 * (1 + scorr)) ** (1.0 / asymms_exp)))

    return a_max


def get_etpy_min(n_bins):

    dens = 1 / n_bins

    etpy = -np.log(dens)

    return etpy


def get_etpy_max(n_bins):

    dens = (1 / (n_bins ** 2))

    etpy = -np.log(dens)

    return etpy


def get_probs_copula_props(probs, probs_rolled, n_ecop_bins):

    scorr = np.corrcoef(probs, probs_rolled)[0, 1]

    asymm_1, asymm_2 = get_asymms_sample(probs, probs_rolled)
    asymm_1 /= get_asymm_1_max(scorr)
    asymm_2 /= get_asymm_2_max(scorr)

#     plt.scatter(probs, probs_rolled, alpha=0.5)
#     plt.grid()
#     plt.show()
#     plt.close()

    ecop_dens_arr = np.full(
        (n_ecop_bins, n_ecop_bins),
        np.nan,
        dtype=np.float64)

    fill_bi_var_cop_dens(probs, probs_rolled, ecop_dens_arr)

    non_zero_idxs = ecop_dens_arr > 0

    dens = ecop_dens_arr[non_zero_idxs]

    etpy_arr = -(dens * np.log(dens))

    etpy = etpy_arr.sum()

    etpy_min = get_etpy_min(n_ecop_bins)
    etpy_max = get_etpy_max(n_ecop_bins)

    etpy = (etpy - etpy_min) / (etpy_max - etpy_min)

    return asymm_1, asymm_2, etpy, scorr


def get_irfted_vals(mag_spec, phs_spec):

    ft = np.full(mag_spec.size, np.nan, dtype=complex)

    ft.real = mag_spec * np.cos(phs_spec)
    ft.imag = mag_spec * np.sin(phs_spec)

    ift = np.fft.irfft(ft)

    return ift


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Testings\fourtrans_practice\phsann')
    os.chdir(main_dir)

    data_file = Path(
        r'neckar_norm_cop_infill_discharge_1961_2015_20190118.csv')

    beg_time = '1961-01-01'
    end_time = '1961-12-31'

#     beg_time = '1981-01-01'
#     end_time = '2000-12-31'

#     beg_time = '1981-06-01'
#     end_time = '2001-05-30'

    col = '427'

    n_sims = 100

    lag = 5
    n_ecop_bins = 20

    data_ser = pd.read_csv(
        data_file, sep=';', index_col=0).loc[beg_time:end_time, col]

    data = data_ser.values

    if (data.size % 2):
        data = data[:-1]

    data_probs = rankdata(data) / (data.size + 1.0)
    data_norms = norm.ppf(data_probs)

    data_probs, data_probs_rolled = roll_real_2arrs(
            data_probs, data_probs, lag)

    print(get_probs_copula_props(
        data_probs, data_probs_rolled, n_ecop_bins))

    sorted_data_norms = np.sort(data_norms)

    data_mag_spec, data_phs_spec = get_mag_and_phs_spec(data_norms)

    sim_phs_spec = np.full_like(data_phs_spec, np.nan)
    sim_phs_spec[+0] = data_phs_spec[+0]
    sim_phs_spec[-1] = data_phs_spec[-1]

    periods = (data_mag_spec.size * 2) / (
        np.arange(1, data_mag_spec.size + 1))

    plt.figure()
    leg_flag = True
    sim_props = np.full((n_sims, 8), np.nan)
    for i in range(n_sims):

        sim_phs_spec[1:-1] = -np.pi + (
            2 * np.pi * np.random.random(data_phs_spec.size - 2))

        sim_norms = get_irfted_vals(data_mag_spec, sim_phs_spec)

        sim_props[i, :2] = skew(sim_norms), kurtosis(sim_norms)

        re_sim_data = sorted_data_norms[np.argsort(np.argsort(sim_norms))]

        re_sim_mag_spec, re_sim_phs_spec = get_mag_and_phs_spec(re_sim_data)
#         re_sim_mag_spec, re_sim_phs_spec = get_mag_and_phs_spec(sim_norms)

        sim_probs = rankdata(re_sim_data) / (re_sim_data.size + 1.0)
#         sim_probs = rankdata(sim_norms) / (sim_norms.size + 1.0)

        sim_probs, sim_probs_rolled = roll_real_2arrs(
                sim_probs, sim_probs, lag)

        sim_props[i, 2:6] = get_probs_copula_props(
            sim_probs, sim_probs_rolled, n_ecop_bins)

        if leg_flag:
            leg_flag = False
            label = 'sim'

        else:
            label = None

#         plt.semilogx(
#             periods,
#             re_sim_mag_spec.cumsum(),
#             label=label,
#             color='k',
#             alpha=0.3,
#             lw=1)

        plt.semilogx(
            get_max_ft_corr(data_mag_spec, re_sim_mag_spec),
            label=label,
            color='k',
            alpha=0.3,
            lw=1)

        mag_sq_diff = ((data_mag_spec - re_sim_mag_spec) ** 2).sum()
        phs_sq_diff = ((data_phs_spec - re_sim_phs_spec) ** 2).sum()

        sim_props[i, 6:] = mag_sq_diff, phs_sq_diff

#     plt.semilogx(
#         periods,
#         data_mag_spec.cumsum(),
#         label='ref',
#         color='r',
#         alpha=0.75,
#         lw=2)

    plt.semilogx(
        get_max_ft_corr(data_mag_spec, data_mag_spec),
        label='ref',
        color='r',
        alpha=0.75,
        lw=2)

#     plt.xlim(plt.xlim()[::-1])
    plt.grid()
    plt.legend(loc=4)

    print(sim_props.min(axis=0))
    print(sim_props.max(axis=0))

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
