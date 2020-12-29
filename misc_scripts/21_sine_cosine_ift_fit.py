'''
@author: Faizan-Uni-Stuttgart

Dec 15, 2020

9:55:41 AM

'''
import os
import time
import timeit
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import rankdata, norm
import matplotlib.pyplot as plt

from phsann.misc import roll_real_2arrs
from phsann.cyth import get_asymms_sample, fill_bi_var_cop_dens

asymms_exp = 3.0

plt.ioff()

DEBUG_FLAG = False


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


def get_mag_and_phs_spec(data):

    ft = np.fft.rfft(data)

    mag_spec = np.abs(ft)
    phs_spec = np.angle(ft)

    return mag_spec, phs_spec


def get_cos_sin_dists(data):

    mag_spec, phs_spec = get_mag_and_phs_spec(data)

    cosine_ft = np.zeros(mag_spec.size, dtype=complex)
    cosine_ft.real = mag_spec * np.cos(phs_spec)
    cosine_ift = np.fft.irfft(cosine_ft)

    sine_ft = np.zeros(mag_spec.size, dtype=complex)
    sine_ft.imag = mag_spec * np.sin(phs_spec)
    sine_ift = np.fft.irfft(sine_ft)

#     cosine_ift.sort()
#     sine_ift.sort()

    return cosine_ift, sine_ift


def main():

    main_dir = Path(os.getcwd())
    os.chdir(main_dir)

    data_file = Path(
        r'P:\Synchronize\IWS\Testings\fourtrans_practice\phsann\neckar_norm_cop_infill_discharge_1961_2015_20190118.csv')

    beg_time = '2009-01-01'
    end_time = '2019-12-31'

    data_ser = pd.read_csv(
        data_file, sep=';', index_col=0).loc[beg_time:end_time, '420']

    data = data_ser.values

    if (data.size % 2):
        data = data[:-1]

    data_probs_orig = rankdata(data) / (data.size + 1.0)
    data_norms = norm.ppf(data_probs_orig)
    data_cos_ift_nosort, data_sin_ift_nosort = get_cos_sin_dists(data_norms)

    data_probs, data_probs_rolled = roll_real_2arrs(
            data_probs_orig, data_probs_orig, 1)

    print(get_probs_copula_props(
        data_probs, data_probs_rolled, 20))

#     data_cos_ift, data_sin_ift = get_cos_sin_dists(data)

    data_cos_ift, data_sin_ift = np.sort(data_cos_ift_nosort), np.sort(data_sin_ift_nosort)

    probs = np.arange(1.0, data_sin_ift.size + 1.0) / (data_sin_ift.size + 1)

    plt.figure()
    plt.title('Observed')
    plt.plot(data_cos_ift, probs, label='cos')
    plt.plot(data_sin_ift, probs, label='sin')

    plt.legend()
    plt.grid()

    plt.show(block=False)

    rand_probs = np.random.random(probs.size)
#     rand_probs = data_probs_orig.copy()

    cos_ftn = interp1d(
        probs,
        data_cos_ift,
        fill_value=(data_cos_ift.min(), data_cos_ift.max()),
        bounds_error=False)

#     rand_coss = np.sort(cos_ftn(rand_probs))[
#         np.argsort(np.argsort(data_cos_ift_nosort))]

    rand_coss = cos_ftn(rand_probs)

    sin_ftn = interp1d(
        probs,
        data_sin_ift,
        fill_value=(data_sin_ift.min(), data_sin_ift.max()),
        bounds_error=False)

#     rand_sins = np.sort(sin_ftn(rand_probs))[
#         np.argsort(np.argsort(data_sin_ift_nosort))]

    rand_sins = sin_ftn(rand_probs)

#     rand_ft = np.full_like(data_cos_ift, np.nan, dtype=complex)
#
#     rand_ft.real = rand_coss
#     rand_ft.imag = rand_sins

    rand_ift = rand_coss + rand_sins  # np.fft.irfft(rand_ft)

    rand_probs = rankdata(rand_ift) / (rand_ift .size + 1.0)

    rand_probs, rand_probs_rolled = roll_real_2arrs(
            rand_probs, rand_probs, 1)

    print(get_probs_copula_props(
        rand_probs, rand_probs_rolled, 20))

    print(np.corrcoef(data_norms, rand_ift)[0, 1])

#     sim_norms = data_norms.copy()
#     np.random.shuffle(sim_norms)
#     sim_cos_ift, sim_sin_ift = get_cos_sin_dists(sim_norms)
#
# #     sim_data = data.copy()
# #     np.random.shuffle(sim_data)
# #     sim_cos_ift, sim_sin_ift = get_cos_sin_dists(sim_data)
#
#     plt.figure()
#     plt.title('Simulated')
#     plt.plot(np.sort(sim_cos_ift), probs, label='cos')
#     plt.plot(np.sort(sim_sin_ift), probs, label='sin')
#
#     plt.plot(np.sort(rand_coss), probs, label='rand_cos')
#
#     plt.legend()
#     plt.grid()
#
#     plt.show()

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
