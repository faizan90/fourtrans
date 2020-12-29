'''
@author: Faizan-Uni-Stuttgart

Dec 22, 2020

7:22:35 PM

'''
import os
import time
import timeit
from pathlib import Path

import numpy as np
import pandas as pd
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

#     dens = 1 / n_bins
#
#     etpy = -np.log(dens)

    etpy = 0

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
#     data_norms = norm.ppf(data_probs_orig)

    data_probs, data_probs_rolled = roll_real_2arrs(
            data_probs_orig, data_probs_orig, 1)

    data_asymm = (data_probs - data_probs_rolled) ** asymms_exp

    asymm_mag_spec, asymm_phs_spec = get_mag_and_phs_spec(data_asymm)

    rand_phs_spec = np.empty_like(asymm_phs_spec)

    rand_phs_spec[1:-1] = -np.pi + (
        2 * np.pi * np.random.random(asymm_phs_spec.size - 2))

    rand_phs_spec[+0] = asymm_phs_spec[+0]
    rand_phs_spec[-1] = asymm_phs_spec[-1]

    rand_asymm_ft = np.empty_like(asymm_phs_spec, dtype=complex)
    rand_asymm_ft.real = np.cos(rand_phs_spec) * asymm_mag_spec
    rand_asymm_ft.imag = np.sin(rand_phs_spec) * asymm_mag_spec

    rand_asymm = np.fft.irfft(rand_asymm_ft)

    plt.plot(data_asymm, alpha=0.7, c='red', label='obs')
    plt.plot(rand_asymm, alpha=0.7, c='k', label='rand')
    plt.show()

    print(get_probs_copula_props(
        data_probs, data_probs_rolled, 20))

    data_asymm_probs_orig = rankdata(data_asymm) / (data_asymm.size + 1.0)
    data_asymm_probs, data_asymm_probs_rolled = roll_real_2arrs(
            data_asymm_probs_orig, data_asymm_probs_orig, 1)

    print(get_probs_copula_props(
        data_asymm_probs, data_asymm_probs_rolled, 20))

    rand_probs_orig = rankdata(rand_asymm) / (rand_asymm.size + 1.0)
    rand_probs, rand_probs_rolled = roll_real_2arrs(
            rand_probs_orig, rand_probs_orig, 1)

    print(get_probs_copula_props(
        rand_probs, rand_probs_rolled, 20))

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
