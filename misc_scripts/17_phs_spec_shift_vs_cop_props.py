'''
@author: Faizan-Uni-Stuttgart

Dec 17, 2020

10:12:53 AM

'''
import os
import time
import timeit
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata, norm

from phsann.misc import roll_real_2arrs
from phsann.cyth import get_asymms_sample, fill_bi_var_cop_dens

asymms_exp = 3.0

plt.ioff()

DEBUG_FLAG = False


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

    return asymm_1, asymm_2, etpy


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
    end_time = '1980-12-31'

#     beg_time = '1981-01-01'
#     end_time = '2000-12-31'

#     beg_time = '1981-06-01'
#     end_time = '2001-05-30'

    col = '427'

    lag_steps = np.arange(1, 10, dtype=np.int64)

    n_pi_intvls = 100

    n_cop_bins = 20

    phs_sclrs_exp = 0.0

    plot_asymm_1_flag = True
    plot_asymm_2_flag = True
    plot_etpy_flag = True

#     plot_asymm_1_flag = False
#     plot_asymm_2_flag = False
#     plot_etpy_flag = False

    assert any([plot_asymm_1_flag, plot_asymm_2_flag, plot_etpy_flag])

    data_ser = pd.read_csv(
        data_file, sep=';', index_col=0).loc[beg_time:end_time, col]

    pi_incr = 2 * np.pi / n_pi_intvls

    print('pi_incr:', round(pi_incr, 3))

    data = data_ser.values

    if (data.size % 2):
        data = data[:-1]

    data_norms = norm.ppf(rankdata(data) / (data.size + 1.0))

    data_mag_spec, data_phs_spec = get_mag_and_phs_spec(data_norms)

    rows = int(ceil(lag_steps.size ** 0.5))
    cols = ceil(lag_steps.size / rows)

    _, axes = plt.subplots(rows, cols, squeeze=False, sharex=True, sharey=True)

    phs_sclrs = (
        (1 / np.arange(1, data_phs_spec.size - 1)) ** phs_sclrs_exp) * pi_incr

#     phs_sclrs = (
#         (1 / (np.arange(1, data_phs_spec.size - 1) ** phs_sclrs_exp))) * pi_incr

#     concat_range = (np.arange(1, int(0.5 * data_phs_spec.size + 1)))
#     concat_range = np.concatenate((concat_range[1:][::-1], concat_range))
#     phs_sclrs = ((1 / concat_range) ** phs_sclrs_exp) * pi_incr

    print(
        'phs_sclrs min, max:',
        round(phs_sclrs.min(), 3),
        round(phs_sclrs.max(), 3))

    row = 0
    col = 0
    for i in range(rows * cols):
        phs_spec = data_phs_spec.copy()

        data_props = np.full((n_pi_intvls, 4), np.nan)

        for j in range(n_pi_intvls):
            if j:
                phs_spec[1:-1] += phs_sclrs

            norms = get_irfted_vals(data_mag_spec, phs_spec)

            probs = rankdata(norms) / (data.size + 1.0)

            probs, rolled_probs = roll_real_2arrs(
                probs, probs, lag_steps[i])

            data_props[j, :] = [
                j * pi_incr,
                *get_probs_copula_props(probs, rolled_probs, n_cop_bins)]

        data_props[:, 3] -= data_props[0, 3]  # .reshape(-1, data_props_mins.size)

        if i >= (lag_steps.size):
            axes[row, col].set_axis_off()

        else:
            # Asymmetry 1.
            if plot_asymm_1_flag:
                min_asymm_1_idx = np.argmin(data_props[:, 1])

                axes[row, col].scatter(
                    data_props[:, 0],
                    data_props[:, 1],
                    alpha=0.5,
                    label='asymm_1')

                axes[row, col].scatter(
                    [data_props[min_asymm_1_idx, 0]],
                    [data_props[min_asymm_1_idx, 1]],
                    alpha=0.9,
                    c='k')

            # Asymmetry 2.
            if plot_asymm_2_flag:
                min_asymm_2_idx = np.argmin(data_props[:, 2])

                axes[row, col].scatter(
                    data_props[:, 0],
                    data_props[:, 2],
                    alpha=0.5,
                    label='asymm_2')

                axes[row, col].scatter(
                    [data_props[min_asymm_2_idx, 0]],
                    [data_props[min_asymm_2_idx, 2]],
                    alpha=0.9,
                    c='k')

            # Entropy.
            if plot_etpy_flag:
                min_etpy_idx = np.argmin(data_props[:, 3])

                axes[row, col].scatter(
                    data_props[:, 0], data_props[:, 3], alpha=0.5, label='etpy')

                axes[row, col].scatter(
                    [data_props[min_etpy_idx, 0]],
                    [data_props[min_etpy_idx, 3]],
                    alpha=0.9,
                    c='k')

            axes[row, col].grid()

#             axes[row, col].set_aspect('equal')

            axes[row, col].text(
                0.05,
                0.9,
                f'{lag_steps[i]} step(s) lag',
                alpha=0.5)

            if col:
#                 axes[row, col].set_yticklabels([])
                pass

            else:
                axes[row, col].set_ylabel('Property')

            if row < (rows - 1):
                pass
#                 axes[row, col].set_xticklabels([])

            else:
                axes[row, col].set_xlabel('Phase')

            if i == 0:
                axes[row, col].legend()

        col += 1
        if not (col % cols):
            row += 1
            col = 0

#         plt.title(f'Lag: {lag_step}')
#     #     plt.scatter(data_props[:, 0], data_props[:, 1], alpha=0.5, label='asymm_1')
#     #     plt.scatter(data_props[:, 0], data_props[:, 2], alpha=0.5, label='asymm_2')
#         plt.scatter(data_props[:, 0], data_props[:, 3], alpha=0.5, label='etpy')
#
#         plt.grid()
#         plt.gca().set_axisbelow(True)

#     plt.legend()

    plt.suptitle(f'phs_sclrs_exp: {phs_sclrs_exp}')
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
