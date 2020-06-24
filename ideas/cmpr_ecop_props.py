'''
@author: Faizan-Uni-Stuttgart

8 Jun 2020

14:35:05

'''
import os
import time
import timeit
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
import matplotlib.pyplot as plt

from phsann.cyth import get_asymms_sample, fill_bi_var_cop_dens
from phsann.misc import roll_real_2arrs

plt.ioff()

DEBUG_FLAG = True


def _get_asymm_1_max(scorr):

    a_max = (
        0.5 * (1 - scorr)) * (1 - ((0.5 * (1 - scorr)) ** (1.0 / 3.0)))

    return a_max


def _get_asymm_2_max(scorr):

    a_max = (
        0.5 * (1 + scorr)) * (1 - ((0.5 * (1 + scorr)) ** (1.0 / 3.0)))

    return a_max


def _get_etpy_min(n_bins):

    dens = 1 / n_bins

    etpy = -np.log(dens)

    return etpy


def _get_etpy_max(n_bins):

    dens = (1 / (n_bins ** 2))

    etpy = -np.log(dens)

    return etpy


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Testings\fourtrans_practice\phsann')
    os.chdir(main_dir)

    in_file = Path(r'neckar_norm_cop_infill_discharge_1961_2015_20190118.csv')

    lag_steps = np.arange(1, 31, dtype=np.int64)

    ecop_bins = 20

    fig_size = (15, 10)
    plt_alpha = 0.5

    in_ser = pd.read_csv(in_file, sep=';', index_col=0)['420']

    in_ser.index = pd.to_datetime(in_ser.index, format='%Y-%m-%d')

    idx_labs = np.unique(in_ser.index.year)

    etpy_min = _get_etpy_min(ecop_bins)
    etpy_max = _get_etpy_max(ecop_bins)

    ecop_dens_arrs = np.full((ecop_bins, ecop_bins), np.nan, dtype=np.float64)

    fig, axes = plt.subplots(2, 3, squeeze=False, figsize=fig_size)

    cmap = 'jet'
    sim_clrs = plt.get_cmap(cmap)(
        (idx_labs - idx_labs.min()) / (idx_labs.max() - idx_labs.min()))

    sim_clrs = {idx_lab:sim_clr for (idx_lab, sim_clr) in zip(idx_labs, sim_clrs)}

    cmap_mappable_beta = plt.cm.ScalarMappable(cmap=cmap)

    cmap_mappable_beta.set_array([])

    for idx_lab in idx_labs[::5]:
        data = in_ser.loc[
            f'{idx_lab}-01-01':f'{idx_lab+5}-12-31'].values

        probs = rankdata(data) / (data.size + 1.0)

        scorrs = []
        asymms_1 = []
        asymms_2 = []
        etpys = []
        pcorrs = []
        for lag_step in lag_steps:
            probs_i, rolled_probs_i = roll_real_2arrs(probs, probs, lag_step)
            data_i, rolled_data_i = roll_real_2arrs(data, data, lag_step)

            # scorr.
            scorr = np.corrcoef(probs_i, rolled_probs_i)[0, 1]
            scorrs.append(scorr)

            # asymms.
            asymm_1, asymm_2 = get_asymms_sample(probs_i, rolled_probs_i)

            asymm_1 /= _get_asymm_1_max(scorr)

            asymm_2 /= _get_asymm_2_max(scorr)

            asymms_1.append(asymm_1)
            asymms_2.append(asymm_2)

            # ecop etpy.
            fill_bi_var_cop_dens(probs_i, rolled_probs_i, ecop_dens_arrs)

            non_zero_idxs = ecop_dens_arrs > 0

            dens = ecop_dens_arrs[non_zero_idxs]

            etpy_arr = -(dens * np.log(dens))

            etpy = etpy_arr.sum()

            etpy = (etpy - etpy_min) / (etpy_max - etpy_min)

            etpys.append(etpy)

            # pcorr.
            pcorr = np.corrcoef(data_i, rolled_data_i)[0, 1]
            pcorrs.append(pcorr)

        # plot
        axes[0, 0].plot(
            lag_steps, scorrs, alpha=plt_alpha, color=sim_clrs[idx_lab])

        axes[1, 0].plot(
            lag_steps, asymms_1, alpha=plt_alpha, color=sim_clrs[idx_lab])

        axes[1, 1].plot(
            lag_steps, asymms_2, alpha=plt_alpha, color=sim_clrs[idx_lab])

        axes[0, 1].plot(
            lag_steps, etpys, alpha=plt_alpha, color=sim_clrs[idx_lab])

        axes[0, 2].plot(
            lag_steps, pcorrs, alpha=plt_alpha, color=sim_clrs[idx_lab])

    axes[0, 0].grid()
    axes[1, 0].grid()
    axes[1, 1].grid()
    axes[0, 1].grid()
    axes[0, 2].grid()
    axes[1, 2].grid()

    axes[0, 0].set_ylabel('Spearman correlation')

    axes[1, 0].set_xlabel('Lag steps')
    axes[1, 0].set_ylabel('Asymmetry (Type - 1)')

    axes[1, 1].set_xlabel('Lag steps')
    axes[1, 1].set_ylabel('Asymmetry (Type - 2)')

    axes[0, 1].set_ylabel('Entropy')

    axes[0, 2].set_xlabel('Lag steps')
    axes[0, 2].set_ylabel('Pearson correlation')

#     axes[1, 2].set_xlabel('Nth orders')
#     axes[1, 2].set_ylabel('Dist. Sum')

#     cbaxes = fig.add_axes([0.2, 0.0, 0.65, 0.05])

    plt.colorbar(
        mappable=cmap_mappable_beta,
        cax=axes[1, 2],
        orientation='horizontal',
        label='Relative Timing',
        alpha=plt_alpha,
        drawedges=False)

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
