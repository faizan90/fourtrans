'''
@author: Faizan-Uni-Stuttgart

27 May 2020

10:05:30

'''
import os
import time
import timeit
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm
import matplotlib.pyplot as plt

plt.ioff()

DEBUG_FLAG = False


def get_mag_and_phs_spec(data):

    ft = np.fft.rfft(data, axis=0)

    mag_spec = np.abs(ft)
    phs_spec = np.angle(ft)

    return mag_spec, phs_spec


def get_cumm_ft_corr(mag_spec):

    mag_spec_sq = mag_spec ** 2

    cumm_corr = np.cumsum(mag_spec_sq)
    cumm_corr /= cumm_corr[-1]

    return cumm_corr


def get_data_and_probs_cumm_corr(data):

    if (data.size % 2):
        data = data[:-1]

    probs = rankdata(data) / (data.size + 1.0)

#     norms = norm.ppf(probs)

    data_mag_spec, data_phs_spec = get_mag_and_phs_spec(data)

    probs_mag_spec, probs_phs_spec = get_mag_and_phs_spec(probs)

    data_cumm_corr = get_cumm_ft_corr(data_mag_spec[1:-1])
    probs_cumm_corr = get_cumm_ft_corr(probs_mag_spec[1:-1])

    return data_cumm_corr, probs_cumm_corr


def main():

    main_dir = Path(
        r'P:\Synchronize\IWS\Projects\2016_DFG_SPATE\data\phase_annealed_simulations\for_probs_and_data_ft_cmprs_02')

    os.chdir(main_dir)

    ref_data_file = Path(r'ref_data.csv')
    sim_data_file = Path(r'sim_data_0.csv')

    out_fig_name = 'ref_sim_cumm_corr_cmpr.png'

    col_idx = 0

    fig_size = (15, 10)

    ref_data = np.loadtxt(
        ref_data_file, delimiter=';', ndmin=2, skiprows=1)[:, col_idx]

    sim_data = np.loadtxt(
        sim_data_file, delimiter=';', ndmin=2, skiprows=1)[:, col_idx]

    ref_data_cumm_corr, ref_probs_cumm_corr = get_data_and_probs_cumm_corr(ref_data)
    sim_data_cumm_corr, sim_probs_cumm_corr = get_data_and_probs_cumm_corr(sim_data)

    periods = (ref_data_cumm_corr.size * 2) / np.arange(1, ref_data_cumm_corr.size + 1)

    plt.figure(figsize=fig_size)

    plt.semilogx(
        periods, ref_data_cumm_corr, alpha=0.7, lw=1.5, label='ref_data')

    plt.semilogx(
        periods, ref_probs_cumm_corr, alpha=0.7, lw=1.5, label='ref_probs')

    plt.semilogx(
        periods, sim_data_cumm_corr, alpha=0.7, lw=1.5, label='sim_data')

    plt.semilogx(
        periods, sim_probs_cumm_corr, alpha=0.7, lw=1.5, label='sim_probs')

    plt.grid()
    plt.legend()
    plt.xlim(plt.xlim()[::-1])

    plt.savefig(out_fig_name, bbox_inches='tight')

#     plt.show()

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
