'''
@author: Faizan-Uni-Stuttgart

25 May 2020

23:30:17

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

DEBUG_FLAG = True


def get_mag_and_phs_spec(data):

    ft = np.fft.rfft(data)

    mag_spec = np.abs(ft)
    phs_spec = np.angle(ft)

    return mag_spec, phs_spec


def get_cumm_ft_corr(mag_spec):

    mag_spec_sq = mag_spec ** 2

    cumm_corr = np.cumsum(mag_spec_sq)
    cumm_corr /= cumm_corr[-1]

    return cumm_corr


def main():

    main_dir = Path(os.getcwd())
    os.chdir(main_dir)

    data_file = Path(r'P:\Synchronize\IWS\Testings\fourtrans_practice\phsann\neckar_norm_cop_infill_discharge_1961_2015_20190118.csv')

    beg_time = '2001-01-01'
    end_time = '2015-12-31'

    data_ser = pd.read_csv(
        data_file, sep=';', index_col=0).loc[beg_time:end_time, '420']

    data = data_ser.values

    if (data.size % 2):
        data = data[:-1]

    probs = rankdata(data) / (data.size + 1.0)

    norms = norm.ppf(probs)

    data_mag_spec, data_phs_spec = get_mag_and_phs_spec(data)

    probs_mag_spec, probs_phs_spec = get_mag_and_phs_spec(probs)
    norms_mag_spec, norms_phs_spec = get_mag_and_phs_spec(norms)

    rand_ft = np.zeros(data_mag_spec.size, dtype=np.complex128)

    rand_phss = -np.pi + (
        2 * np.pi * np.random.random(data_phs_spec.size - 2))

    rand_phs_spec = data_phs_spec.copy()
    rand_phs_spec[1:-1] = rand_phss  # * 0.001

    rand_ft.real = data_mag_spec * np.cos(rand_phs_spec)
    rand_ft.imag = data_mag_spec * np.sin(rand_phs_spec)

#     rand_phs_spec = probs_phs_spec.copy()
#     rand_phs_spec[1:-1] = rand_phss * 0.001
#
#     rand_ft.real = probs_mag_spec * np.cos(rand_phs_spec)
#     rand_ft.imag = probs_mag_spec * np.sin(rand_phs_spec)

    rand = np.fft.irfft(rand_ft)

    rand_probs = rankdata(rand) / (rand.size + 1.0)

    rand_norms = norm.ppf(rand_probs)

    rand_probs_mag_spec, rand_probs_phs_spec = get_mag_and_phs_spec(rand_probs)
    rand_norms_mag_spec, rand_norms_phs_spec = get_mag_and_phs_spec(rand_norms)

    data_cumm_corr = get_cumm_ft_corr(data_mag_spec[1:-1])
    probs_cumm_corr = get_cumm_ft_corr(probs_mag_spec[1:-1])
    norms_cumm_corr = get_cumm_ft_corr(norms_mag_spec[1:-1])
    rand_probs_cumm_corr = get_cumm_ft_corr(rand_probs_mag_spec[1:-1])
    rand_norms_cumm_corr = get_cumm_ft_corr(rand_norms_mag_spec[1:-1])

    periods = (data_cumm_corr.size * 2) / np.arange(1, data_cumm_corr.size + 1)

    plt.semilogx(periods, data_cumm_corr, alpha=0.7, lw=1.5, label='data')
    plt.semilogx(periods, probs_cumm_corr, alpha=0.7, lw=1.5, label='probs')
    plt.semilogx(periods, rand_probs_cumm_corr, alpha=0.7, lw=1.5, label='rand_probs')
    plt.semilogx(periods, norms_cumm_corr, alpha=0.7, lw=1.5, label='norms')
    plt.semilogx(periods, rand_norms_cumm_corr, alpha=0.7, lw=1.5, label='rand_norms')

    plt.grid()
    plt.legend()
    plt.xlim(plt.xlim()[::-1])

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
