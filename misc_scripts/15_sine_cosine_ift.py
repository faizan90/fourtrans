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
from scipy.stats import rankdata, norm
import matplotlib.pyplot as plt

plt.ioff()

DEBUG_FLAG = True


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

    cosine_ift.sort()
    sine_ift.sort()

    return cosine_ift, sine_ift


def main():

    main_dir = Path(os.getcwd())
    os.chdir(main_dir)

    data_file = Path(r'P:\Synchronize\IWS\Testings\fourtrans_practice\phsann\hourly_bw_discharge__2008__2019.csv')

    beg_time = '2009-01-01'
    end_time = '2019-12-31'

    data_ser = pd.read_csv(
        data_file, sep=';', index_col=0).loc[beg_time:end_time, '420']

    data = data_ser.values

    if (data.size % 2):
        data = data[:-1]

    data_norms = norm.ppf(rankdata(data) / (data.size + 1.0))
    data_cos_ift, data_sin_ift = get_cos_sin_dists(data_norms)

#     data_cos_ift, data_sin_ift = get_cos_sin_dists(data)

    probs = np.arange(1.0, data_sin_ift.size + 1.0) / (data_sin_ift.size + 1)

    plt.figure()
    plt.title('Observed')
    plt.plot(data_cos_ift, probs, label='cos')
    plt.plot(data_sin_ift, probs, label='sin')

    plt.legend()
    plt.grid()

    plt.show(block=False)

    sim_norms = data_norms.copy()
    np.random.shuffle(sim_norms)
    sim_cos_ift, sim_sin_ift = get_cos_sin_dists(sim_norms)

#     sim_data = data.copy()
#     np.random.shuffle(sim_data)
#     sim_cos_ift, sim_sin_ift = get_cos_sin_dists(sim_data)

    plt.figure()
    plt.title('Simulated')
    plt.plot(sim_cos_ift, probs, label='cos')
    plt.plot(sim_sin_ift, probs, label='sin')

    plt.legend()
    plt.grid()

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
