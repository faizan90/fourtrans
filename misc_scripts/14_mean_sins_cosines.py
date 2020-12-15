'''
@author: Faizan-Uni-Stuttgart

Dec 15, 2020

9:55:01 AM

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


def show_data_phs_spec_props(phs_spec, vb=False):

    tan_spec = np.tan(phs_spec)
    cos_spec = np.cos(phs_spec)
    sin_spec = np.sin(phs_spec)

    tan_spec_mean = tan_spec.mean()
    cos_spec_mean = cos_spec.mean()
    sin_spec_mean = sin_spec.mean()
    sin_cos_spec_mean = sin_spec_mean / cos_spec_mean

    if vb:
        print('Mean Tangent:', round(tan_spec_mean, 4))
        print('Mean Cosine:', round(cos_spec_mean, 4))
        print('Mean Sine:', round(sin_spec_mean, 4))
        print('Mean Sin/Cos:', round(sin_cos_spec_mean, 4))

    return tan_spec_mean, cos_spec_mean, sin_spec_mean, sin_cos_spec_mean


def main():

    main_dir = Path(os.getcwd())
    os.chdir(main_dir)

    data_file = Path(r'P:\Synchronize\IWS\Testings\fourtrans_practice\phsann\hourly_bw_discharge__2008__2019.csv')

    beg_time = '2009-01-01'
    end_time = '2019-12-31'

    data_ser = pd.read_csv(
        data_file, sep=';', index_col=0).loc[beg_time:end_time, '3465']

    n_sims = 10000

    data = data_ser.values

    if (data.size % 2):
        data = data[:-1]

    data_norms = norm.ppf(rankdata(data) / (data.size + 1.0))

    _, data_phs_spec = get_mag_and_phs_spec(data_norms)

    print('Data Props:')

    data_props = show_data_phs_spec_props(data_phs_spec[1:-1], True)

    print('\n')

    print('Sim Props:')

    sim_props = np.full((n_sims, 4), np.nan)
    for i in range(n_sims):
        rand_phs_spec = (
            -np.pi + (2 * np.pi * np.random.random(data_phs_spec.size - 2)))

        sim_props[i, :] = show_data_phs_spec_props(rand_phs_spec)

    assert np.all(np.isfinite(sim_props))

    sim_props = np.sort(sim_props, axis=0)

    probs = np.arange(1.0, n_sims + 1.0) / (n_sims + 1.0)

    ttls = ['Tan', 'Cos', 'Sin', 'Sin/Cos']
    for i in range(sim_props.shape[1]):
        plt.figure()

        plt.plot(sim_props[:, i], probs, alpha=0.75)

        data_prob = np.interp(
            data_props[i], sim_props[:, i], probs, left=0.0, right=1.0)

        plt.scatter([data_props[i]], [data_prob])

        plt.title(ttls[i])

        plt.show(block=False)

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
