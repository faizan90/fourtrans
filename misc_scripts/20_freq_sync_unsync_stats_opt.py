'''
@author: Faizan-Uni-Stuttgart

Dec 21, 2020

11:45:30 AM

'''
import os
import time
import timeit
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import differential_evolution
from scipy.stats import rankdata, norm

plt.ioff()

DEBUG_FLAG = True


def get_mag_and_phs_spec(data):

    ft = np.fft.rfft(data)

    mag_spec = np.abs(ft)
    phs_spec = np.angle(ft)

    return mag_spec, phs_spec


def get_freq_ifts(mag_spec, phs_spec):

    ift_data = np.full(
        (mag_spec.size - 2, (mag_spec.size * 2) - 2), np.nan, dtype=float)

    for i in range(ift_data.shape[0]):

        ft_sub = np.zeros_like(mag_spec, dtype=complex)

        ft_sub.real[i + 1] = mag_spec[i + 1] * np.cos(phs_spec[i + 1])
        ft_sub.imag[i + 1] = mag_spec[i + 1] * np.sin(phs_spec[i + 1])

#         ft_sub.real[i + 1] = np.cos(phs_spec[i + 1])
#         ft_sub.imag[i + 1] = np.sin(phs_spec[i + 1])

        ift_sub = np.fft.irfft(ft_sub)

        ift_data[i] = ift_sub

    return ift_data


def obj_ftn(sim_phs_spec, mag_spec, ref_vals, obj_wts):

    ifts = get_freq_ifts(mag_spec, sim_phs_spec)

    sim_vals = np.sort(ifts.mean(axis=0))

    obj_val = (((ref_vals - sim_vals) ** 2) * obj_wts).sum()

#     obj_val = ((ref_vals - sim_vals) ** 2).sum()

    print(obj_val)

    return obj_val


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Testings\fourtrans_practice\phsann')
    os.chdir(main_dir)

    data_file = Path(
        r'neckar_norm_cop_infill_discharge_1961_2015_20190118.csv')

    beg_time = '1961-01-01'
    end_time = '2000-12-31'

    col = '427'

    opt_flag = False

    plot_stats = ['mean', 'std']

    n_sims = 100

    max_iters = 1
    pop_size = 1

    data_ser = pd.read_csv(
        data_file, sep=';', index_col=0).loc[beg_time:end_time, col]

    data = data_ser.values

    if (data.size % 2):
        data = data[:-1]

    probs = np.arange(1.0, data.size + 1.0) / (data.size + 1.0)

    obj_wts = ((1 / probs.size) / (1 - probs)) ** 2

    data_probs = rankdata(data) / (data.size + 1.0)
    data_norms = norm.ppf(data_probs)

    data_mag_spec, data_phs_spec = get_mag_and_phs_spec(data_norms)

    ift_data = get_freq_ifts(data_mag_spec, data_phs_spec)

#     for row in range(10):  # (ift_data.shape[0]):
#         plt.plot(ift_data[row,:], alpha=0.3)
#
#     plt.show()

    if 'mean' in plot_stats:
        ift_data_means = ift_data.mean(axis=0)

    if 'std' in plot_stats:
        ift_data_stds = ift_data.std(axis=0)

    if opt_flag:
        bds = np.empty((data_phs_spec.size, 2))
        bds[:, 0] = -np.pi
        bds[:, 1] = +np.pi

        opt_ress = differential_evolution(
            obj_ftn,
            bds,
            args=(data_mag_spec, ift_data_means, obj_wts),
            maxiter=max_iters,
            popsize=pop_size,
            polish=False)

        sim_phs_spec = opt_ress.x
        sim_phs_spec[+0] = data_phs_spec[+0]
        sim_phs_spec[-1] = data_phs_spec[-1]

        ift_sim = get_freq_ifts(data_mag_spec, sim_phs_spec)

        if 'mean' in plot_stats:
            ift_sim_means = ift_sim.mean(axis=0)
            plt.plot(
                np.sort(ift_sim_means), probs, alpha=0.5, label='sim_mean')

        if 'std' in plot_stats:
            ift_sim_stds = ift_sim.std(axis=0)
            plt.plot(np.sort(ift_sim_stds), probs, alpha=0.5, label='sim_std')

    else:
        leg_flag = False
        for i in range(n_sims):
            print('Rand. sim.:', i)

            rand_phs_spec = np.full_like(data_phs_spec, np.nan)

            rand_phs_spec[1:-1] = -np.pi + (
                2 * np.pi * np.random.random(rand_phs_spec.size - 2))

            rand_phs_spec[+0] = data_phs_spec[+0]
            rand_phs_spec[-1] = data_phs_spec[-1]

            ift_rand = get_freq_ifts(data_mag_spec, rand_phs_spec)

            if leg_flag:
                leg_flag = False
                mean_lab = 'rand_mean'
                std_lab = 'rand_std'

            else:
                mean_lab = None
                std_lab = None

            if 'mean' in plot_stats:
                ift_rand_means = ift_rand.mean(axis=0)
                plt.plot(
                    np.sort(ift_rand_means),
                    probs,
                    alpha=0.1,
                    label=mean_lab,
                    c='k')

            if 'std' in plot_stats:
                ift_rand_stds = ift_rand.std(axis=0)
                plt.plot(
                    np.sort(ift_rand_stds),
                    probs,
                    alpha=0.1,
                    label=std_lab,
                    c='k')

    if 'mean' in plot_stats:
        plt.plot(
            np.sort(ift_data_means),
            probs,
            alpha=0.75,
            label='data_mean',
            lw=2)

    if 'std' in plot_stats:
        plt.plot(
            np.sort(ift_data_stds),
            probs,
            alpha=0.75,
            label='data_std',
            lw=2)

    plt.grid()
    plt.gca().set_axisbelow(True)

    plt.legend()

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
