'''
Nov 29, 2018
@author: Faizan-Uni-Stuttgart

'''
import os
import timeit
import time
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def plot_ts(obsrd_ts, model_ts, fig_size, out_fig):

    plt.figure(figsize=fig_size)

    plt.plot(model_ts, alpha=0.7, color='k', label='Model', lw=2)

    plt.plot(obsrd_ts, alpha=0.7, color='r', label='Observed', lw=2)

#     plt.title(
#         f'Correlation contribution per fourier frequency\n'
#         f'n_sims: {n_sims}, n_steps: {n_steps}')

    plt.xlabel('Time (days)')
    plt.ylabel('Discharge (m$^3$/s)')

    plt.legend()

    plt.grid()
    plt.savefig(out_fig, bbox_inches='tight')

    plt.close()

    return


def main():

    main_dir = Path(r'P:\Synchronize\IWS\2016_DFG_SPATE\Presentations\20190226_Stuttgart')
    os.chdir(main_dir)

#     in_file = r'cat_411_qsims_valid_1961-06-01.csv'
    in_file = r'cat_411_qsims_valid_1961-06-01.npy'

    out_fig = r'ft_corrs_contrib.png'

    n_yrs = 2
#     sep = ';'
    take_sims = 1

    scale = 1

    max_freqs = 200

    fig_size = (20, 9)

#     in_arr = np.loadtxt(
#         in_file,
#         delimiter=sep,
#         skiprows=1,
#         dtype=float)[365:365 + (365 * n_yrs), :]

    in_arr = np.load(in_file)[730:730 + (365 * n_yrs), :]

    n_steps = in_arr.shape[0]
    n_sims = min(take_sims, in_arr.shape[1])

    print(f'n_steps: {n_steps}, n_sims: {n_sims}')

#     obs_probs = (np.argsort(np.argsort(in_arr[:, 0])) + 1) / (n_steps + 1)
#     obs_norms = norm.ppf(obs_probs)

    obs_norms = in_arr[:, 0]

    obs_ft = np.fft.fft(obs_norms)

    obs_phis = np.angle(obs_ft)

    obs_amps = np.abs(obs_ft)

    use_obs_phis = obs_phis[1: (n_steps // 2)]
    use_obs_amps = obs_amps[1: (n_steps // 2)]

    plt.figure(figsize=fig_size)

    mpl.rc('font', size=16)

    for i in range(1, n_sims + 1):
#         sim_probs = (np.argsort(np.argsort(in_arr[:, i])) + 1) / (n_steps + 1)
#
#         sim_norms = norm.ppf(sim_probs)

        sim_norms = in_arr[:, i] * scale

        sim_ft = np.fft.fft(sim_norms)

        sim_phis = np.angle(sim_ft)

        sim_amps = np.abs(sim_ft)

        use_sim_phis = sim_phis[1: (n_steps // 2)]
        use_sim_amps = sim_amps[1: (n_steps // 2)]

        indiv_cov_arr = (use_obs_amps * use_sim_amps) * (
            np.cos(use_obs_phis - use_sim_phis))

        tot_cov = indiv_cov_arr.sum()

        cumm_rho_arr = indiv_cov_arr.cumsum() / tot_cov

        ft_corr = tot_cov / (
            ((use_obs_amps ** 2).sum() * (use_sim_amps ** 2).sum()) ** 0.5)

        print(i, ft_corr)

        plt.plot((cumm_rho_arr * ft_corr)[:max_freqs], alpha=0.7, color='k', label='Model', lw=2)

    indiv_cov_arr = (use_obs_amps * use_obs_amps) * (
        np.cos(use_obs_phis - use_obs_phis))

    tot_cov = indiv_cov_arr.sum()

    cumm_rho_arr = indiv_cov_arr.cumsum() / tot_cov

    ft_corr = tot_cov / (
        ((use_obs_amps ** 2).sum() * (use_obs_amps ** 2).sum()) ** 0.5)

    print(0, ft_corr)

    plt.plot((cumm_rho_arr * ft_corr)[:max_freqs], alpha=0.7, color='r', label='Observed', lw=2)

#     plt.title(
#         f'Correlation contribution per fourier frequency\n'
#         f'n_sims: {n_sims}, n_steps: {n_steps}')

    plt.xlabel('Frequency no.')
    plt.ylabel('Contribution (-)')

    plt.legend()

    plt.grid()
    plt.ylim(0, 1)
    plt.savefig(out_fig, bbox_inches='tight')

    plt.close()

    plot_ts(obs_norms, sim_norms, fig_size, 'ft_sim.png')

    mpl.rcdefaults()
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
    START = timeit.default_timer()  # to get the runtime of the program

    main()

    STOP = timeit.default_timer()  # Ending time
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))

    if _save_log_:
        log_link.stop()
