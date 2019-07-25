'''
@author: Faizan-Uni-Stuttgart

'''
import os
import timeit
import time
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.ioff()


def main():

    mpl.rc('font', size=16)

    main_dir = Path(r'P:\Synchronize\IWS\fourtrans_practice\fourtrans_two_series_example')
    os.chdir(main_dir)

    in_file = (main_dir.parents[0]) / r'cat_411_qsims_valid_1961-06-01.npy'

    n_yrs = 10

    n_bef_steps = 4
    n_aft_steps = 10
    n_waves = 4
    n_smth_steps = 50

    in_arr = np.load(in_file)[730:730 + (365 * n_yrs), :2]

    if in_arr.shape[0] % 2:
        in_arr = in_arr[:-1, :]

    assert np.all(np.isfinite(in_arr))

    peak_val_idx = np.argmax(in_arr[:, 0])

    bef_idx = max(0, peak_val_idx - n_bef_steps)
    aft_idx = max(n_aft_steps, peak_val_idx + n_aft_steps)

    if aft_idx - bef_idx % 2:
        aft_idx += 1

    event_arr = in_arr[bef_idx:aft_idx + 1, :]
    assert not event_arr.shape[0] % 2

    n_event_steps = event_arr.shape[0]

    event_ft = np.fft.fft(event_arr, axis=0)

    n_ft = event_ft.shape[0]

    event_mags = np.abs(event_ft)
    event_phas = np.angle(event_ft)

    ofst = 1
    lrgst_wave_idxs = np.argsort(
        event_mags[ofst:n_event_steps // 2, 0])[::-1][:n_waves] + ofst

    axs = plt.subplots(nrows=2, ncols=2, figsize=(17, 9))[1]

    axs[0, 0].plot(event_arr[:, 0], label='ref')
    axs[0, 0].plot(event_arr[:, 1], label='sim')

    ref_corrs = (event_mags[lrgst_wave_idxs, 0] ** 2).cumsum()

    cos_diff = np.cos(
        event_phas[lrgst_wave_idxs, 0] - event_phas[lrgst_wave_idxs, 1])

    sim_corrs = (
        event_mags[lrgst_wave_idxs, 0] *
        event_mags[lrgst_wave_idxs, 1] *
        cos_diff).cumsum()

    sim_corrs /= ref_corrs[-1]
    ref_corrs /= ref_corrs[-1]

    axs[0, 1].plot(lrgst_wave_idxs, ref_corrs, label='ref')
    axs[0, 1].plot(lrgst_wave_idxs, sim_corrs, label='sim')

    wave_x_crds = np.arange(ofst, ofst + n_ft)
    for i in lrgst_wave_idxs:
        wave_ft = event_ft.copy()

        chop_off_idxs = np.ones(n_ft, dtype=bool)

        chop_off_idxs[i] = 0

        wave_ft[chop_off_idxs, :] = 0.0

        wave_arr = np.fft.ifft(wave_ft, axis=0)

        axs[1, 0].plot(
            wave_x_crds, wave_arr[:, 0].real, label=f'wave {i:02d}')

        axs[1, 1].plot(
            wave_x_crds, wave_arr[:, 1].real, label=f'wave {i:02d}')

# #     smth_arr = np.zeros((n_event_steps, 2))
#     smth_arr = np.zeros((n_smth_steps * n_event_steps, 2))
#     ft_cnst = 2 * np.pi * 1j
#
#     for j in range(2):
#         for k in range(n_ft):
#             for m in range(n_ft):
#                 for n in range(n_smth_steps):
#                     noise = (1.0 / n_ft) * (-0.5 + ((n + 1) / n_smth_steps))
#                     smth_arr[int((k * n_smth_steps) + n), j] += (
#                         event_ft[m, j] *
#                         np.exp(ft_cnst * (noise + (m * k / (n_ft))))
#                         ).real
#
# #                     smth_arr[int((k * n_smth_steps) + n), j] += (
# #                         event_ft[m, j] *
# #                         np.exp(ft_cnst * ((k * m)) / (n_ft))
# #                         ).real
#
# #                     print(noise)
#
#     axs[0, 1].plot(smth_arr[:, 0], label='ref')
#     axs[0, 1].plot(smth_arr[:, 1], label='sim')

    for i in range(2):
        for j in range(2):
            axs[i, j].grid()

    axs[0, 0].legend()
    axs[0, 0].set_xlabel('Time step (-)')
    axs[0, 0].set_ylabel('Discharge (m$^3$/s)')

    axs[0, 1].legend()
    axs[0, 1].set_xlabel('Frequency')
    axs[0, 1].set_ylabel('Cumm. correlation')
    axs[0, 1].set_xticks(lrgst_wave_idxs)
    axs[0, 1].set_xticklabels(axs[0, 1].get_xticks())

    axs[1, 0].set_xlabel('Frequency')
    axs[1, 1].set_xlabel(axs[1, 0].get_xlabel())

    axs[1, 0].set_ylabel('Magnitude')
    axs[1, 1].set_ylabel(axs[1, 0].get_ylabel())

    axs[1, 0].set_xlim(ofst, n_event_steps)
    axs[1, 1].set_xlim(*axs[1, 0].get_xlim())

    axs[1, 1].set_ylim(*axs[1, 0].get_ylim())

#     plt.show()

    plt.savefig(f'two_series_event_waves_corr.png', bbox_inches='tight')
    plt.close()

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
