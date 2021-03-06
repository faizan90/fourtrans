'''
@author: Faizan-Uni-Stuttgart

'''
import os
import time
import timeit
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

plt.ioff()


def get_corr_mat(phas):

    n_phas, = phas.shape

    corr_mat = np.empty((n_phas, n_phas), dtype=float)

    for j in range(n_phas):
        for k in range(n_phas):
            corr_mat[j, k] = np.cos(phas[j] - phas[k])

    return corr_mat


def main():

    main_dir = Path(
        r'P:\Synchronize\IWS\Testings\fourtrans_practice'
        r'\unit_function_time_shift')

    os.chdir(main_dir)

    n_vals = 100

    cen_idx = int(0.2 * n_vals)

    width = 0.1

    t_shift = 80

#     fig_suff = (
#         f'_test')

    fig_suff = (
        f'_cen_{cen_idx}_wid_{int(n_vals * width)}_'
        f'shift_{t_shift}')

    in_arr = np.zeros(n_vals)

    in_arr[
        cen_idx - int(n_vals * width * 0.5):
        cen_idx + int(n_vals * width * 0.5)] = 1

    ft = np.fft.fft(in_arr)

    pwr_spec = np.abs(ft)
    phs_spec = np.angle(ft)

    ft_shift = ft.copy()
    ft_shift *= (
        np.exp(-1j * 2 * np.pi *
               (np.arange(float(ft.shape[0])) / ft.shape[0]) * t_shift))

    phs_spec_shift = np.angle(ft_shift)

    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))

    axs[0, 0].plot(in_arr, alpha=0.9, label='input')
    axs[0, 0].plot(np.fft.ifft(ft_shift).real, alpha=0.9, label='output')
    axs[0, 0].grid()
    axs[0, 0].legend()

    for i in range(1, 7):
        wav_arr = np.zeros(ft.shape[0], dtype=np.complex)
        wav_arr[i] = ft[i]

        axs[0, 1].plot(np.fft.fft(wav_arr).real, alpha=0.9, label=f'{i}')

        wav_shift_arr = np.zeros(ft_shift.shape[0], dtype=np.complex)
        wav_shift_arr[i] = ft_shift[i]

        axs[1, 1].plot(np.fft.fft(wav_shift_arr).real, alpha=0.9, label=f'{i}')

    axs[0, 1].grid()
    axs[0, 1].legend()

    axs[1, 1].grid()
    axs[1, 1].legend()

    corr_mat = get_corr_mat(phs_spec)
    corr_mat_shift = get_corr_mat(phs_spec_shift)

    fig.colorbar(
        axs[0, 2].imshow(corr_mat, cmap='jet'),
        ax=axs[0, 2],
        label='phase correlation (orig)')

    fig.colorbar(
        axs[1, 2].imshow(corr_mat_shift, cmap='jet'),
        ax=axs[1, 2],
        label='phase correlation (shift)')

    axs[1, 0].plot(pwr_spec, alpha=0.9, label='power-orig')
    axs[1, 0].plot(np.abs(ft_shift), alpha=0.9, label='power-shift')
    axs[1, 0].grid()
    axs[1, 0].legend()

    plt.suptitle(
        f'Time shift FT comparison (shift: {t_shift} step(s))')

    plt.savefig(f'time_shift_ft{fig_suff}.png', bbox_inches='tight')

    plt.close()

#     plt.show()

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

#     try:
#         main()
#
#     except:
#         import pdb
#         pdb.post_mortem()

    main()

    STOP = timeit.default_timer()
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))

    if _save_log_:
        log_link.stop()
