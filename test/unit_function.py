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

    main_dir = Path(r'P:\Synchronize\IWS\Testings\fourtrans_practice')
    os.chdir(main_dir)

    fig_suff = '_peak_at_50_long'
    n_vals = 100
    in_arr = np.zeros(n_vals)

    in_arr[int(n_vals * 0.2): int(n_vals * 0.8)] = 1

    ft = np.fft.rfft(in_arr)

    pwr_spec = np.abs(ft)
    phs_spec = np.angle(ft)

    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))

    axs[0, 0].plot(in_arr, alpha=0.9, label='input')
    axs[0, 0].grid()
    axs[0, 0].legend()

    for i in range(1, 7):
        wav_arr = np.zeros(ft.shape[0], dtype=np.complex)
        wav_arr[i] = ft[i]

        axs[0, 1].plot(np.fft.irfft(wav_arr), alpha=0.9, label=f'{i}')

    axs[0, 1].grid()
    axs[0, 1].legend()

    fig.colorbar(
        axs[0, 2].imshow(get_corr_mat(phs_spec), cmap='jet'),
        ax=axs[0, 2],
        label='phase correlation')

    axs[1, 0].plot(pwr_spec, alpha=0.9, label='power')
    axs[1, 0].grid()
    axs[1, 0].legend()

    axs[1, 1].plot(phs_spec, alpha=0.9, label='phase')
    axs[1, 1].grid()
    axs[1, 1].legend()

    plt.suptitle('FT of the unit function')

    plt.savefig(f'unit_ft{fig_suff}.png', bbox_inches='tight')

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

    try:
        main()

    except:
        import pdb
        pdb.post_mortem()

    STOP = timeit.default_timer()
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))

    if _save_log_:
        log_link.stop()
