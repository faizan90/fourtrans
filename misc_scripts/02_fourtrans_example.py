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
from scipy.interpolate import interp1d

plt.ioff()


def main():

    main_dir = Path(r'P:\Synchronize\IWS\2016_DFG_SPATE\Presentations\20190226_Stuttgart')
    os.chdir(main_dir)

#     ref_arr = np.array([0, 2, 1, 1, 2, 0])
    ref_arr = np.array([0, 1, 1, 1, 2, 0])

    ft = np.fft.rfft(ref_arr)

    phas = np.angle(ft)

    n_frows, n_fcols = 2, 2
    _fig, axes = plt.subplots(n_frows, n_fcols, figsize=(20, 9))

    mpl.rc('font', size=16)

    ift_x = np.arange(0, ref_arr.shape[0], dtype=float)
    ift_x_interp = np.linspace(0, ref_arr.shape[0] - 1, 500)

    ift_wave_sum = np.zeros(ref_arr.shape[0])

    for i in range(ft.shape[0]):
        i_ft = np.zeros(ft.shape[0], dtype=complex)

        i_ft[i] = ft[i]

        i_ift = np.fft.irfft(i_ft)

        ift_wave_sum += i_ift

        axes[0, 1].plot(i_ift, label=f'wave {i}')

        ift_interp_ftn = interp1d(ift_x, i_ift, kind='quadratic')
        i_ift_interp = ift_interp_ftn(ift_x_interp)

        axes[1, 1].plot(ift_x_interp, i_ift_interp, label=f'wave {i}')

        i_waves = np.zeros(ft.shape[0], dtype=complex)
        i_wave_mags = np.zeros(ft.shape[0])

        i_wave_mags[i] = 1

        i_waves.real = i_wave_mags * np.cos(phas)
        i_waves.imag = i_wave_mags * np.sin(phas)

        i_waves = np.fft.irfft(i_waves)

        print(i_waves)

        i_wave_interp_ftn = interp1d(ift_x, i_waves, kind='quadratic')
        i_wave_ift_interp = i_wave_interp_ftn(ift_x_interp)

        if i:
            i_wave_ift_interp = (i_wave_ift_interp - i_wave_ift_interp.min()) / (
                i_wave_ift_interp.max() - i_wave_ift_interp.min())

        else:
            i_wave_ift_interp[:] = 0.5

        axes[1, 0].plot(ift_x_interp, i_wave_ift_interp, label=f'wave {i}')

    axes[0, 0].plot(ref_arr, label='observed')

    axes[0, 1].plot(ift_wave_sum, label=f'waves sum')

    titles_arr = np.array([
        ['Observed series',
         'Absolute frequency contribution (unsmoothed)'],
        ['Relative frequencies (smoothed)',
         'Absolute frequency contribution (smoothed)']])

    for i in range(n_frows):
        for j in range(n_fcols):
            axes[i, j].grid()

            if not i:
                axes[i, j].legend()

            axes[i, j].set_title(titles_arr[i, j])

            if i == (n_frows - 1):
                axes[i, j].set_xlabel('Time step')

            if not j:
                axes[i, j].set_ylabel('Value')

#     plt.show()

    plt.savefig('test_four_trans_example.png', bbox_inches='tight')
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
