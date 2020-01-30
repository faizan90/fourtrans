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


def get_sines(beg_val, mult, n_vals, n_tiles):

    pis = -np.pi + ((2 * np.pi) * np.linspace(0, 1, n_vals))
    pis = np.tile(pis, n_tiles)

    sines = np.roll(mult * np.sin(pis), beg_val)

    return sines


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Presentations\Phase_Annealing_data')
    os.chdir(main_dir)

    # NOTE:
    # number of vals in ser_1 and ser_2 at the end the is equal i.e.
    # n_vals_1 * n_tiles_1 == n_vals_2 * n_tiles_2
    ser_1 = get_sines(1000, 1.2, 5000, 1)
    ser_2 = get_sines(0, 1, 2500, 2)

    ref_arr = ser_1 + ser_2

    ft = np.fft.rfft(ref_arr)

    mag_spec = np.abs(ft)
    phs_spec = np.angle(ft)

    mag_idxs = np.argsort(mag_spec)[::-1]

    plt.rcParams.update({'font.size': 20})

    _fig, axes = plt.subplots(1, 2, figsize=(16, 9), squeeze=False)

    axes[0, 0].plot(ser_1, label='series 1', lw=2, alpha=0.7)
    axes[0, 0].plot(ser_2, label='series 2', lw=2, alpha=0.7)
    axes[0, 0].plot(ref_arr, label='series 1 + 2', lw=4, alpha=0.7)

    axes[0, 0].grid()

    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Magnitude')

    axes[0, 0].legend()

    for mag_idx in mag_idxs[:4]:
        ft_sub = np.zeros_like(ft)
        ft_sub[mag_idx] = ft[mag_idx]

        ift = np.fft.irfft(ft_sub)

        axes[0, 1].plot(ift.real, label=f'freq: {mag_idx}', lw=2, alpha=0.7)

    axes[0, 1].grid()

    axes[0, 1].set_xlabel('Step')

    axes[0, 1].set_ylim(*axes[0, 0].get_ylim())

    axes[0, 1].legend()

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
