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


def main():

    main_dir = Path(os.getcwd())
    os.chdir(main_dir)

    n_vals = 1000
    t_shift = 2

    x = np.exp(-1j * 2 * np.pi * (
            np.arange(float(n_vals)) / n_vals))

    y = np.exp(-1j * 2 * np.pi * t_shift * (
            np.arange(float(n_vals)) / n_vals))

    plt.plot(x.real, alpha=0.8, label='ref.real')
    plt.plot(y.real, alpha=0.8, label='dst.real')
    plt.plot(x.imag, alpha=0.8, label='ref.imag')
    plt.plot(y.imag, alpha=0.8, label='dst.imag')

    plt.xlabel('Frequency')
    plt.ylabel('Adjustment factor')

    plt.grid()
    plt.legend()

    plt.title(f'Time shift sinusoids ({t_shift} step(s) shift)')

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
