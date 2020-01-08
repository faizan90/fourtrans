'''
@author: Faizan-Uni-Stuttgart

Jan 8, 2020

2:21:48 PM

'''
import os
import time
import timeit
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

plt.ioff()

DEBUG_FLAG = True


def main():

    main_dir = Path(os.getcwd())
    os.chdir(main_dir)

    scorrs = np.linspace(-1.0, 1.0, 1000)
    a_maxs = (0.5 * (1 - scorrs)) * (1 - ((0.5 * (1 - scorrs)) ** (1.0 / 3.0)))

    print(a_maxs)

    a_maxs_max_idx = np.argmax(a_maxs)
    print(a_maxs[a_maxs_max_idx], scorrs[a_maxs_max_idx])
    plt.scatter(scorrs, a_maxs, alpha=0.2)

    plt.xlabel('Spearman correlation')
    plt.ylabel('A$_{max}$')

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
