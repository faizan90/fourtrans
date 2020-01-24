'''
@author: Faizan-Uni-Stuttgart

Jan 23, 2020

12:26:11 PM

'''
import os
import time
import timeit
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DEBUG_FLAG = True

plt.ioff()


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Testings\fourtrans_practice\phsann')
    os.chdir(main_dir)

    in_file_path = r'neckar_norm_cop_infill_discharge_1961_2015_20190118.csv'

    stn_no = '427'

    time_fmt = '%Y-%m-%d'

    sep = ';'

    beg_time = '1999-01-01'
    end_time = '1999-12-31'

    in_df = pd.read_csv(in_file_path, index_col=0, sep=sep)
    in_df.index = pd.to_datetime(in_df.index, format=time_fmt)

    in_ser = in_df.loc[beg_time:end_time, stn_no]

    in_vals = in_ser.values

    if in_vals.size % 2:
        in_vals = in_vals[:-1]

    ft = np.fft.rfft(in_vals)

#     phs = np.angle(ft[1:])

    pwr = np.abs(ft[1:])

    pwr_dens = pwr / pwr.sum()

    print(pwr_dens.sum())

    x = np.arange(pwr_dens.size)
    y1 = np.zeros(x.size)
    y2 = pwr_dens

    plt.fill_between(x, y1, y2, alpha=0.5)

    plt.xlabel('Frequency index')
    plt.ylabel('Frequency power density')

    plt.grid()

    plt.show()

    plt.close()

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
