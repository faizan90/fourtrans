'''
@author: Faizan-Uni-Stuttgart

Dec 2, 2020

12:38:57 PM

'''

import os
import gc
import time
import timeit
from shutil import copy2
from pathlib import Path

os.environ[str('MKL_NUM_THREADS')] = str(1)
os.environ[str('NUMEXPR_NUM_THREADS')] = str(1)
os.environ[str('OMP_NUM_THREADS')] = str(1)

import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt

# raise Exception

plt.ioff()

DEBUG_FLAG = True


def main():

    main_dir = Path(
        r'P:\Synchronize\IWS\Testings\fourtrans_practice\multisite_phs_spec_corr\5min\v7')

    os.chdir(main_dir)

    # order matters: cos, sin
    # orig is copied as the new ouputs with overwritten values.
    # orig itself gets its time and units updated.
    parts = [
        'cos/cos.nc',
        'sin/sin.nc',
        ]

    var = 'OK'

    # Output related.

    out_dir = Path('phss')

    # NOTE: It is a copy of a reference file. Only the data values are updated.
    out_nc = out_dir / r'phss.nc'

    assert len(parts) == 2

    out_dir.mkdir(exist_ok=True)

    coss = sins = None
    for i, part in enumerate(parts):

        print(i, part)

        nc_hdl = nc.Dataset(part, 'r')

        if i == 0:
            coss = nc_hdl[var][...].data

        elif i == 1:
            sins = nc_hdl[var][...].data

        else:
            raise ValueError

        nc_hdl.close()

    phss = np.arctan2(sins, coss)

    copy2(parts[0], out_nc)

    nc_hdl = nc.Dataset(out_nc, 'r+')

    nc_hdl[var][:] = phss

    nc_hdl.close()
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
