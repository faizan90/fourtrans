'''
@author: Faizan-Uni-Stuttgart

Nov 12, 2020

10:24:11 AM

'''
import os
import time
import timeit
from pathlib import Path
from shutil import copy2

import h5py
import numpy as np
import pandas as pd

DEBUG_FLAG = True


def main():

    main_dir = Path(
        r'P:\Synchronize\IWS\Testings\fourtrans_practice\multisite_phs_spec_corr\5min\ppt_kg')

    os.chdir(main_dir)

    in_mag_file = Path(r'mag.h5')
    in_phs_file = Path(r'phs.h5')

    beg_time = '2009-01-01 00:00:00'
    end_time = '2009-12-31 23:59:00'
    freq = '5min'

    resample_steps = 288  # 1440//5

    out_file = Path(r'neckar_ppt_2009_5min.h5')

    le_zero_flag = False

    mag_ds_label = 'kg_1km_mag'
    phs_ds_label = 'kg_1km_phs'
    ift_ds_label = 'kg_1km'

    copy2(in_mag_file, out_file)

    dates_times = pd.date_range(beg_time, end_time, freq=freq)[::resample_steps]

    with h5py.File(in_mag_file, 'r') as mag_hdl, \
         h5py.File(in_phs_file, 'r') as phs_hdl, \
         h5py.File(out_file, 'a') as ift_hdl:

        time_strs = dates_times.strftime('%Y%m%dT%H%M%S')

        time_grp = ift_hdl.create_group('time')

        h5_str_dt = h5py.special_dtype(vlen=str)

        in_time_strs_ds = time_grp.create_dataset(
            'time_strs', (time_strs.shape[0],), dtype=h5_str_dt)

        in_time_strs_ds[:] = time_strs

        for interp_type in mag_hdl[mag_ds_label]:
            print('interp_type:', interp_type)
            for cat in mag_hdl[f'{mag_ds_label}/{interp_type}']:
                print('cat:', cat)
                mags = mag_hdl[f'{mag_ds_label}/{interp_type}/{cat}'][...]
                phss = phs_hdl[f'{phs_ds_label}/{interp_type}/{cat}'][...]

                assert np.all(mags.shape == phss.shape)

                ft_coeffs = np.full(mags.shape, np.nan, dtype=np.complex128)

                phss[+0, :] = 0
#                 phss[-1, :] = 0

                ft_coeffs.real[:] = np.cos(phss) * mags
                ft_coeffs.imag[:] = np.sin(phss) * mags

                ift = np.fft.irfft(ft_coeffs, axis=0)

                ift = ift.reshape(-1, resample_steps, ift.shape[1])

                ift = ift.sum(axis=1)

                assert ift.shape[0] == time_strs.shape[0]

                if le_zero_flag:
                    ift[ift < 0] = 0.0

#                 print((ift < 0).sum(), ift.size)

                ift_hdl[f'{ift_ds_label}/{interp_type}/{cat}'] = ift

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
