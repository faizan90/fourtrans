'''
@author: Faizan-Uni-Stuttgart

Nov 12, 2020

10:24:11 AM

'''
import os
import time
import timeit
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

DEBUG_FLAG = True


def main():

    main_dir = Path(
        r'P:\Synchronize\IWS\Testings\fourtrans_practice\multisite_phs_spec_corr\temperature_kriging')

    os.chdir(main_dir)

    in_file = Path(r'neckar_6cats_temp_1991_1991.h5')

    beg_time = '1991-01-01'
    end_time = '1991-12-30'
    freq = 'D'

    le_zero_flag = False

    mag_ds_label = 'kriging_1km_mag'
    cos_ds_label = 'kriging_1km_cos'
    sin_ds_label = 'kriging_1km_sin'
    ift_ds_label = 'kriging_1km'

    dates_times = pd.date_range(beg_time, end_time, freq=freq)

    with h5py.File(in_file, 'a') as h5_hdl:

        for interp_type in h5_hdl[mag_ds_label]:
            for cat in h5_hdl[f'{mag_ds_label}/{interp_type}']:
                mags = h5_hdl[f'{mag_ds_label}/{interp_type}/{cat}'][...]
                coss = h5_hdl[f'{cos_ds_label}/{interp_type}/{cat}'][...]
                sins = h5_hdl[f'{sin_ds_label}/{interp_type}/{cat}'][...]

                ft_coeffs = np.full(mags.shape, np.nan, dtype=np.complex128)

                phss = np.arctan2(sins, coss)

                phss[+0, :] = 0
#                 phss[-1, :] = 0

                ft_coeffs.real[:] = np.cos(phss) * mags
                ft_coeffs.imag[:] = np.sin(phss) * mags

                ift = np.fft.irfft(ft_coeffs, axis=0)

                if le_zero_flag:
                    ift[ift < 0] = 0.0

#                 print((ift < 0).sum(), ift.size)

                h5_hdl[f'{ift_ds_label}/{interp_type}/{cat}'] = ift

        time_strs = dates_times.strftime('%Y%m%dT%H%M%S')

        time_grp = h5_hdl.create_group('time')

        h5_str_dt = h5py.special_dtype(vlen=str)

        in_time_strs_ds = time_grp.create_dataset(
            'time_strs', (time_strs.shape[0],), dtype=h5_str_dt)

        in_time_strs_ds[:] = time_strs

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
