'''
@author: Faizan-Uni-Stuttgart

Dec 2, 2020

12:56:26 PM

'''
import os
import time
import timeit
from pathlib import Path

import numpy as np
import pandas as pd
import netCDF4 as nc

DEBUG_FLAG = False


def main():

    main_dir = Path(
        r'P:\Synchronize\IWS\Testings\fourtrans_practice\multisite_phs_spec_corr\5min\v7')

    os.chdir(main_dir)

    interp_type = 'ifted'

    in_nc_file = Path(f'{interp_type}/{interp_type}.nc')

    data_var_lab = 'OK'
    time_var_lab = 'time'
    x_var_lab = 'X'
    y_var_lab = 'Y'

    # Numpy array ftns only.
    resample_type = 'sum'
    resample_res = '1D'

    zero_le_flag = True

    out_nc_file = Path(
        f'{interp_type}/{interp_type}__RR{resample_res}_RT{resample_type}.nc')

    with nc.Dataset(in_nc_file, 'r') as in_nc_hdl, nc.Dataset(out_nc_file , 'w') as out_nc_hdl:
        time_var = in_nc_hdl[time_var_lab]

        time_vals = time_var[:]
        nc_units = time_var.units
        nc_calendar = time_var.calendar

        time_idxs = pd.DatetimeIndex(
            nc.num2date(time_vals, nc_units, nc_calendar)).round('1s')

        # Last one is not taken, later.
        resample_time_idxs = pd.date_range(
            time_idxs[0], time_idxs[-1], freq=resample_res)

        data_var = in_nc_hdl[data_var_lab]

        assert data_var.ndim == 3

        out_shape = (
            resample_time_idxs.size - 1, data_var.shape[1], data_var.shape[2])

        out_data = np.full(out_shape, np.nan, dtype=float)

        for i in range(resample_time_idxs.size - 1):
            pt_i = resample_time_idxs[i]
            ct_i = resample_time_idxs[i + 1]

            pt_j = time_idxs.get_loc(pt_i) + 1
            ct_j = time_idxs.get_loc(ct_i) + 1

            assert pt_j < ct_j

            data_tvals = data_var[pt_j:ct_j, :, :].data

            if zero_le_flag:
                data_tvals[data_tvals < 0] = 0

            resample_data_vals = getattr(data_tvals, resample_type)(axis=0)

            out_data[i, :, :] = resample_data_vals

        resample_time_vals = nc.date2num(
            resample_time_idxs.to_pydatetime()[:-1], nc_units, nc_calendar)

        out_nc_hdl.set_auto_mask(False)
        out_nc_hdl.createDimension(x_var_lab, in_nc_hdl[x_var_lab].shape[0])
        out_nc_hdl.createDimension(y_var_lab, in_nc_hdl[y_var_lab].shape[0])
        out_nc_hdl.createDimension(time_var_lab, resample_time_vals.shape[0])

        x_coords_nc = out_nc_hdl.createVariable(
            x_var_lab, 'd', dimensions=x_var_lab)

        y_coords_nc = out_nc_hdl.createVariable(
            y_var_lab, 'd', dimensions=y_var_lab)

        time_nc = out_nc_hdl.createVariable(
            time_var_lab, 'i8', dimensions=time_var_lab)

        out_data_var = out_nc_hdl.createVariable(
            data_var_lab,
            'd',
            dimensions=(time_var_lab, y_var_lab, x_var_lab),
            fill_value=False)

        x_coords_nc[:] = in_nc_hdl[x_var_lab][:]
        y_coords_nc[:] = in_nc_hdl[y_var_lab][:]

        time_nc[:] = resample_time_vals
        time_nc.units = nc_units
        time_nc.calendar = nc_calendar

        out_data_var[:] = out_data

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
