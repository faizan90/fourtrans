'''
@author: Faizan-Uni-Stuttgart

Nov 12, 2020

7:52:52 PM

'''
import os
import time
import timeit
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import netCDF4 as nc
from scipy.stats import norm
import matplotlib.pyplot as plt

plt.ioff()

DEBUG_FLAG = False


def main():

    main_dir = Path(
        r'P:\Synchronize\IWS\Testings\fourtrans_practice\multisite_phs_spec_corr\temperature_kriging')

    os.chdir(main_dir)

    # order matters: data, mag, cos, sin
    parts = [
        'data/temperature_kriging_0_to_182_1km_data.nc',
        'mag/temperature_kriging_0_to_182_1km_mag.nc',
        'cos/temperature_kriging_0_to_182_1km_cos.nc',
        'sin/temperature_kriging_0_to_182_1km_sin.nc',
        ]

    var = 'OK'

    time_var = 'time'

    beg_time = '1991-01-01'
    end_time = '1991-12-30'
    freq = 'D'
    nc_time_units = 'days since 1900-01-01 00:00:00.0'
    nc_calendar = 'gregorian'

    var_type = 'temperature (C)'

    out_dir = Path('ifted')

    # NOTE: It is a copy of a reference file. Only the data values are updated.
    out_nc = out_dir / r'temperature_kriging_0_to_182_1km.nc'

    assert len(parts) == 4

    out_dir.mkdir(exist_ok=True)

    out_figs_dir = out_dir / 'interp_figs'

    out_figs_dir.mkdir(exist_ok=True)

    x_crds = y_crds = None

    mags = sins = coss = sorted_datas = None
    for i, part in enumerate(parts):
        nc_hdl = nc.Dataset(part, 'r')

        if x_crds is None:
            x_crds = nc_hdl['X'][...].data
            y_crds = nc_hdl['Y'][...].data

        if i == 0:
            sorted_datas = nc_hdl[var][...].data

        elif i == 1:
            mags = nc_hdl[var][...].data

        elif i == 2:
            coss = nc_hdl[var][...].data

        elif i == 3:
            sins = nc_hdl[var][...].data

        else:
            raise ValueError

        nc_hdl.close()

    coss[0, :, :] = 1.0
    sins[0, :, :] = 0.0

    phs_norm_mags = ((sins ** 2) + (coss ** 2)) ** 0.5
    coss /= phs_norm_mags
    sins /= phs_norm_mags

    phss = np.arctan2(sins, coss)

    ft_arr = np.full_like(phss, np.nan, dtype=complex)
    ft_arr.real[:] = mags * np.cos(phss)
    ft_arr.imag[:] = mags * np.sin(phss)

    norms_arr = np.fft.irfft(ft_arr, axis=0)

    probs_arr = norm.cdf(norms_arr)

    datas = np.full_like(probs_arr, np.nan, dtype=float)
    for i in range(datas.shape[1]):
        for j in range(datas.shape[2]):

            if np.isnan(probs_arr[0, i, j]):
                continue

            probs_ij = probs_arr[:, i, j]

            assert np.all(np.isfinite(probs_ij))

            sorted_datas_ij = np.sort(sorted_datas[:, i, j])
#             sorted_datas_ij = (sorted_datas[:, i, j])

            assert np.all(np.isfinite(sorted_datas_ij))

            data_ij = sorted_datas_ij[np.argsort(np.argsort(probs_ij))]

            datas[:, i, j] = data_ij

    nc_hdl = nc.Dataset(str(out_nc), mode='r+')

    dates_times = pd.date_range(beg_time, end_time, freq=freq)

    time_steps = nc.date2num(
        dates_times.to_pydatetime(),
        nc_time_units,
        nc_calendar)

    nc_hdl[time_var][:] = time_steps
    nc_hdl[time_var].units = nc_time_units
    nc_hdl[time_var].calendar = nc_calendar

    nc_hdl[var][:, :, :] = datas

    nc_hdl.sync()

    nc_hdl.close()

    x_crds_plt_msh, y_crds_plt_msh = np.meshgrid(x_crds, y_crds)

    for i in range(datas.shape[0]):

        interp_fld = datas[i]

        time_str = dates_times[i].strftime('%Y_%m_%d_T_%H_%M')
        print('Plotting:', time_str)

        out_fig_name = f'{var.lower()}_{time_str}.png'

        fig, ax = plt.subplots()

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            grd_min = np.nanmin(interp_fld)
            grd_max = np.nanmax(interp_fld)

        pclr = ax.pcolormesh(
            x_crds_plt_msh, y_crds_plt_msh,
            interp_fld,
            vmin=grd_min,
            vmax=grd_max)

        cb = fig.colorbar(pclr)

        cb.set_label(var_type)

        ax.set_xlabel('Easting')
        ax.set_ylabel('Northing')

#         title = f'Time: {i:04d}'

        title = (
            f'Time: {time_str}\n'
            f'Min.: {grd_min:0.4f}, Max.: {grd_max:0.4f}')

        ax.set_title(title)

        plt.setp(ax.get_xmajorticklabels(), rotation=70)
        ax.set_aspect('equal', 'datalim')

        plt.savefig(str(out_figs_dir / out_fig_name), bbox_inches='tight')
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
