'''
@author: Faizan-Uni-Stuttgart

Nov 4, 2020

7:07:08 PM

'''
import os
import time
import timeit
import warnings
from pathlib import Path

import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt

plt.ioff()

DEBUG_FLAG = False


def main():

    main_dir = Path(
        r'P:\Synchronize\IWS\Testings\fourtrans_practice\ft_spatio_temporal_interps\precipitation_interpolation_validation')

    os.chdir(main_dir)

    # order matters, first real then imag.
    parts = [
        'real/precipitation_kriging_0_to_730_1km_real.nc',
        'imag/precipitation_kriging_0_to_730_1km_imag.nc']

    var = 'OK'

    time_var = 'time'

    var_type = 'precipitation (mm)'

    out_dir = Path('combined_real_imag')

    # NOTE: It is a copy of a reference file. Only the data values are updated.
    out_nc = out_dir / r'precipitation_kriging_1989-01-01_to_1992-12-30_1km_ft.nc'

    assert len(parts) == 2

    out_dir.mkdir(exist_ok=True)

    out_figs_dir = out_dir / 'interp_figs'

    out_figs_dir.mkdir(exist_ok=True)

    ft_arr = x_crds = y_crds = None
    for i, part in enumerate(parts):
        nc_hdl = nc.Dataset(part, 'r')

        if ft_arr is None:
            ft_arr = np.full_like(nc_hdl[var], np.nan, dtype=complex)

            x_crds = nc_hdl['X'][...]
            y_crds = nc_hdl['Y'][...]

        if i == 0:
            ft_arr.real[...] = nc_hdl[var][...]

        elif i == 1:
            ft_arr.imag[...] = nc_hdl[var][...]

        else:
            raise ValueError

        nc_hdl.close()

    val_arr = np.fft.irfft(ft_arr, axis=0)

    nc_hdl = nc.Dataset(str(out_nc), mode='r+')

    time_steps = nc.num2date(
        nc_hdl[time_var][:].data,
        nc_hdl[time_var].units,
        nc_hdl[time_var].calendar)

    nc_hdl[var][:, :, :] = val_arr

    nc_hdl.sync()

    nc_hdl.close()

    x_crds_plt_msh, y_crds_plt_msh = np.meshgrid(x_crds, y_crds)

    for i in range(val_arr.shape[0]):

        interp_fld = val_arr[i]

        time_str = time_steps[i].strftime('%Y_%m_%d_T_%H_%M')

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
