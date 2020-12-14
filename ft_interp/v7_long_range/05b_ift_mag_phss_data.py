'''
@author: Faizan-Uni-Stuttgart

Nov 12, 2020

7:52:52 PM

'''
import os
import gc
import time
import timeit
import warnings
from shutil import copy2
from pathlib import Path

os.environ[str('MKL_NUM_THREADS')] = str(1)
os.environ[str('NUMEXPR_NUM_THREADS')] = str(1)
os.environ[str('OMP_NUM_THREADS')] = str(1)

import numpy as np
import pandas as pd
import netCDF4 as nc
# from scipy.stats import norm
import matplotlib.pyplot as plt

from cyth import fill_norm_probs_arr

# raise Exception

plt.ioff()

DEBUG_FLAG = False

import sys


def sizeof_fmt(num, suffix='B'):

    ''' by Fred Cirera,

    https://stackoverflow.com/a/1094933/1870254, modified'''

    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:

        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)

        num /= 1024.0

    return "%.1f %s%s" % (num, 'Yi', suffix)


def display_sorted_var_sizes(vars_dict):

    for name, size in sorted(
        ((name, sys.getsizeof(value))
         for name, value in vars_dict), key=lambda x:-x[1]):

        print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))

    return


def main():

    main_dir = Path(
        r'P:\Synchronize\IWS\Testings\fourtrans_practice\multisite_phs_spec_corr\5min\v7_long_range')

    os.chdir(main_dir)

    # order matters: data, mag, phs, data.
    # orig is copied as the new outputs with overwritten values.
    # orig itself gets its time and units updated.
    parts = [
        'data/data.nc',
        'mags/mags.nc',
        'phss/phss.nc',
        'data/data.nc',
        ]

    var = 'OK'

    # Output related.
    time_var = 'time'
    beg_time = '2009-01-01 00:00:00'
    end_time = '2009-03-31 23:59:00'
    freq = '5min'
    nc_time_units = 'minutes since 2009-01-01 00:00:00.0'
    nc_calendar = 'gregorian'

    var_type = 'Precipitation (mm)'

    out_dir = Path('ifted')

    plot_flag = False
    out_figs_dir = out_dir / 'interp_figs'

    # NOTE: It is a copy of a reference file. Only the data values are updated.
    out_nc = out_dir / r'ifted.nc'

    assert len(parts) == 4

    out_dir.mkdir(exist_ok=True)

    out_figs_dir.mkdir(exist_ok=True)

    x_crds = y_crds = None

    mags = phss = None  # sorted_datas =
    nnan_idxs = None
    for i, part in enumerate(parts):

        print(i, part)

        nc_hdl = nc.Dataset(part, 'r')

        if x_crds is None:
            x_crds = nc_hdl['X'][...].data
            y_crds = nc_hdl['Y'][...].data

        if i == 0:
            pass
#             sorted_datas = nc_hdl[var][...].data

        elif i == 1:
            nc_data = nc_hdl[var][...].data

            assert nc_data.flags.c_contiguous

            nnan_idxs = ~np.isnan(nc_data[:, :, :])

            mags = nc_data[nnan_idxs]

            mags = mags.reshape(-1, nnan_idxs[0, :, :].sum())

            del nc_data, nnan_idxs;

        elif i == 2:
            nc_data = nc_hdl[var][...].data

            assert nc_data.flags.c_contiguous

            nnan_idxs = ~np.isnan(nc_data[:, :, :])

            phss = nc_data[nnan_idxs]

            phss = phss.reshape(-1, nnan_idxs[0, :, :].sum())

            del nc_data, nnan_idxs;

        elif i == 3:
#             origs = nc_hdl[var][...].data

            pass

        else:
            raise ValueError

        gc.collect()

        nc_hdl.close()

        display_sorted_var_sizes(list(locals().items()))

        print('\n\n')

    phss[0, :] = 0.0

    print('FT filling...')
    ft_arr = np.full_like(phss, np.nan, dtype=complex)
    ft_arr[:].real = mags * np.cos(phss)
    ft_arr[:].imag = mags * np.sin(phss)

    del phss
    del mags
    gc.collect()

    print('IFFT...')
    norms_arr = np.fft.irfft(ft_arr, axis=0)

    del ft_arr
    gc.collect()

#     probs_arr = norm.cdf(norms_arr)
    probs_arr = np.empty_like(norms_arr)

    print('From norms to probs...')
    means_stds = np.empty((norms_arr.shape[1], 2), order='c')
    means_stds[:, 0] = norms_arr.mean(axis=0)
    means_stds[:, 1] = norms_arr.std(axis=0)

    fill_norm_probs_arr(norms_arr, means_stds, probs_arr)

    del norms_arr, means_stds
    gc.collect()

    print('Reading sorted data...')
    nc_hdl = nc.Dataset(parts[0], 'r')
    sorted_datas = nc_hdl[var][...].data
    assert sorted_datas.flags.c_contiguous

    # This one is needed later.
    nnan_idxs = ~np.isnan(sorted_datas[:, :, :])

    sorted_datas = sorted_datas[nnan_idxs]

    sorted_datas = sorted_datas.reshape(-1, nnan_idxs[0, :, :].sum())

    nc_hdl.close()

    print('Computing new series at interp locs...')
    datas = np.full_like(probs_arr, np.nan, dtype=float)
    for j in range(datas.shape[1]):

        if np.any(np.isnan(probs_arr[:, j])):
            continue

        probs_ij = probs_arr[:, j]

        assert np.all(np.isfinite(probs_ij))

        sorted_datas_ij = np.sort(sorted_datas[:, j])
#             sorted_datas_ij = (sorted_datas[:, j])

        assert np.all(np.isfinite(sorted_datas_ij))

        data_ij = sorted_datas_ij[np.argsort(np.argsort(probs_ij))]

        datas[:, j] = data_ij

    del probs_arr, sorted_datas

    # interp
    print('Writing interp data to NC...')
    copy2(parts[-1], out_nc)
    nc_hdl = nc.Dataset(str(out_nc), mode='r+')

    dates_times = pd.date_range(beg_time, end_time, freq=freq)

    time_steps = nc.date2num(
        dates_times.to_pydatetime(),
        nc_time_units,
        nc_calendar)

    nc_hdl[time_var][:] = time_steps
    nc_hdl[time_var].units = nc_time_units
    nc_hdl[time_var].calendar = nc_calendar

    out_data = np.full(nnan_idxs.shape, np.nan, dtype=float)
    out_data[nnan_idxs] = datas.ravel()

    nc_hdl[var][...] = out_data

    del datas

    nc_hdl.sync()

    nc_hdl.close()

    if plot_flag:
        x_crds_plt_msh, y_crds_plt_msh = np.meshgrid(x_crds, y_crds)

        for i in range(out_data.shape[0]):

            interp_fld = out_data[i]

            time_str = dates_times[i].strftime('%Y_%m_%d_T_%H_%M')
            print('Plotting:', time_str)

            out_fig_name = f'{var.lower()}_{time_str}.png'

            fig, ax = plt.subplots()

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')

                grd_min = np.nanmin(interp_fld)
                grd_max = np.nanmax(interp_fld)

            pclr = ax.pcolormesh(
                x_crds_plt_msh,
                y_crds_plt_msh,
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
