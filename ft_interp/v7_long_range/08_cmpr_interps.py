'''
@author: Faizan-Uni-Stuttgart

Nov 16, 2020

12:35:41 PM

'''
import os
import time
import timeit
from pathlib import Path

import numpy as np
import pandas as pd
import netCDF4 as nc
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

plt.ioff()

DEBUG_FLAG = False


def main():

    main_dir = Path(
        r'P:\Synchronize\IWS\Testings\fourtrans_practice\multisite_phs_spec_corr\5min\v7_long_range')

    os.chdir(main_dir)

    # Two files allowed only.
    data_files = [
        Path(r'orig/orig__RR1D_RTsum.nc'),
        Path(r'ifted/ifted__RR1D_RTsum.nc'), ]

    data_dss = ['OK', 'OK']
    data_labs = ['orig', 'ifted']

    stn_data_file = Path(r'../neckar_1min_ppt_data_20km_buff_Y2009__RRD_RTsum.pkl')
    stn_crds_file = Path(r'../metadata_ppt_gkz3_crds.csv')

    sep = ';'
    time_fmt_csv = '%Y-%m-%d %H:%M:%S'
    x_crds_lab_csv, y_crds_lab_csv = 'X', 'Y'

    cmap = 'jet_r'
    fig_size = (17, 10)
    dpi = 150
    cbar_lab = 'Precipitation (mm)'

    time_var = 'time'
    x_crds_lab = 'X'
    y_crds_lab = 'Y'

    out_dir = Path(r'cmpr_figs__interp__RR1D_RTsum')

    assert len(data_files) == 2

    out_dir.mkdir(exist_ok=True)

    datas = None
    x_crds = y_crds = None
    time_step_ds = None
    for data_file, data_ds in zip(data_files, data_dss):
        with nc.Dataset(data_file, 'r') as nc_hdl:

            if datas is None:
                datas = []

                datas.append(nc_hdl[data_ds][...].data)

            else:
                assert np.all(nc_hdl[data_ds].shape == datas[-1].shape)

                datas.append(nc_hdl[data_ds][...].data)

            if time_step_ds is None:
                time_step_ds = nc_hdl[time_var]

                time_steps = pd.DatetimeIndex(nc.num2date(
                    time_step_ds[:],
                    time_step_ds.units,
                    time_step_ds.calendar))

            else:
                assert not time_steps.difference(
                    pd.DatetimeIndex(nc.num2date(
                    nc_hdl[time_var][:],
                    nc_hdl[time_var].units,
                    nc_hdl[time_var].calendar))).size

            if x_crds is None:
                x_crds = nc_hdl[x_crds_lab][...].data
                y_crds = nc_hdl[y_crds_lab][...].data

            else:
                assert not np.setdiff1d(
                    x_crds, nc_hdl[x_crds_lab][...].data).size

                assert not np.setdiff1d(
                    y_crds, nc_hdl[y_crds_lab][...].data).size

    time_steps = time_steps.round('1s')

    fig_rows, fig_cols = 1, 2

    assert (fig_rows * fig_cols) == len(datas)

    assert x_crds.ndim == y_crds.ndim == 1

    x_cell_size = x_crds[1] - x_crds[0]
    assert np.all(np.isclose(x_crds[1:] - x_crds[:-1], x_cell_size))

    y_cell_size = y_crds[1] - y_crds[0]
    assert np.all(np.isclose(y_crds[1:] - y_crds[:-1], y_cell_size))

    x_crds = np.concatenate(
        (x_crds - (x_cell_size * 0.5), [x_crds[-1] + x_cell_size]))

    y_crds = np.concatenate(
        (y_crds - (y_cell_size * 0.5), [y_crds[-1] + y_cell_size]))

    x_crds_mesh, y_crds_mesh = np.meshgrid(x_crds, y_crds)

    if stn_data_file.suffix == '.csv':
        stn_data_df = pd.read_csv(stn_data_file, sep=sep, index_col=0)

        stn_data_df.index = pd.to_datetime(
            stn_data_df.index, format=time_fmt_csv)

    elif stn_data_file.suffix == '.pkl':
        stn_data_df = pd.read_pickle(stn_data_file)

    else:
        raise NotImplementedError(
            f'Unknown extension of in_data_file: {stn_data_file.suffix}!')

    stn_crds_df = pd.read_csv(
        stn_crds_file, sep=sep, index_col=0)[[x_crds_lab_csv, y_crds_lab_csv]]

    time_step_strs = time_steps.strftime('%Y%m%dT%H%M%S')
    for i, time_step_str in enumerate(time_step_strs):

        print('Plotting:', time_step_str)

        # Observed.
        stn_step_ser = stn_data_df.loc[time_steps[i]].dropna()
        stn_step_x_crds = stn_crds_df.loc[stn_step_ser.index, x_crds_lab_csv]
        stn_step_y_crds = stn_crds_df.loc[stn_step_ser.index, y_crds_lab_csv]

        fig, axes = plt.subplots(
            fig_rows,
            fig_cols,
            squeeze=False,
            figsize=fig_size,
            sharex=True,
            sharey=True)

        vmin = min([np.nanmin(data[i, :, :]) for data in datas])
        vmax = max([np.nanmax(data[i, :, :]) for data in datas])

        # Original.
        row, col = 0, 0
        axes[row, col].pcolormesh(
            x_crds_mesh,
            y_crds_mesh,
            datas[col][i, :, :],
            vmin=vmin,
            vmax=vmax,
            cmap=cmap)

        axes[row, col].set_title(
            f'{data_labs[col].upper()}\n'
            f'Avg: {np.nanmean(datas[col][i, :, :]):0.2f}, '
            f'Var: {np.nanvar(datas[col][i, :, :]):0.2f}, '
            f'Min: {np.nanmin(datas[col][i, :, :]):0.2f}, '
            f'Max: {np.nanmax(datas[col][i, :, :]):0.2f}')

        axes[row, col].grid()
        axes[row, col].set_axisbelow(True)

        axes[row, col].set_xlabel('Eastings (m)')
        axes[row, col].set_ylabel('Northings (m)')

        axes[row, col].scatter(
            stn_step_x_crds.values,
            stn_step_y_crds.values,
            s=1,
            c='k',
            marker='o')

        # IFTED
        row, col = 0, 1
        axes[row, col].pcolormesh(
            x_crds_mesh,
            y_crds_mesh,
            datas[col][i, :, :],
            vmin=vmin,
            vmax=vmax,
            cmap=cmap)

        axes[row, col].set_title(
            f'{data_labs[col].upper()}\n'
            f'Avg: {np.nanmean(datas[col][i, :, :]):0.2f}, '
            f'Var: {np.nanvar(datas[col][i, :, :]):0.2f}, '
            f'Min: {np.nanmin(datas[col][i, :, :]):0.2f}, '
            f'Max: {np.nanmax(datas[col][i, :, :]):0.2f}')

        axes[row, col].grid()
        axes[row, col].set_axisbelow(True)

        axes[row, col].set_xlabel('Eastings (m)')

        axes[row, col].scatter(
            stn_step_x_crds.values,
            stn_step_y_crds.values,
            s=1,
            c='k',
            marker='o')

        plt.suptitle(
            f'OBSERVED\n'
            f'Avg: {stn_step_ser.values.mean():0.2f}, '
            f'Var: {stn_step_ser.values.var():0.2f}, '
            f'Min: {stn_step_ser.values.min():0.2f}, '
            f'Max: {stn_step_ser.values.max():0.2f}\n\n')

        # Colorbar
        cbaxes = fig.add_axes([0.2, 0.0, 0.65, 0.05])

        cmap_mappable_beta = plt.cm.ScalarMappable(
            norm=Normalize(vmin, vmax, clip=True),
            cmap=cmap)

        cmap_mappable_beta.set_array([])

        plt.colorbar(
            mappable=cmap_mappable_beta,
            cax=cbaxes,
            orientation='horizontal',
            label=cbar_lab,
            drawedges=False)

        # Save.
        plt.savefig(
            out_dir / f'interp_{time_step_str}.png',
            bbox_inches='tight',
            dpi=dpi)

        plt.close()

#         break

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
