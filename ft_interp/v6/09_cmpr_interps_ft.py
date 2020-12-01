'''
@author: Faizan-Uni-Stuttgart

Nov 16, 2020

12:35:41 PM

'''
import os
import time
import timeit
from pathlib import Path

import matplotlib as mpl
mpl.rc('font', size=6)

mpl.rcParams['font.family'] = 'monospace'

import numpy as np
import pandas as pd
import netCDF4 as nc
import matplotlib.pyplot as plt

plt.ioff()

DEBUG_FLAG = False


def main():

    main_dir = Path(
        r'P:\Synchronize\IWS\Testings\fourtrans_practice\multisite_phs_spec_corr\5min')

    os.chdir(main_dir)

    # Two files allowed only.
    data_files = [
        Path(r'mags/mags.nc'),
        Path(r'phss/phss.nc'), ]

    data_dss = ['OK', 'OK']
    data_labs = ['mag', 'phs']

    stn_data_files = [Path(r'mags/mags.csv'), Path(r'phss/phss.csv')]
    stn_crds_file = Path(r'metadata_ppt_gkz3_crds.csv')

    vg_str_files = [
        Path(r'mags/vg_strs.csv'), Path(r'phss/vg_strs.csv')]

    sep = ';'
    x_crds_lab_csv, y_crds_lab_csv = 'X', 'Y'

    cmap = 'jet_r'
    fig_size = (15, 10)
    dpi = 150
    cbar_labs = ['Magnitude', 'Phase']

    freq_var = 'time'
    x_crds_lab = 'X'
    y_crds_lab = 'Y'

    out_dir = Path(r'cmpr_figs__interp_ft')

    assert len(data_files) == 2

    out_dir.mkdir(exist_ok=True)

    datas = None
    x_crds = y_crds = None
    freq_ds = None
    for data_file, data_ds in zip(data_files, data_dss):
        with nc.Dataset(data_file, 'r') as nc_hdl:

            if datas is None:
                datas = []

                datas.append(nc_hdl[data_ds][...].data)

            else:
                assert np.all(nc_hdl[data_ds].shape == datas[-1].shape)

                datas.append(nc_hdl[data_ds][...].data)

            if freq_ds is None:
                freq_ds = nc_hdl[freq_var]
                freqs = freq_ds[:]

            else:
                assert not np.setdiff1d(freqs, nc_hdl[freq_var][...]).size

            if x_crds is None:
                x_crds = nc_hdl[x_crds_lab][...].data
                y_crds = nc_hdl[y_crds_lab][...].data

            else:
                assert not np.setdiff1d(
                    x_crds, nc_hdl[x_crds_lab][...].data).size

                assert not np.setdiff1d(
                    y_crds, nc_hdl[y_crds_lab][...].data).size

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

    stn_data_dfs = [
        pd.read_csv(stn_data_file, sep=sep, index_col=0)
        for stn_data_file in stn_data_files]

    stn_crds_df = pd.read_csv(
        stn_crds_file, sep=sep, index_col=0)[[x_crds_lab_csv, y_crds_lab_csv]]

    vg_str_sers = [
        pd.read_csv(vg_str_file, sep=sep, index_col=0, squeeze=True)
        for vg_str_file in vg_str_files]

#     vg_labs = ['VG', 'VG']

    freq_strs = [str(freq) for freq in freqs]
    for i, freq_str in enumerate(freq_strs):

        print('Plotting:', freq_str)

        axes = plt.subplots(
            fig_rows,
            fig_cols,
            squeeze=False,
            figsize=fig_size,
            sharex=True,
            sharey=True)[1]

        # Original.
        row, col = 0, 0
        vmin = np.nanmin(datas[col][i, :, :])
        vmax = np.nanmax(datas[col][i, :, :])

        mappable = axes[row, col].pcolormesh(
            x_crds_mesh,
            y_crds_mesh,
            datas[col][i, :, :],
            vmin=vmin,
            vmax=vmax,
            cmap=cmap)

        stn_step_ser = stn_data_dfs[col].loc[freqs[i]].dropna()
        stn_step_x_crds = stn_crds_df.loc[stn_step_ser.index, x_crds_lab_csv]
        stn_step_y_crds = stn_crds_df.loc[stn_step_ser.index, y_crds_lab_csv]

        axes[row, col].set_title(
            f'{data_labs[col].upper():8s} | '
            f'Avg: {np.nanmean(datas[col][i, :, :]):0.2f}, '
            f'Var: {np.nanvar(datas[col][i, :, :]):0.2f}, '
            f'Min: {np.nanmin(datas[col][i, :, :]):0.2f}, '
            f'Max: {np.nanmax(datas[col][i, :, :]):0.2f}\n'
            f'OBSERVED | '
            f'Avg: {stn_step_ser.values.mean():0.2f}, '
            f'Var: {stn_step_ser.values.var():0.2f}, '
            f'Min: {stn_step_ser.values.min():0.2f}, '
            f'Max: {stn_step_ser.values.max():0.2f}\n'
            f'VG: {vg_str_sers[col].loc[i]}\n\n',
            loc='left')

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

        plt.colorbar(
            mappable=mappable,
            ax=axes[row, col],
            orientation='horizontal',
            label=cbar_labs[col],
            drawedges=False)

        axes[row, col].set_aspect('equal')
        axes[row, col].xaxis.set_tick_params(rotation=45)

        # IFTED
        row, col = 0, 1
        vmin = np.nanmin(datas[col][i, :, :])
        vmax = np.nanmax(datas[col][i, :, :])

        mappable = axes[row, col].pcolormesh(
            x_crds_mesh,
            y_crds_mesh,
            datas[col][i, :, :],
            vmin=vmin,
            vmax=vmax,
            cmap=cmap)

        stn_step_ser = stn_data_dfs[col].loc[freqs[i]].dropna()
        stn_step_x_crds = stn_crds_df.loc[stn_step_ser.index, x_crds_lab_csv]
        stn_step_y_crds = stn_crds_df.loc[stn_step_ser.index, y_crds_lab_csv]

        axes[row, col].set_title(
            f'{data_labs[col].upper():8s} | '
            f'Avg: {np.nanmean(datas[col][i, :, :]):0.2f}, '
            f'Var: {np.nanvar(datas[col][i, :, :]):0.2f}, '
            f'Min: {np.nanmin(datas[col][i, :, :]):0.2f}, '
            f'Max: {np.nanmax(datas[col][i, :, :]):0.2f}\n'
            f'OBSERVED | '
            f'Avg: {stn_step_ser.values.mean():0.2f}, '
            f'Var: {stn_step_ser.values.var():0.2f}, '
            f'Min: {stn_step_ser.values.min():0.2f}, '
            f'Max: {stn_step_ser.values.max():0.2f}\n'
            f'VG: {vg_str_sers[col].loc[i]}\n\n',
            loc='left')

        axes[row, col].grid()
        axes[row, col].set_axisbelow(True)

        axes[row, col].set_xlabel('Eastings (m)')

        axes[row, col].scatter(
            stn_step_x_crds.values,
            stn_step_y_crds.values,
            s=1,
            c='k',
            marker='o')

        plt.colorbar(
            mappable=mappable,
            ax=axes[row, col],
            orientation='horizontal',
            label=cbar_labs[col],
            drawedges=False)

        axes[row, col].set_aspect('equal')
        axes[row, col].xaxis.set_tick_params(rotation=45)

        # Save.
        plt.savefig(
            out_dir / f'interp_{freq_str}.png',
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
