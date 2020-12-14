'''
@author: Faizan-Uni-Stuttgart

Nov 17, 2020

7:57:25 PM

'''
import os
import time
import timeit
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

from spinterps import get_theo_vg_vals

plt.ioff()

DEBUG_FLAG = False


def get_mv_vals_crds(x_crds, mag_vals, phs_vals, ws):

    mag_mean_arr = np.zeros(mag_vals.shape[0] - ws)
    phs_mean_arr = mag_mean_arr.copy()
    ws_xcrds = mag_mean_arr.copy()
    for i in range(mag_vals.shape[0] - ws):
#         mean_arr[i] = vals[i:i + ws].mean()
#         ws_xcrds[i] = x_crds[i:i + ws].mean()

        mag_mean_arr[i] = np.median(mag_vals[i:i + ws])
        phs_mean_arr[i] = np.median(phs_vals[i:i + ws])
        ws_xcrds[i] = np.median(x_crds[i:i + ws])

    return (ws_xcrds, mag_mean_arr, phs_mean_arr)


def cmpt_cftn(
        data_file,
        crds_file,
        out_dir,
        ref_mag_vg_str,
        ref_phs_vg_str,
        beg_time,
        end_time):

    sep = ';'
    time_fmt = '%Y-%m-%d %H:%M:%S'

    fig_size = (15, 7)

#     cell_x_width = 1000
#     n_corr_intervals = 100

    if data_file.suffix == '.csv':
        data_df = pd.read_csv(data_file, sep=sep, index_col=0)
        data_df.index = pd.to_datetime(data_df.index, format=time_fmt)

    elif data_file.suffix == '.pkl':
        data_df = pd.read_pickle(data_file)

    else:
        raise NotImplementedError(
            f'Unknown extension of in_data_file: {data_file.suffix}!')

    data_df = data_df.loc[beg_time:end_time]

    crds_df = pd.read_csv(crds_file, sep=sep, index_col=0)
    crds_df.index = crds_df.index.astype(str)

#     data_df.dropna(how='any', axis=0, inplace=True)

    assert np.all(np.isfinite(data_df.values))
    assert np.all(np.isfinite(crds_df[['X', 'Y']].values))

    print(data_df.shape)
    print(crds_df.shape)

    assert all(data_df.shape)

    all_stns = data_df.columns

    if data_df.shape[0] % 2:
        data_df = data_df.iloc[:-1, :]
        print('Dropped last record in data_df!')

    n_stns = data_df.shape[1]

    probs_df = data_df.rank(axis=0) / (data_df.shape[0] + 1)

    norms_df = pd.DataFrame(
        data=norm.ppf(probs_df.values), columns=data_df.columns)

    ft_df = pd.DataFrame(
        data=np.fft.rfft(norms_df, axis=0),
        columns=data_df.columns)

    phs_spec_df = pd.DataFrame(
        data=np.angle(ft_df)[1:-1], columns=data_df.columns)

    n_freqs = phs_spec_df.shape[0]

    mag_spec_df = pd.DataFrame(
        data=np.abs(ft_df)[1:-1], columns=data_df.columns)

    dist_and_corrs_mat = np.full(
        (int(n_stns * (n_stns - 1) * 0.5), 3), np.nan)

    print(dist_and_corrs_mat.shape)

    print('Filling matrix...')

    idx = 0
    for i in range(n_stns):
        x_crd_i, y_crd_i = crds_df.loc[all_stns[i], ['X', 'Y']]
        for j in range(n_stns):
            if j <= i:
                continue

            x_crd_j, y_crd_j = crds_df.loc[
                all_stns[j], ['X', 'Y']]

            dist = (
                ((x_crd_i - x_crd_j) ** 2) +
                ((y_crd_i - y_crd_j) ** 2))

            dist **= 0.5

            phs_corr = np.cos(
                phs_spec_df.loc[:, all_stns[i]].values -
                phs_spec_df.loc[:, all_stns[j]].values).sum() / n_freqs

            mag_num = mag_spec_df.loc[
                :, [all_stns[i], all_stns[j]]].product(axis=1).values.sum()

            mag_denom = mag_spec_df.loc[
                :, [all_stns[i], all_stns[j]]].values ** 2

            mag_denom = mag_denom.sum(axis=0)

            mag_denom = np.product(mag_denom)

            mag_denom **= 0.5

            mag_corr = mag_num / mag_denom

            dist_and_corrs_mat[idx, 0] = dist
            dist_and_corrs_mat[idx, 1] = phs_corr
            dist_and_corrs_mat[idx, 2] = mag_corr

            idx += 1

    assert np.all(np.isfinite(dist_and_corrs_mat))
    print('Done filling!')

    dist_sort_idxs = np.argsort(dist_and_corrs_mat[:, 0])

    exp_vg_vals_x_mw, exp_vg_vals_mag_mw, exp_vg_vals_phs_mw = get_mv_vals_crds(
        dist_and_corrs_mat[dist_sort_idxs, 0],
        dist_and_corrs_mat[dist_sort_idxs, 2],
        dist_and_corrs_mat[dist_sort_idxs, 1],
        50)

#     max_corr = 1.0  # dist_and_corrs_mat[:, [1, 2]].max()
#     min_corr = 0.0  # dist_and_corrs_mat[:, [1, 2]].min()

#     max_dist = dist_and_corrs_mat[:, 0].max()
#     rem_dist = max_dist % cell_x_width
#     max_dist += (cell_x_width - rem_dist)
#
#     n_pcm_cols = int(np.ceil(max_dist / cell_x_width))
#     n_pcm_rows = int(n_corr_intervals)
#
#     cell_y_height = (max_corr - min_corr) / (n_pcm_rows - 1)
#
#     # Phase pcm.
#     pcm_mesh = np.zeros((n_pcm_rows, n_pcm_cols), dtype=float)
#     for i in range(dist_and_corrs_mat.shape[0]):
#         col = int(dist_and_corrs_mat[i, 0] // cell_x_width)
#
#         row = int((max_corr - dist_and_corrs_mat[i, 1]) // cell_y_height)
#
#         pcm_mesh[row, col] += 1
#
#     pcm_mesh /= dist_and_corrs_mat.shape[0]
#     pcm_mesh *= 100
#
#     plt.figure(figsize=fig_size)
#
#     pcm_x_crds, pcm_y_crds = np.meshgrid(
#         np.linspace(0, max_dist, n_pcm_cols + 1),
#         np.linspace(max_corr, min_corr, n_pcm_rows + 1))
#
#     plt.pcolormesh(pcm_x_crds, pcm_y_crds, pcm_mesh, cmap='jet')
#
#     plt.colorbar(label='Relative frequency (%)')
#
#     plt.xlabel('Distance (m)')
#     plt.ylabel('Phs. Spec. Corr. (-)')
#
#     plt.savefig(
#         str(out_dir / f'{data_file.stem}__phs_corr_pcm.png'), bbox_inches='tight')
#
# #     plt.show()
#     plt.close()
#
#     # Magnitude pcm
#     pcm_mesh = np.zeros((n_pcm_rows, n_pcm_cols), dtype=float)
#     for i in range(dist_and_corrs_mat.shape[0]):
#         col = int(dist_and_corrs_mat[i, 0] // cell_x_width)
#
#         row = int((max_corr - dist_and_corrs_mat[i, 2]) // cell_y_height)
#
#         pcm_mesh[row, col] += 1
#
#     pcm_mesh /= dist_and_corrs_mat.shape[0]
#     pcm_mesh *= 100
#
#     plt.figure(figsize=fig_size)
#
#     pcm_x_crds, pcm_y_crds = np.meshgrid(
#         np.linspace(0, max_dist, n_pcm_cols + 1),
#         np.linspace(max_corr, min_corr, n_pcm_rows + 1))
#
#     plt.pcolormesh(pcm_x_crds, pcm_y_crds, pcm_mesh, cmap='jet')
#
#     plt.colorbar(label='Relative frequency (%)')
#
#     plt.xlabel('Distance (m)')
#     plt.ylabel('Mag. Spec. Corr. (-)')
#
#     plt.savefig(
#         str(out_dir / f'{data_file.stem}__mag_corr_pcm.png'), bbox_inches='tight')
#
# #     plt.show()
#     plt.close()

    # Phase scatter.
    plt.figure(figsize=fig_size)

    plt.scatter(
        dist_and_corrs_mat[:, 0],
        dist_and_corrs_mat[:, 1],
        alpha=0.6,
        color='red',
        label='interp')

    plt.plot(
        exp_vg_vals_x_mw,
        exp_vg_vals_phs_mw,
        alpha=0.6,
        color='green',
        label='mw')

    plt.scatter(
        dist_and_corrs_mat[:, 0],
        1 - get_theo_vg_vals(ref_phs_vg_str, dist_and_corrs_mat[:, 0]),
        alpha=0.9,
        label='ref',
        color='blue')

    plt.grid()
    plt.legend()

    plt.xlabel('Distance (m)')
    plt.ylabel('Phs. Spec. Corr. (-)')

    plt.gca().set_axisbelow(True)

    plt.xlim(0, plt.xlim()[1])
#     plt.ylim(min_corr, max_corr)

    plt.savefig(
        str(out_dir / f'{data_file.stem}__phs_corr_sctr.png'),
        bbox_inches='tight')

#     plt.show()
    plt.close()

    # Magnitude scatter
    plt.figure(figsize=fig_size)

    plt.scatter(
        dist_and_corrs_mat[:, 0],
        dist_and_corrs_mat[:, 2],
        alpha=0.6,
        color='red',
        label='interp')

    plt.plot(
        exp_vg_vals_x_mw,
        exp_vg_vals_mag_mw,
        alpha=0.6,
        color='green',
        label='mw')

    plt.scatter(
        dist_and_corrs_mat[:, 0],
        1 - get_theo_vg_vals(ref_mag_vg_str, dist_and_corrs_mat[:, 0]),
        alpha=0.6,
        label='ref',
        color='blue')

    plt.grid()
    plt.legend()

    plt.xlabel('Distance (m)')
    plt.ylabel('Mag. Spec. Corr. (-)')

    plt.gca().set_axisbelow(True)

    plt.xlim(0, plt.xlim()[1])
#     plt.ylim(min_corr, max_corr)

    plt.savefig(
        str(out_dir / f'{data_file.stem}__mag_corr_sctr.png'),
        bbox_inches='tight')

#     plt.show()
    plt.close()

    return


def main():

    main_dir = Path(
        r'P:\Synchronize\IWS\Testings\fourtrans_practice\multisite_phs_spec_corr\5min\v7_long_range')

    os.chdir(main_dir)

    suff = '__RR1D_RTsum'

    in_data_files = [Path(f'orig/ts_OK{suff}.csv'), Path(f'ifted/ts_OK{suff}.csv')]
    in_crds_files = [Path(r'orig/ts_crds.csv'), Path(r'ifted/ts_crds.csv')]

    mag_cftn_file = Path(f'mags/ppt_median_cftns{suff}.csv')
    phs_cftn_file = Path(f'phss/ppt_median_cftns{suff}.csv')

    beg_time = '2009-01-01 00:00:00'
    end_time = '2009-03-31 23:59:00'

    out_dirs = [data_file.parents[0] for data_file in in_data_files]

    ref_mag_vg_str = pd.read_csv(mag_cftn_file, sep=';', index_col=0, squeeze=True).loc['mag']
    ref_phs_vg_str = pd.read_csv(phs_cftn_file, sep=';', index_col=0, squeeze=True).loc['phs']

    for data_file, crds_file, out_dir in zip(
        in_data_files, in_crds_files, out_dirs):

        print(
            'Going through:',
            data_file, crds_file, out_dir)

        cmpt_cftn(
            data_file,
            crds_file,
            out_dir,
            ref_mag_vg_str,
            ref_phs_vg_str,
            beg_time,
            end_time)

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
