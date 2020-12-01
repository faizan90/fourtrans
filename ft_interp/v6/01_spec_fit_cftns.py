'''
@author: Faizan-Uni-Stuttgart

Nov 11, 2020

11:52:34 AM

'''
import os
import time
import timeit
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

plt.ioff()

DEBUG_FLAG = False

mw_ftn = 'median'


def get_mv_vals_crds(x_crds, mag_vals, phs_vals, ws):

    mag_mean_arr = np.zeros(mag_vals.shape[0] - ws)
    phs_mean_arr = mag_mean_arr.copy()
    ws_xcrds = mag_mean_arr.copy()
    for i in range(mag_vals.shape[0] - ws):
#         mean_arr[i] = vals[i:i + ws].mean()
#         ws_xcrds[i] = x_crds[i:i + ws].mean()

        mag_mean_arr[i] = getattr(np, mw_ftn)(mag_vals[i:i + ws])
        phs_mean_arr[i] = getattr(np, mw_ftn)(phs_vals[i:i + ws])
        ws_xcrds[i] = getattr(np, mw_ftn)(x_crds[i:i + ws])

    return (ws_xcrds, mag_mean_arr, phs_mean_arr)


def nug_vg(h_arr, arg):
    # arg = (range, sill)
    nug_vg = np.full(h_arr.shape, arg[1])
    return nug_vg


def sph_vg(h_arr, arg):
    # arg = (range, sill)
    a = (1.5 * h_arr) / arg[0]
    b = h_arr ** 3 / (2 * arg[0] ** 3)
    sph_vg = (arg[1] * (a - b))
    sph_vg[h_arr > arg[0]] = arg[1]
    return sph_vg


def exp_vg(h_arr, arg):
    # arg = (range, sill)
    a = -3 * (h_arr / arg[0])
    exp_vg = (arg[1] * (1 - np.exp(a)))
    return exp_vg


def get_nug_sph_cftn(args, h_arr):

    # args = sph range, sph sill

    sph_rng = args[0]
    sph_sill = args[1]
    exp_rng = args[2]
    exp_sill = args[3]
    sph2_rng = args[4]
    sph2_sill = args[5]

    sph_vals = sph_vg(h_arr, [sph_rng, sph_sill])
    exp_vals = exp_vg(h_arr, [exp_rng, exp_sill])

    sph2_vals = sph_vg(h_arr, [sph2_rng, sph2_sill])

    cftn = 1 - (sph_vals + exp_vals + sph2_vals)

    return cftn


def obj_ftn(args, h_arr, c_arr):

    cftn = get_nug_sph_cftn(args, h_arr)
    sq_diff = (((cftn - c_arr) / (h_arr ** 1.0)) ** 2).sum()

    return sq_diff


def main():

    main_dir = Path(
        r'P:\Synchronize\IWS\Testings\fourtrans_practice\multisite_phs_spec_corr\5min')

    os.chdir(main_dir)

#     in_data_file = Path(r'temperature_avg.csv')
#     in_crds_file = Path(r'temperature_avg_coords.csv')  # has X, Y, Z cols
#     out_fig_pref = f'temperature_{mw_ftn}'

    in_data_file = Path(r'neckar_1min_ppt_data_20km_buff_Y2009__RR5min_RTsum.pkl')
    in_crds_file = Path(r'metadata_ppt_gkz3_crds.csv')  # has X, Y cols
    out_fig_pref = f'ppt_{mw_ftn}'

    sep = ';'
    time_fmt = '%Y-%m-%d %H:%M:%S'

    beg_time = '2009-01-01 00:00:00'
    end_time = '2009-03-31 23:59:00'

    fig_size = (15, 7)

    cut_off_dist = 5e5
    rng_bds = [1e0, 1e9]
    sill_bds = [0.0, 1.0]

    out_dir = main_dir

    phss_out_dir = out_dir / 'phss'
    phss_out_dir.mkdir(exist_ok=True, parents=True)

    mags_out_dir = out_dir / 'mags'
    mags_out_dir.mkdir(exist_ok=True, parents=True)

    if in_data_file.suffix == '.csv':
        data_df = pd.read_csv(in_data_file, sep=sep, index_col=0)
        data_df.index = pd.to_datetime(data_df.index, format=time_fmt)

    elif in_data_file.suffix == '.pkl':
        data_df = pd.read_pickle(in_data_file)

    else:
        raise NotImplementedError(
            f'Unknown extension of in_data_file: {in_data_file.suffix}!')

    crds_df = pd.read_csv(in_crds_file, sep=sep, index_col=0)

    data_df = data_df.loc[beg_time:end_time]

    data_df.dropna(axis=1, how='any', inplace=True)

    crds_df = crds_df.loc[data_df.columns]

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
        data=np.fft.rfft(norms_df.values, axis=0), columns=data_df.columns)

    phs_spec_df = pd.DataFrame(
        data=np.angle(ft_df), columns=data_df.columns)

#     phs_le_idxs = phs_spec_df < 0
#
#     phs_spec_df[phs_le_idxs] = (2 * np.pi) + phs_spec_df[phs_le_idxs]

    mag_spec_df = pd.DataFrame(
        data=np.abs(ft_df), columns=data_df.columns)

    n_freqs = phs_spec_df.shape[0]

    # Test to verify that forward and backward transforms are
    # working as expected.
#     fft_vals = np.empty_like(mag_spec_df.values, dtype=complex)
#
#     fft_vals[:].real = mag_spec_df.values * np.cos(phs_spec_df.values)
#     fft_vals[:].imag = mag_spec_df.values * np.sin(phs_spec_df.values)
#
#     ift_vals = np.fft.irfft(fft_vals, axis=0)

    phs_spec_df.to_csv(str(phss_out_dir / f'phss.csv'), sep=sep)

    mag_spec_df.to_csv(str(mags_out_dir / f'mags.csv'), sep=sep)

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

#     max_corr = dist_and_corrs_mat[:, [1, 2]].max()
#     min_corr = dist_and_corrs_mat[:, [1, 2]].min()

    print('Optimizing...')
    dist_sort_idxs = np.argsort(dist_and_corrs_mat[:, 0])

    (exp_vg_vals_x_mw,
     exp_vg_vals_mag_mw,
     exp_vg_vals_phs_mw) = get_mv_vals_crds(
        dist_and_corrs_mat[dist_sort_idxs, 0],
        dist_and_corrs_mat[dist_sort_idxs, 2],
        dist_and_corrs_mat[dist_sort_idxs, 1],
        50)

    opt_dist_idxs = exp_vg_vals_x_mw < cut_off_dist

    bds = [
        rng_bds,
        sill_bds,
        rng_bds,
        sill_bds,
        rng_bds,
        sill_bds]

    opt_res = differential_evolution(
        obj_ftn,
        bds,
        popsize=50,
        args=(exp_vg_vals_x_mw[opt_dist_idxs],
              exp_vg_vals_mag_mw[opt_dist_idxs]))

    opt_prms = opt_res.x
    sq_diff = opt_res.fun

    print(np.round(opt_prms, 3))
    print(sq_diff)

    print('Done optimizing.')

    mag_cftn_str = (
        f'{opt_prms[1]:0.5f} Sph({opt_prms[0]:0.1f}) + '
        f'{opt_prms[3]:0.5f} Exp({opt_prms[2]:0.1f}) + '
        f'{opt_prms[5]:0.5f} Sph({opt_prms[4]:0.1f})')

    with open(str(mags_out_dir / f'{out_fig_pref}_cftns.csv'), 'w') as txt_hdl:
        txt_hdl.write(f'ft_type;cftn\n')
        txt_hdl.write(f'mag;{mag_cftn_str}\n')

    with open(str(mags_out_dir / f'vg_strs.csv'), 'w') as txt_hdl:
        txt_hdl.write(f'freq;vg\n')

        for i in range(n_freqs):
            txt_hdl.write(f'{i};{mag_cftn_str}\n')

    exp_vg_vals = get_nug_sph_cftn(opt_prms, dist_and_corrs_mat[:, 0])

    # Magnitude scatter
    plt.figure(figsize=fig_size)

    plt.scatter(
        dist_and_corrs_mat[:, 0],
        dist_and_corrs_mat[:, 2],
        alpha=0.6,
        color='red',
        label='obs')

    plt.scatter(
        dist_and_corrs_mat[:, 0],
        exp_vg_vals,
        alpha=0.6,
        color='blue',
        label='fit')

    plt.plot(
        exp_vg_vals_x_mw,
        exp_vg_vals_mag_mw,
        alpha=0.6,
        color='green',
        label='mw')

    plt.title(mag_cftn_str)
    plt.grid()

    plt.legend()

    plt.xlabel('Distance (m)')
    plt.ylabel('Mag. Spec. Corr. (-)')

    plt.gca().set_axisbelow(True)

    plt.xlim(0, plt.xlim()[1])
#     plt.ylim(min_corr, max_corr)

    plt.savefig(
        str(mags_out_dir / f'{out_fig_pref}_mag_corr_cftn.png'),
        bbox_inches='tight')

#     plt.show()
    plt.close()

    # Phase spectrum.
    opt_res = differential_evolution(
        obj_ftn,
        bds,
        popsize=50,
        args=(exp_vg_vals_x_mw[opt_dist_idxs],
              exp_vg_vals_phs_mw[opt_dist_idxs]))

    opt_prms = opt_res.x
    sq_diff = opt_res.fun

    print(np.round(opt_prms, 3))
    print(sq_diff)

    print('Done optimizing.')

    phs_cftn_str = (
        f'{opt_prms[1]:0.5f} Sph({opt_prms[0]:0.1f}) + '
        f'{opt_prms[3]:0.5f} Exp({opt_prms[2]:0.1f}) + '
        f'{opt_prms[5]:0.5f} Sph({opt_prms[4]:0.1f})')

    with open(str(phss_out_dir / f'{out_fig_pref}_cftns.csv'), 'w') as txt_hdl:
        txt_hdl.write(f'ft_type;cftn\n')
        txt_hdl.write(f'phs;{phs_cftn_str}\n')

    with open(str(phss_out_dir / f'vg_strs.csv'), 'w') as txt_hdl:
        txt_hdl.write(f'freq;vg\n')

        for i in range(n_freqs):
            txt_hdl.write(f'{i};{phs_cftn_str}\n')

    exp_vg_vals = get_nug_sph_cftn(opt_prms, dist_and_corrs_mat[:, 0])

    # Phase scatter
    plt.figure(figsize=fig_size)

    plt.scatter(
        dist_and_corrs_mat[:, 0],
        dist_and_corrs_mat[:, 1],
        alpha=0.6,
        color='red',
        label='obs')

    plt.scatter(
        dist_and_corrs_mat[:, 0],
        exp_vg_vals,
        alpha=0.6,
        color='blue',
        label='fit')

    plt.plot(
        exp_vg_vals_x_mw,
        exp_vg_vals_phs_mw,
        alpha=0.6,
        color='green',
        label='mw')

    plt.title(phs_cftn_str)
    plt.grid()

    plt.legend()

    plt.xlabel('Distance (m)')
    plt.ylabel('Phs. Spec. Corr. (-)')

    plt.gca().set_axisbelow(True)

    plt.xlim(0, plt.xlim()[1])
#     plt.ylim(min_corr, max_corr)

    plt.savefig(
        str(phss_out_dir / f'{out_fig_pref}_phs_corr_cftn.png'),
        bbox_inches='tight')

#     plt.show()
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
