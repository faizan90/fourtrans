'''
@author: Faizan-Uni-Stuttgart

'''
import os
import timeit
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import norm, rankdata
from scipy.interpolate import interp1d

plt.ioff()


def get_rank_auto_corrs(data, n_corr_lags):

    ranks = rankdata(data)

    # Normed reference vs. surrogate autocorrelation
    rank_auto_corrs = [1.0]

    for i in range(1, n_corr_lags):
        rank_auto_corrs.append(np.corrcoef(ranks, np.roll(ranks, i))[0, 1])

    rank_auto_corrs = np.array(rank_auto_corrs)
    return rank_auto_corrs


def get_exp_tfm_auto_corrs(data, n_corr_lags, lam):

    ranks = rankdata(data) / (data.shape[0] + 1.0)

    tfm_data = -np.log(1 - ranks) / lam

    # Normed reference vs. surrogate autocorrelation
    auto_corrs = [1.0]

    for i in range(1, n_corr_lags + 1):
        auto_corrs.append(
            np.corrcoef(tfm_data, np.roll(tfm_data, i))[0, 1])

    auto_corrs = np.array(auto_corrs)
    return auto_corrs


def get_norm_tfm_auto_corrs(data, n_corr_lags, mu, sig):

    probs = rankdata(data) / (data.shape[0] + 1.0)

    tfm_data = norm.ppf(probs, loc=mu, scale=sig)

    # Normed reference vs. surrogate autocorrelation
    auto_corrs = [1.0]

    for i in range(1, n_corr_lags + 1):
        auto_corrs.append(
            np.corrcoef(tfm_data, np.roll(tfm_data, i))[0, 1])

    auto_corrs = np.array(auto_corrs)
    return auto_corrs


def plot_rank_vs_valu_corrs(
        data,
        data_orig,
        n_corr_lags,
        fig_size,
        figs_dir,
        keep_phases_idx,
        n_vals,
        data_normed_angls,
        data_normed_ampts,
        data_normed):

    plt.figure(figsize=fig_size)
    ranks_auto_corrs_orig = get_rank_auto_corrs(data_orig, n_corr_lags)
    auto_corrs_orig = [1.0]

    for i in range(1, n_corr_lags):
        auto_corrs_orig.append(
            np.corrcoef(data_orig, np.roll(data_orig, i))[0, 1])

    auto_corrs_orig = np.array(auto_corrs_orig)

    sim_auto_corrs = []
    sim_rank_corrs = []

    for _i in range(10):
        if keep_phases_idx:
            n_rands = (n_vals // 2) - keep_phases_idx
            if n_rands > 0:
                rands = np.random.random(n_rands)

            else:
                rands = np.array([])

            rand_angls = np.concatenate((
                data_normed_angls[1:keep_phases_idx],
                -np.pi + (2 * np.pi * rands)))

        else:
            rands = np.random.random((n_vals // 2) - 1)
            rand_angls = -np.pi + (2 * np.pi * rands)

        ph_rand_angls = np.concatenate((
            [data_normed_angls[0]],
            rand_angls,
            [data_normed_angls[n_vals // 2]],
            rand_angls[::-1] * -1))

        assert ph_rand_angls.shape[0] == n_vals

        normed_ft = np.full(n_vals, np.nan, dtype=complex)
        normed_ft.real = data_normed_ampts * np.cos(ph_rand_angls)
        normed_ft.imag = data_normed_ampts * np.sin(ph_rand_angls)

        normed_ift = np.fft.ifft(normed_ft).real
        assert np.all(np.isfinite(normed_ift))

        assert np.isclose(normed_ift.mean(), data_normed.mean())
        assert np.isclose(normed_ift.std(), data_normed.std())

        normed_ift_ranks = rankdata(normed_ift)

        surr = np.sort(data)[(normed_ift_ranks - 1).astype(int)]
        assert np.all(np.isfinite(surr))

        auto_corrs = [1.0]
        rank_auto_corrs = [1.0]

        for i in range(1, n_corr_lags):
            auto_corrs.append(np.corrcoef(surr, np.roll(surr, i))[0, 1])

            rank_auto_corrs.append(
                np.corrcoef(
                    normed_ift_ranks,
                    np.roll(normed_ift_ranks, i))[0, 1])

        auto_corrs = np.array(auto_corrs)
        rank_auto_corrs = np.array(rank_auto_corrs)

        sim_auto_corrs.append(auto_corrs)
        sim_rank_corrs.append(rank_auto_corrs)

    sim_auto_corrs = np.array(sim_auto_corrs)
    sim_rank_corrs = np.array(sim_rank_corrs)

    auto_corrs_mins = sim_auto_corrs.min(axis=0)
    auto_corrs_maxs = sim_auto_corrs.max(axis=0)

    rank_auto_corrs_mins = sim_rank_corrs.min(axis=0)
    rank_auto_corrs_maxs = sim_rank_corrs.max(axis=0)

    plt.fill_between(
        np.arange(n_corr_lags),
        auto_corrs_mins,
        auto_corrs_maxs,
        alpha=0.2,
        color='r',
        label='sim_pcorr')

    plt.fill_between(
        np.arange(n_corr_lags),
        rank_auto_corrs_mins,
        rank_auto_corrs_maxs,
        alpha=0.2,
        color='b',
        label='sim_scorr')

    plt.plot(
        sim_auto_corrs.mean(axis=0),
        alpha=0.5,
        color='r',
        label='mean_sim_pcorr',
        lw=1.5)

    plt.plot(
        sim_rank_corrs.mean(axis=0),
        alpha=0.5,
        color='b',
        label='mean_sim_scorr',
        lw=1.5)

    plt.plot(
        auto_corrs_orig, alpha=0.8, color='r', label='orig_pcorr', lw=2)

    plt.plot(
        ranks_auto_corrs_orig, alpha=0.8, color='b', label='orig_scorr', lw=2)

    plt.xlabel('Lag')
    plt.ylabel('Correlation')

    plt.grid()

    plt.legend()

    plt.savefig(
        str(figs_dir / 'fft_tfm_auto_corrs.png'), bbox_inches='tight')

    plt.close()
    return


def target_to_probs(trgt_arr):

    assert isinstance(trgt_arr, np.ndarray)
    assert trgt_arr.ndim == 1

    ranks = rankdata(trgt_arr)

    probs = ranks / (trgt_arr.shape[0] + 1)

    return probs


def main():

    main_dir = Path(r'P:\Synchronize\IWS\fourtrans_practice')
    os.chdir(main_dir)

    in_file = Path(r'P:\Synchronize\IWS\QGIS_Neckar\hydmod\input_hyd_data\neckar_daily_discharge_1961_2015.csv')

    n_vals = 365 * 20
    n_corr_lags = 60

    figs_dir = 'test_surr_ranks_tfm'
    fig_size = (20, 9)

    data = pd.read_csv(in_file, sep=';', index_col=0)['454'].values[:n_vals]

    assert not np.isnan(data).sum()

    if data.shape[0] % 2:
        print(10 * '#', 'Removed last value to make n_vals even!', 10 * '#')

        data = data[:-1]
        n_vals -= 1

    assert n_vals >= 4
    assert n_vals > n_corr_lags

    data_probs = target_to_probs(data)

    data_normed = data_probs  # norm.ppf(data_probs)

    data_normed_ft = np.fft.fft(data_normed)

    data_normed_angls = np.angle(data_normed_ft)
    data_normed_ampts = np.absolute(data_normed_ft)

    keep_phases_idx = n_vals // 90

    assert keep_phases_idx <= (n_vals // 2)

    if keep_phases_idx:
        n_rands = (n_vals // 2) - keep_phases_idx
        if n_rands > 0:
            rands = np.random.random(n_rands)

        else:
            rands = np.array([])

        rand_angls = np.concatenate((
            data_normed_angls[1:keep_phases_idx],
            -np.pi + (2 * np.pi * rands)))

#         rand_angls = np.concatenate((
#             data_normed_angls[1:keep_phases_idx],
#             -np.pi + (2 * np.pi * rands)))
#
#         rands = data_normed_angls[keep_phases_idx:(n_vals // 2)].copy()
#         np.random.shuffle(rands)
#
#         rand_angls = np.concatenate((
#             data_normed_angls[1:keep_phases_idx],
#             rands))

    else:
        rands = np.random.random((n_vals // 2) - 1)
        rand_angls = -np.pi + (2 * np.pi * rands)

#         rand_angls = data_normed_angls[1:(n_vals // 2)].copy()
#         np.random.shuffle(rand_angls)

    ph_rand_angls = np.concatenate((
        [data_normed_angls[0]],
        rand_angls,
        [data_normed_angls[n_vals // 2]],
        rand_angls[::-1] * -1))

    assert ph_rand_angls.shape[0] == n_vals

    normed_ft = np.full(n_vals, np.nan, dtype=complex)
    normed_ft.real = data_normed_ampts * np.cos(ph_rand_angls)
    normed_ft.imag = data_normed_ampts * np.sin(ph_rand_angls)

    normed_ift = np.fft.ifft(normed_ft).real
    assert np.all(np.isfinite(normed_ift))

    assert np.isclose(normed_ift.mean(), data_normed.mean())
    assert np.isclose(normed_ift.std(), data_normed.std())

#     normed_ift_probs = norm.cdf(normed_ift)
#     assert not np.isnan(normed_ift_probs).sum()

#     probs_to_data_ftn = interp1d(
#         np.sort(data_probs),
#         np.sort(data),
#         bounds_error=False,
#         fill_value=(data.min(), data.max()))

#     surr = probs_to_data_ftn(normed_ift_probs)

    surr = np.sort(data)[(rankdata(normed_ift) - 1).astype(int)]
    assert np.all(np.isfinite(surr))

    figs_dir = Path(figs_dir)
    if not figs_dir.exists():
        figs_dir.mkdir(exist_ok=True)

    plot_rank_vs_valu_corrs(
            surr,
            data,
            n_corr_lags,
            fig_size,
            figs_dir,
            keep_phases_idx,
            n_vals,
            data_normed_angls,
            data_normed_ampts,
            data_normed)

#     plot_rank_vs_valu_corrs(surr, n_corr_lags, fig_size, figs_dir)

    mpl.rc('font', size=16)

    # Reference vs. surrogate
    plt.figure(figsize=fig_size)
    plt.plot(data, alpha=0.5, label='Ref.')
    plt.plot(surr, alpha=0.5, label='Surr.')

    plt.title(
        f'Reference vs. surrogate comparison\n'
        f'ref. min: {data.min():0.3f}, '
        f'ref. max: {data.max():0.3f}, '
        f'ref. mean: {data.mean():0.3f}, '
        f'ref. std: {data.std():0.3f}\n'
        f'surr. min: {surr.min():0.3f}, '
        f'surr. max: {surr.max():0.3f}, '
        f'surr. mean: {surr.mean():0.3f} '
        f'surr std: {surr.std():0.3f}'
        )

    plt.ylabel('Discharge (m$^3$/s)')
    plt.xlabel('Time Step (days)')
    plt.grid()
    plt.legend()

    plt.savefig(str(figs_dir / 'ref_surr_compare.png'), bbox_inches='tight')
    plt.close()

    # Reference vs. surrogate normed
    plt.figure(figsize=fig_size)
    plt.plot(data_normed, alpha=0.5, label='Ref.')
    plt.plot(normed_ift, alpha=0.5, label='Surr.')

    plt.title(
        f'Reference vs. surrogate normed comparison\n'
        f'ref. min: {data_normed.min():0.3f}, '
        f'ref. max: {data_normed.max():0.3f}, '
        f'ref. mean: {data_normed.mean():0.3f}, '
        f'ref. std: {data_normed.std():0.3f}\n'
        f'surr. min: {normed_ift.min():0.3f}, '
        f'surr. max: {normed_ift.max():0.3f}, '
        f'surr. mean: {normed_ift.mean():0.3f} '
        f'surr std: {normed_ift.std():0.3f}'
        )

    plt.grid()
    plt.legend()

    plt.savefig(str(figs_dir / 'ref_surr_normed_compare.png'), bbox_inches='tight')
    plt.close()

    # reference vs. randomized angles
    plt.figure(figsize=fig_size)
    plt.plot(data_normed_angls, alpha=0.5, label='Ref. angles')
    plt.plot(ph_rand_angls, alpha=0.5, label='Surr. angles')

    plt.title(
        f'Reference vs. surrogate fourier angles comparison\n'
        f'ref. min: {data_normed_angls.min():0.3f}, '
        f'ref. max: {data_normed_angls.max():0.3f}, '
        f'ref. mean: {data_normed_angls.mean():0.3f}\n'
        f'surr. min: {ph_rand_angls.min():0.3f}, '
        f'surr. max: {ph_rand_angls.max():0.3f}, '
        f'surr. mean: {ph_rand_angls.mean():0.3f}'
        )

    plt.grid()
    plt.legend()

    plt.savefig(str(figs_dir / 'ref_surr_angles.png'), bbox_inches='tight')
    plt.close()

    # Amplitudes
    plt.figure(figsize=fig_size)
    plt.plot(data_normed_ampts, alpha=0.5, label='Amplitudes')

    plt.title(
        f'Reference fourier amplitudes\n'
        f'ref. min: {data_normed_ampts.min():0.3f}, '
        f'ref. max: {data_normed_ampts.max():0.3f}, '
        f'ref. mean: {data_normed_ampts.mean():0.3f}'
        )

    plt.grid()
    plt.legend()

    plt.savefig(str(figs_dir / 'ref_surr_ampts.png'), bbox_inches='tight')
    plt.close()

    # Reference vs. surrogate autocorrelation
    data_auto_corrs = [1.0]
    surr_auto_corrs = [1.0]
    for i in range(1, n_corr_lags):
        data_auto_corrs.append(np.corrcoef(data, np.roll(data, i))[0, 1])
        surr_auto_corrs.append(np.corrcoef(surr, np.roll(surr, i))[0, 1])

    data_auto_corrs = np.array(data_auto_corrs)
    surr_auto_corrs = np.array(surr_auto_corrs)

    plt.figure(figsize=fig_size)
    plt.plot(data_auto_corrs, alpha=0.5, label='Ref.')
    plt.plot(surr_auto_corrs, alpha=0.5, label='Surr.')

    plt.title(
        f'Reference vs. surrogate auto correlation comparison\n'
        f'ref. min: {data_auto_corrs[1:].min():0.3f}, '
        f'ref. max: {data_auto_corrs[1:].max():0.3f}, '
        f'ref. mean: {data_auto_corrs[1:].mean():0.3f}\n'
        f'surr. min: {surr_auto_corrs[1:].min():0.3f}, '
        f'surr. max: {surr_auto_corrs[1:].max():0.3f}, '
        f'surr. mean: {surr_auto_corrs[1:].mean():0.3f}'
        )

    plt.grid()
    plt.legend()

    plt.savefig(str(figs_dir / 'ref_surr_auto_corr.png'), bbox_inches='tight')
    plt.close()

    # Normed reference vs. surrogate autocorrelation
    norm_auto_corrs = [1.0]
    norm_ift_auto_corrs = [1.0]
    for i in range(1, n_corr_lags):
        norm_auto_corrs.append(
            np.corrcoef(data_normed, np.roll(data_normed, i))[0, 1])

        norm_ift_auto_corrs.append(
            np.corrcoef(normed_ift, np.roll(normed_ift, i))[0, 1])

    norm_auto_corrs = np.array(norm_auto_corrs)
    norm_ift_auto_corrs = np.array(norm_ift_auto_corrs)

    plt.figure(figsize=fig_size)
    plt.plot(norm_auto_corrs, alpha=0.5, label='Ref. norm.')
    plt.plot(norm_ift_auto_corrs, alpha=0.5, label='Surr. norm.')

    plt.title(
        f'Normed reference vs. surrogate auto correlation comparison\n'
        f'ref. min: {norm_auto_corrs[1:].min():0.3f}, '
        f'ref. max: {norm_auto_corrs[1:].max():0.3f}, '
        f'ref. mean: {norm_auto_corrs[1:].mean():0.3f}\n'
        f'surr. min: {norm_ift_auto_corrs[1:].min():0.3f}, '
        f'surr. max: {norm_ift_auto_corrs[1:].max():0.3f}, '
        f'surr. mean: {norm_ift_auto_corrs[1:].mean():0.3f}')

    plt.grid()
    plt.legend()

    plt.savefig(
        str(figs_dir / 'ref_surr_auto_corr_norm.png'),
        bbox_inches='tight')
    plt.close()

    # angle cdfs
    probs = np.arange(1.0, n_vals // 2) / (n_vals + 1)
    data_plot_angls = np.sort(data_normed_angls[1:n_vals // 2])
    surr_plot_angls = np.sort(rand_angls)

    plt.figure(figsize=fig_size)
    plt.plot(data_plot_angls, probs, alpha=0.5, label='Ref.')
    plt.plot(surr_plot_angls, probs, alpha=0.5, label='Surr.')

    plt.title(
        f'Reference vs. surrogate angle CDFs\n'
        f'ref. min: {data_plot_angls.min():0.3f}, '
        f'ref. max: {data_plot_angls.max():0.3f}, '
        f'ref. mean: {data_plot_angls.mean():0.3f}\n'
        f'surr. min: {surr_plot_angls.min():0.3f}, '
        f'surr. max: {surr_plot_angls.max():0.3f}, '
        f'surr. mean: {surr_plot_angls.mean():0.3f}')

    plt.grid()
    plt.legend()

    plt.savefig(
        str(figs_dir / 'ref_surr_angle_cdfs.png'),
        bbox_inches='tight')
    plt.close()

    # Reference vs. surrogate angle autocorrelation
    data_angle_auto_corrs = [1.0]
    surr_angle_auto_corrs = [1.0]
    for i in range(1, n_corr_lags):
        data_angle_auto_corrs.append(
            np.corrcoef(
                data_normed_angls, np.roll(data_normed_angls, i))[0, 1])

        surr_angle_auto_corrs.append(
            np.corrcoef(ph_rand_angls, np.roll(ph_rand_angls, i))[0, 1])

    data_angle_auto_corrs = np.array(data_angle_auto_corrs)
    surr_angle_auto_corrs = np.array(surr_angle_auto_corrs)

    plt.figure(figsize=fig_size)
    plt.plot(data_angle_auto_corrs, alpha=0.5, label='Ref.')
    plt.plot(surr_angle_auto_corrs, alpha=0.5, label='Surr.')

    plt.title(
        f'Reference vs. surrogate phase auto correlation comparison\n'
        f'ref. min: {data_angle_auto_corrs[1:].min():0.3f}, '
        f'ref. max: {data_angle_auto_corrs[1:].max():0.3f}, '
        f'ref. mean: {data_angle_auto_corrs[1:].mean():0.3f}\n'
        f'surr. min: {surr_angle_auto_corrs[1:].min():0.3f}, '
        f'surr. max: {surr_angle_auto_corrs[1:].max():0.3f}, '
        f'surr. mean: {surr_angle_auto_corrs[1:].mean():0.3f}'
        )

    plt.grid()
    plt.legend()

    plt.savefig(
        str(figs_dir / 'ref_surr_angle_auto_corr.png'), bbox_inches='tight')
    plt.close()

    # keep_phase_freq
    plt.figure(figsize=fig_size)
    for i in range(min(10, keep_phases_idx + 1)):
        keep_phase_ft = np.zeros(n_vals, dtype=complex)
        keep_phase_ft.real[i] = (
            data_normed_ampts[i] *
            np.cos(data_normed_angls[i]))

        keep_phase_ft.imag[i] = (
            data_normed_ampts[i] *
            np.sin(data_normed_angls[i]))

        keep_phase_ift = np.fft.ifft(keep_phase_ft).real
        assert np.all(np.isfinite(keep_phase_ift))

        plt.plot(keep_phase_ift, alpha=0.4, label=i)

    plt.title(
        f'Keep phase ift\n'
#         f'ref. min: {keep_phase_ift.min():0.3f}, '
#         f'ref. max: {keep_phase_ift.max():0.3f}, '
#         f'ref. mean: {keep_phase_ift.mean():0.3f}'
        )

    plt.grid()
#     plt.legend()

    plt.savefig(str(figs_dir / 'keep_phase_wave.png'), bbox_inches='tight')
    plt.close()

    mpl.rcdefaults()
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
    START = timeit.default_timer()  # to get the runtime of the program

    main()

    STOP = timeit.default_timer()  # Ending time
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))

    if _save_log_:
        log_link.stop()
