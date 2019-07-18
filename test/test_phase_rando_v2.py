'''
@author: Faizan-Uni-Stuttgart

'''
import os
import timeit
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

plt.ioff()


def plot_cum_mag_phas(
        data_phas_df,
        beg_data_phas_df,
        end_data_phas_df,
        mags_df,
        out_path):

    alpha = 0.8
    stns = data_phas_df.columns

    axs = plt.subplots(nrows=1, ncols=stns.shape[0], figsize=(12, 5))[1]

    labels = ('ref', 'beg', 'end')

    for i, stn in enumerate(stns):
        data_phas_mag_cumsum = (mags_df[stn].values ** 2).cumsum()

        mags_denom = data_phas_mag_cumsum[-1]

        beg_phas = beg_data_phas_df[stn].values
        end_phas = end_data_phas_df[stn].values

        beg_phas_mag_cumsum = ((mags_df[stn].values ** 2) * (np.cos(
            data_phas_df[stn].values - beg_phas))).cumsum()

        end_phas_mag_cumsum = ((mags_df[stn].values ** 2) * (np.cos(
            data_phas_df[stn].values - end_phas))).cumsum()

        data_phas_mag_cumsum /= mags_denom
        beg_phas_mag_cumsum /= mags_denom
        end_phas_mag_cumsum /= mags_denom

        axs[i].plot(data_phas_mag_cumsum, label=labels[0], alpha=alpha)
        axs[i].plot(beg_phas_mag_cumsum, label=labels[1], alpha=alpha)
        axs[i].plot(end_phas_mag_cumsum, label=labels[2], alpha=alpha)

        axs[i].set_xlabel('Frequency (-)')
        axs[i].set_ylabel('Cummulative contribution (-)')

        axs[i].legend()

        axs[i].grid()

        axs[i].set_title(f'Station: {stn}')

    plt.savefig(str(out_path), bbox_inches='tight')
    plt.close()
    return


def plot_lag_ecops(prob_df, beg_prob_df, end_prob_df, out_path_suff, lag):

    alpha = 0.01
    stns = prob_df.columns

    titles = ('Reference', 'Begin', 'End')

    for stn in stns:
        axs = plt.subplots(nrows=1, ncols=3, figsize=(17, 5))[1]

        axs[0].scatter(
            prob_df[stn].values,
            np.roll(prob_df[stn].values, lag),
            alpha=alpha)

        axs[1].scatter(
            beg_prob_df[stn].values,
            np.roll(beg_prob_df[stn].values, lag),
            alpha=alpha)

        axs[2].scatter(
            end_prob_df[stn].values,
            np.roll(end_prob_df[stn].values, lag),
            alpha=alpha)

        corrs = []

        corrs.append(np.corrcoef(
            prob_df[stn].values,
            np.roll(prob_df[stn].values, lag))[0, 1])

        corrs.append(np.corrcoef(
            beg_prob_df[stn].values,
            np.roll(beg_prob_df[stn].values, lag))[0, 1])

        corrs.append(np.corrcoef(
            end_prob_df[stn].values,
            np.roll(end_prob_df[stn].values, lag))[0, 1])

        for i, ax in enumerate(axs):
            ax.set_xlabel('Original (-)')
            ax.set_ylabel(f'Lagged {lag} steps (-)')

            ax.grid()

            ax.set_title(f'{titles[i]} (corr: {corrs[i]:0.3f})')

        plt.savefig(f'{out_path_suff}_{stn}.png', bbox_inches='tight')
        plt.close()
    return


def plot_phas_diffs(phas_df, beg_phas_df, end_phas_df, out_path):

    alpha = 0.7
    stns = phas_df.columns

    axs = plt.subplots(nrows=1, ncols=stns.shape[0], figsize=(17, 5))[1]

    for i, stn in enumerate(stns):
        axs[i].plot(
            beg_phas_df[stn].values - phas_df[stn].values,
            label='beg',
            alpha=alpha,
            lw=3)

        axs[i].plot(
            end_phas_df[stn].values - phas_df[stn].values,
            label='end',
            alpha=alpha,
            lw=1.5)

    for i, ax in enumerate(axs):
        ax.set_xlabel('Frequency index (-)')
        ax.set_ylabel('Phase difference from reference (-)')

        ax.grid()
        ax.legend()

        ax.set_title(f'Station: {stns[i]}')

    plt.savefig(str(out_path), bbox_inches='tight')
    plt.close()
    return


def plot_stn_sims(prob_df, beg_prob_df, end_prob_df, out_path_suff):

    stns = prob_df.columns

    alpha = 0.01

    axs_labs = (('ref', 'beg'), ('ref', 'end'), ('beg', 'end'))

    for stn in stns:
        axs = plt.subplots(nrows=1, ncols=3, figsize=(17, 5))[1]

        axs[0].scatter(
            prob_df[stn].values, beg_prob_df[stn].values, alpha=alpha)

        axs[1].scatter(
            prob_df[stn].values, end_prob_df[stn].values, alpha=alpha)

        axs[2].scatter(
            beg_prob_df[stn].values, end_prob_df[stn].values, alpha=alpha)

        corrs = []

        corrs.append(
            np.corrcoef(prob_df[stn].values, beg_prob_df[stn].values)[0, 1])

        corrs.append(
            np.corrcoef(prob_df[stn].values, end_prob_df[stn].values)[0, 1])

        corrs.append(np.corrcoef(
            beg_prob_df[stn].values, end_prob_df[stn].values)[0, 1])

        for j, ax in enumerate(axs):
            ax.set_xlabel(axs_labs[j][0])
            ax.set_ylabel(axs_labs[j][1])

            ax.grid()

            ax.set_title(f'SCorr: {corrs[j]:0.3f}')

        plt.suptitle(f'Station: {stn}')

        plt.savefig(f'{out_path_suff}_{stn}.png', bbox_inches='tight')
        plt.close()

    return


def get_phas_rand(in_ts, rand_phas=None):

    assert not (in_ts.shape[0] % 2)

    ft = np.fft.rfft(in_ts, axis=0)

    mags = np.abs(ft)
    phas = np.angle(ft)

    n_cols = in_ts.shape[1]

    if rand_phas is None:
        rand_phas = -np.pi + (
            (2 * np.pi) * np.random.random(size=phas.shape[0]))

    sim_ft = np.empty_like(ft)

    for i in range(n_cols):
        sim_ft[:, i].real = mags[:, i] * np.cos(phas[:, i] + rand_phas)
        sim_ft[:, i].imag = mags[:, i] * np.sin(phas[:, i] + rand_phas)

    rnd_ts = np.fft.irfft(sim_ft, axis=0)
    return rnd_ts


def plot_dists_ref_beg_end(data_df, beg_data_df, end_data_df, out_path):

    alpha = 0.9
    stns = data_df.columns

    _, axs = plt.subplots(nrows=1, ncols=data_df.shape[1], figsize=(12, 5))

    y_crds = np.arange(1.0, data_df.shape[0] + 1.0) / (data_df.shape[0] + 1.0)

    titles = stns

    types = ('ref', 'beg', 'end')

    for i, stn in enumerate(stns):
        axs[i].plot(
            np.sort(data_df[stn].values),
            y_crds,
            label=types[0],
            alpha=alpha)

        axs[i].plot(
            np.sort(beg_data_df[stn].values),
            y_crds,
            label=types[1],
            alpha=alpha)

        axs[i].plot(
            np.sort(end_data_df[stn].values),
            y_crds,
            label=types[2],
            alpha=alpha)

        axs[i].grid()
        axs[i].legend()

        axs[i].set_xlabel('Norm value (-)')
        axs[i].set_ylabel('Probability (-)')

        axs[i].set_title(f'{titles[i]}')

    plt.savefig(str(out_path), bbox_inches='tight')
    plt.close()
    return


def plot_ecops_ref_beg_end(prob_df, beg_prob_df, end_prob_df, out_path):

    alpha = 0.01
    stns = prob_df.columns

    _, axs = plt.subplots(nrows=1, ncols=3, figsize=(17, 5))

    (ax_data, ax_beg, ax_end) = axs

    ax_data.scatter(
        prob_df[stns[0]].values, prob_df[stns[1]].values, alpha=alpha)

    ax_beg.scatter(
        beg_prob_df[stns[0]].values, beg_prob_df[stns[1]].values, alpha=alpha)

    ax_end.scatter(
        end_prob_df[stns[0]].values, end_prob_df[stns[1]].values, alpha=alpha)

    corrs = []

    corrs.append(
        np.corrcoef(prob_df[stns[0]].values, prob_df[stns[1]].values)[0, 1])

    corrs.append(np.corrcoef(
        beg_prob_df[stns[0]].values, beg_prob_df[stns[1]].values)[0, 1])

    corrs.append(np.corrcoef(
        end_prob_df[stns[0]].values, end_prob_df[stns[1]].values)[0, 1])

    titles = ('Reference', 'Begin', 'End')

    for i, ax in enumerate(axs):
        ax.set_xlabel(stns[0])
        ax.set_ylabel(stns[1])

        ax.set_title(f'{titles[i]} (corr: {corrs[i]:0.3f})')

        ax.grid()

    plt.savefig(str(out_path), bbox_inches='tight')
    plt.close()
    return


def main():

    main_dir = Path(
        r'P:\Synchronize\IWS\fourtrans_practice\test_phase_rando_v2_non_norm')

    os.chdir(main_dir)

    in_file_path = (
        r'P:\Synchronize\IWS\Discharge_data_longer_series\neckar_norm_cop_'
        r'infill_discharge_1961_2015_20190118\02_combined_station_'
        r'outputs\infilled_var_df_infill_stns.csv')

    ref_stns = ['427', '454']
#     ref_stns = ['454']

    beg_time = '1961-01-01'
    end_time = '2015-12-31'

    out_pref = 'norm_pi'
    shift_phas = np.pi

#     out_pref = 'norm_half_pi'
#     shift_phas = 0.5 * np.pi

    thresh_rho = 0.9

    thresh_rho_fmt = '5.4e'

    lags = [5]

    plot_stn_ecops_flag = True
    plot_stn_dists_flag = True
    plot_ref_sim_ecop_cmpr_flag = True
    plot_phas_diffs_flag = True
    plot_lag_ecops_flag = True
    plot_cumm_var_cntrib_flag = True

#     plot_stn_ecops_flag = False
#     plot_stn_dists_flag = False
#     plot_ref_sim_ecop_cmpr_flag = False
#     plot_phas_diffs_flag = False
#     plot_lag_ecops_flag = False
#     plot_cumm_var_cntrib_flag = False

    in_data_df = pd.read_csv(in_file_path, sep=';', index_col=0)
    in_data_df.index = pd.to_datetime(in_data_df.index, format='%Y-%m-%d')

    data_df = in_data_df.loc[beg_time:end_time, ref_stns]

    if data_df.shape[0] % 2:
        data_df = data_df.iloc[:-1]

    n_data_steps = data_df.shape[0]

    assert not data_df.isna().values.sum()

    prob_df = data_df.rank() / (n_data_steps + 1.0)

    norm_df = pd.DataFrame(
        index=data_df.index, data=norm.ppf(prob_df), columns=data_df.columns)

#     norm_df = prob_df.copy()

    data_ft = np.fft.rfft(norm_df, axis=0)

    data_mags = np.abs(data_ft)
    data_phas = np.angle(data_ft)

    data_mags_sum = np.atleast_2d((data_mags ** 2).sum(axis=0))

    beg_cumsum_covs = (data_mags ** 2).cumsum(axis=0) / data_mags_sum
    end_cumsum_covs = 1 - beg_cumsum_covs

    beg_cumsum_thresh_idxs = np.argmin(
        (beg_cumsum_covs - thresh_rho) ** 2, axis=0)

    end_cumsum_thresh_idxs = np.argmin(
        (end_cumsum_covs - thresh_rho) ** 2, axis=0)

    print('Len data_mags:', data_mags.shape[0])
    print('beg_cumsum_thresh_idxs:', beg_cumsum_thresh_idxs)
    print('end_cumsum_thresh_idxs:', end_cumsum_thresh_idxs)

    beg_thresh_phas = data_phas.copy()
    end_thresh_phas = beg_thresh_phas.copy()

    for i, thresh_idx in enumerate(beg_cumsum_thresh_idxs):
        beg_thresh_phas[thresh_idx + 1:, i] = (
            beg_thresh_phas[thresh_idx + 1:, i] + (shift_phas))

        print(i, 'beg_cumsum_covs:', beg_cumsum_covs[thresh_idx, i])

    for i, thresh_idx in enumerate(end_cumsum_thresh_idxs):
        end_thresh_phas[:thresh_idx, i] = (
            end_thresh_phas[:thresh_idx, i] + (shift_phas))

        print(i, 'end_cumsum_covs:', end_cumsum_covs[thresh_idx, i])

    beg_data_ft = np.empty_like(data_ft)
    end_data_ft = beg_data_ft.copy()

    beg_data_ft.real = (
        data_mags * np.cos(beg_thresh_phas))

    beg_data_ft.imag = (
        data_mags * np.sin(beg_thresh_phas))

    beg_data_df = np.fft.irfft(beg_data_ft, axis=0)

    end_data_ft.real = (
        data_mags * np.cos(end_thresh_phas))

    end_data_ft.imag = (
        data_mags * np.sin(end_thresh_phas))

    end_data_df = np.fft.irfft(end_data_ft, axis=0)

    beg_data_df = pd.DataFrame(
        index=data_df.index, data=beg_data_df, columns=data_df.columns)

    end_data_df = pd.DataFrame(
        index=data_df.index, data=end_data_df, columns=data_df.columns)

    rand_phas = (-np.pi + (
        (2 * np.pi) * np.random.random(size=data_phas.shape[0]))) * 1.0

    rand_phas[0] = 0
    rand_phas[(n_data_steps // 2)] = 0

    sim_df = pd.DataFrame(
        index=data_df.index,
        data=get_phas_rand(norm_df.values, rand_phas),
        columns=data_df.columns)

    beg_sim_df = pd.DataFrame(
        index=data_df.index,
        data=get_phas_rand(beg_data_df.values, rand_phas),
        columns=data_df.columns)

    end_sim_df = pd.DataFrame(
        index=data_df.index,
        data=get_phas_rand(end_data_df.values, rand_phas),
        columns=data_df.columns)

    #==========================================================================
    # Plots
    #==========================================================================

    if plot_phas_diffs_flag:
        plot_phas_diffs(
            pd.DataFrame(data=data_phas, columns=data_df.columns),
            pd.DataFrame(data=beg_thresh_phas, columns=data_df.columns),
            pd.DataFrame(data=end_thresh_phas, columns=data_df.columns),
            f'{out_pref}_{thresh_rho:{thresh_rho_fmt}}_phas_diffs.png')

    if plot_stn_ecops_flag:
        plot_ecops_ref_beg_end(
            prob_df,
            beg_data_df.rank() / (n_data_steps + 1.0),
            end_data_df.rank() / (n_data_steps + 1.0),
            f'{out_pref}_{thresh_rho:{thresh_rho_fmt}}_'
            f'ecop_ref_beg_end_cmpr.png')

        plot_ecops_ref_beg_end(
            sim_df.rank() / (n_data_steps + 1.0),
            beg_sim_df.rank() / (n_data_steps + 1.0),
            end_sim_df.rank() / (n_data_steps + 1.0),
            f'{out_pref}_{thresh_rho:{thresh_rho_fmt}}_'
            f'ecop_sim_beg_end_cmpr.png')

    if plot_stn_dists_flag:
        plot_dists_ref_beg_end(
            norm_df,
            beg_data_df,
            end_data_df,
            f'{out_pref}_{thresh_rho:{thresh_rho_fmt}}_'
            f'dists_ref_beg_end_cmpr.png')

        plot_dists_ref_beg_end(
            sim_df,
            beg_sim_df,
            end_sim_df,
            f'{out_pref}_{thresh_rho:{thresh_rho_fmt}}_'
            f'dists_sim_beg_end_cmpr.png')

    if plot_ref_sim_ecop_cmpr_flag:
        plot_stn_sims(
            sim_df.rank() / (n_data_steps + 1.0),
            beg_sim_df.rank() / (n_data_steps + 1.0),
            end_sim_df.rank() / (n_data_steps + 1.0),
            f'{out_pref}_{thresh_rho:{thresh_rho_fmt}}_'
            f'ecop_cmpr_sim_indiv_stn.png')

        plot_stn_sims(
            prob_df,
            beg_data_df.rank() / (n_data_steps + 1.0),
            end_data_df.rank() / (n_data_steps + 1.0),
            f'{out_pref}_{thresh_rho:{thresh_rho_fmt}}_'
            f'ecop_cmpr_ref_indiv_stn.png')

    if plot_lag_ecops_flag:
        for lag in lags:
            plot_lag_ecops(
                prob_df,
                beg_data_df.rank() / (n_data_steps + 1.0),
                end_data_df.rank() / (n_data_steps + 1.0),
                f'{out_pref}_{thresh_rho:{thresh_rho_fmt}}_'
                f'lag_{lag}_ref_ecop_cmpr',
                lag)

            plot_lag_ecops(
                sim_df.rank() / (n_data_steps + 1.0),
                beg_sim_df.rank() / (n_data_steps + 1.0),
                end_sim_df.rank() / (n_data_steps + 1.0),
                f'{out_pref}_{thresh_rho:{thresh_rho_fmt}}_'
                f'lag_{lag}_sim_ecop_cmpr',
                lag)

    if plot_cumm_var_cntrib_flag:
        plot_cum_mag_phas(
            pd.DataFrame(data=data_phas, columns=data_df.columns),
            pd.DataFrame(data=beg_thresh_phas, columns=data_df.columns),
            pd.DataFrame(data=end_thresh_phas, columns=data_df.columns),
            pd.DataFrame(data=data_mags, columns=data_df.columns),
            f'{out_pref}_{thresh_rho:{thresh_rho_fmt}}_'
            f'cum_cntrib_ref.png')

        plot_cum_mag_phas(
            pd.DataFrame(data=data_phas, columns=data_df.columns),
            pd.DataFrame(
                data=beg_thresh_phas + rand_phas.reshape(-1, 1),
                columns=data_df.columns),
            pd.DataFrame(
                data=end_thresh_phas + rand_phas.reshape(-1, 1),
                columns=data_df.columns),
            pd.DataFrame(data=data_mags, columns=data_df.columns),
            f'{out_pref}_{thresh_rho:{thresh_rho_fmt}}_'
            f'cum_cntrib_sim.png')

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
