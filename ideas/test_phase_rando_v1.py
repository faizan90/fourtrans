'''
@author: Faizan-Uni-Stuttgart

'''
import os
import timeit
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

# from numpy.linalg import cholesky


def get_corld_ser(in_data):

    n_rows, n_cols = in_data.shape

    assert n_rows
    assert n_cols == 2

    assert np.all(np.isfinite(in_data))

    x1 = np.random.normal(size=n_rows)
    x2 = np.random.normal(size=n_rows)

    pcorr = np.corrcoef(in_data[:, 0], in_data[:, 1])[0, 1]

    x3 = (pcorr * x1) + (((1 - pcorr ** 2) ** 0.5) * x2)

    mu1, mu2 = in_data.mean(axis=0)
    si1, si2 = in_data.var(axis=0)

    y1 = mu1 + (si1 * x1)
    y2 = mu2 + (si2 * x3)

    rand_phas = np.dstack((y1, y2))[0]

    return rand_phas


def get_corld_phas_ser(in_data_phas, n_data_steps, rfft_flag):

    rand_phas = get_corld_ser(in_data_phas[1:(n_data_steps // 2)])

    if rfft_flag:
        rand_phas = np.concatenate((
             np.atleast_2d(in_data_phas[0,:]),
             rand_phas,
             np.atleast_2d(in_data_phas[(n_data_steps // 2),:])))

    else:
        rand_phas = np.concatenate((
            np.atleast_2d(in_data_phas[0,:]),
            rand_phas,
            np.atleast_2d(in_data_phas[(n_data_steps // 2),:]),
            -rand_phas[::-1,:]))

    return rand_phas

# def obj_ftn(sigs, mus, in_data, data_corr, data_sigs_inv):
#
#     sim_rands = np.array([
#         np.matmul(sigs.T, np.matmul(data_sigs_inv, in_data[i]))
#         for i in range(in_data.shape[0])])
#
#     rando_data = in_data + sim_rands.reshape(-1, 1)
#
#     rando_corr = np.corrcoef(rando_data[:, 0], rando_data[:, 1])[0, 1]
#
#     diff = (rando_corr - data_corr) ** 2
#
#     if diff < 1e-5:
#         diff = 1e-5
#
#     print(sigs, diff)
#     return diff


def get_rand_phas_incs(in_data, norm_mean, correl):

    # norm_noise = np.random.normal(0.0, norm_mean, in_data.shape[0])
#     norm_noise = np.random.uniform(
#         low=-np.pi, high=np.pi, size=in_data.shape[0])

    norm_noise = np.random.exponential(norm_mean, size=in_data.shape[0])

    sim_rands = (in_data[:, 0] * correl) + norm_noise

    return sim_rands


def obj_ftn(prms, in_data, data_corr):

    sim_rands = get_rand_phas_incs(in_data, prms[0], prms[1])

    rando_data = in_data + sim_rands.reshape(-1, 1)

    rando_corr = np.corrcoef(rando_data[:, 0], rando_data[:, 1])[0, 1]

    diff = (rando_corr - data_corr) ** 2

    if diff < 1e-7:
        diff = 1e-7

#     print(prms, diff)

    return diff


def get_sim_phas_series(in_data_phas, n_data_steps):

    in_data = in_data_phas[1:(n_data_steps // 2),:]

#     mus = in_data.mean(axis=0)

    data_corr = np.corrcoef(in_data[:, 0], in_data[:, 1])[0, 1]

    bounds = ((0.0, 4.0), (-1.0, +1.0))

    opt = differential_evolution(
        obj_ftn,
        bounds=bounds,
        args=(in_data, data_corr),
    polish=False)

    print(opt.x)
    print(opt.fun)
    return opt.x


def main():

    main_dir = Path(r'P:\Synchronize\IWS\fourtrans_practice\test_phase_rando_v1')
    os.chdir(main_dir)

    in_file_path = (
        r'P:\Synchronize\IWS\Discharge_data_longer_series\neckar_norm_cop_infill_discharge_1961_2015_20190118\02_combined_station_outputs\infilled_var_df_infill_stns.csv')

    ref_stns = ['427', '454']

    beg_time = '1990-01-01'
    end_time = '2009-12-31'

    fig_size = (12, 9)

    in_data_df = pd.read_csv(in_file_path, sep=';', index_col=0)
    in_data_df.index = pd.to_datetime(in_data_df.index, format='%Y-%m-%d')

    data_df = in_data_df.loc[beg_time:end_time, ref_stns]

    if data_df.shape[0] % 2:
        data_df = data_df.iloc[:-1]

    n_data_steps = data_df.shape[0]

    assert not data_df.isna().values.sum()

    prob_df = data_df.rank() / (n_data_steps + 1.0)
    # prob_df = data_df.copy()

    data_vals_sp_corr = np.corrcoef(
        prob_df.values[:, 0], prob_df.values[:, 1])[0, 1]

    plt.figure(figsize=(fig_size))

    plt.scatter(
        prob_df.iloc[:, 0].values,
        prob_df.iloc[:, 1].values,
        alpha=0.3,
        color='b',
        marker='o')

    plt.xlabel(prob_df.columns[0])
    plt.ylabel(prob_df.columns[1])

    plt.grid()

    plt.title(f'Reference Empirical Copula (SCorr: {data_vals_sp_corr:0.3f})')

    plt.savefig('ref_ecop.png', bbox_inches='tight')
#     plt.show()
    plt.close('all')

    rfft_flag = True

    if rfft_flag:
        data_ft = np.fft.rfft(prob_df.values, axis=0)

    else:
        data_ft = np.fft.fft(prob_df.values, axis=0)

    data_mags = np.abs(data_ft)
    data_phas = np.angle(data_ft)

    data_phas_diff_cos_mat = np.empty(
        (n_data_steps // 2 + 1, n_data_steps // 2 + 1), dtype=float)

#     stn_idx = 0
    np_cos = np.cos
    for i in range(data_phas_diff_cos_mat.shape[0]):
        i_phas = data_phas[i, 0]
        for j in range(data_phas_diff_cos_mat.shape[1]):
            if i <= j:
                data_phas_diff_cos_mat[i, j] = np_cos(i_phas - data_phas[j, 1])
                data_phas_diff_cos_mat[j, i] = data_phas_diff_cos_mat[i, j]

    plt.figure(figsize=(fig_size))

    plt.imshow(
        data_phas_diff_cos_mat,
        alpha=0.8,
        cmap='seismic',
        vmin=-1.0,
        vmax=+1.0)

    plt.xlabel(f'{prob_df.columns[0]} - Phase Index')
    plt.ylabel(f'{prob_df.columns[1]} - Phase Index')

    plt.colorbar()

    plt.title(f'Reference phase difference cosine matrix')

    plt.savefig('ref_diff_cos.png', bbox_inches='tight')
#     plt.show()
    plt.close('all')
# #     raise Exception

#     mod_data_phas = data_phas + np.pi
#     mod_data_phas[mod_data_phas > np.pi] = (
#         (2 * np.pi) - mod_data_phas[mod_data_phas > np.pi])
#
#     mod_data_phas = mod_data_phas - (0.5 * np.pi)
#
#     mod_phas_corr = np.corrcoef(
#         mod_data_phas[:, 0], mod_data_phas[:, 1])[0, 1]
#
#     plt.figure(figsize=(fig_size))
#
#     plt.scatter(
#         mod_data_phas[:, 0],
#         mod_data_phas[:, 1],
#         alpha=0.4,
#         color='b',
#         marker='o')
#
#     plt.xlabel(prob_df.columns[0])
#     plt.ylabel(prob_df.columns[1])
#
#     plt.grid()
#
#     plt.title(
#         f'Reference Modified Phase Spectrum Scatter '
#         f'(PCorr: {mod_phas_corr:0.3f})')
#
#     plt.show()
#     plt.close('all')
#
#     plt.figure(figsize=(fig_size))
#
#     plt.plot(
#         np.sort(mod_data_phas[:, 0]),
#         alpha=0.4,
#         color='b',
#         marker='o',
#         label=prob_df.columns[0])
#
#     plt.plot(
#         np.sort(mod_data_phas[:, 1]),
#         alpha=0.4,
#         color='r',
#         marker='o',
#         label=prob_df.columns[1])
#
#     plt.grid()
#
#     plt.title(
#         f'Reference Modified Phase Spectrum Dsitribution '
#         f'(PCorr: {mod_phas_corr:0.3f})')
#
#     plt.show()
#     plt.close('all')

#     rand_phas_incs_var, rand_phas_incs_correl = get_sim_phas_series(
#         data_phas, n_data_steps)

#     raise Exception

    # corrrelations should be excluding the first and the nyquist frequency
    data_mags_corr = np.corrcoef(
        data_mags[1:(n_data_steps // 2), 0],
        data_mags[1:(n_data_steps // 2), 1])[0, 1]

    data_phas_corr = np.corrcoef(
        data_phas[1:(n_data_steps // 2), 0],
        data_phas[1:(n_data_steps // 2), 1])[0, 1]

    plt.figure(figsize=(fig_size))

    plt.semilogy(
        data_mags[:, 0],
        label=prob_df.columns[0],
        alpha=0.6,
        color='r')

    plt.semilogy(
        data_mags[:, 1],
        label=prob_df.columns[1],
        alpha=0.6,
        color='g')

    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')

    plt.title(
        f'Reference Fourier Magnitude Spectrum (PCorr: {data_mags_corr:0.3f})')

    plt.grid()
    plt.legend()

    plt.savefig('ref_four_mags.png', bbox_inches='tight')
#     plt.show()
    plt.close('all')

    plt.figure(figsize=(fig_size))

    plt.plot(data_phas[:, 0], label=prob_df.columns[0], alpha=0.6, color='r')
    plt.plot(data_phas[:, 1], label=prob_df.columns[1], alpha=0.6, color='g')

    plt.xlabel('Frequency')
    plt.ylabel('Phase')

    plt.title(
        f'Reference Fourier Phase Spectrum (PCorr: {data_phas_corr:0.3f})')

    plt.grid()
    plt.legend()

#     plt.show()
    plt.savefig('ref_four_phas.png', bbox_inches='tight')
    plt.close('all')

#     sim_phas = np.empty(data_mags.shape)
    sims_ft = np.empty(data_mags.shape, dtype=complex)
    sims = np.empty(prob_df.shape, dtype=complex)

#     rand_phas_main = -np.pi + ((2 * np.pi) * np.random.random((n_data_steps // 2) - 1))
    # rand_phas_main = get_rand_corr_phas(rand_phas_var, rand_phas_corr).ravel()

    # rand_phas[:] = 0.001

    # rand_phas = np.full(n_ft_steps, 0.0)
    # rand_phas[0] = 0.0
    # rand_phas[(n_data_steps // 2)] = 0.0

#     sim_phas = get_sim_phas_series(data_phas, n_data_steps, rfft_flag)
#     rand_phas_incs = get_rand_phas_incs(
#         data_phas[1:(n_data_steps // 2)],
#         rand_phas_incs_var,
#         rand_phas_incs_correl)

    rand_phas_incs = (-np.pi + (
        (2 * np.pi) * np.random.random((n_data_steps // 2) - 1)))

#     rand_phas_incs = np.random.normal(loc=2, scale=0.1, size=(n_data_steps // 2) - 1) + np.random.normal(loc=-2, scale=0.1, size=(n_data_steps // 2) - 1)
#     rand_phas_incs /= rand_phas_incs.max()
#     rand_phas_incs = -np.pi + ((2 * np.pi) * rand_phas_incs)

#     phas_diff_cos = np.cos(np.diff(data_phas[1:(n_data_steps // 2)], axis=1)).ravel()
#     rand_phas_incs_mask = (phas_diff_cos > 0.0)
#
# #     mag_diffs = np.cos(np.diff(data_mags[1:(n_data_steps // 2)], axis=1)).ravel()
# #     rand_phas_incs_mask = (mag_diffs < 0)
#
#     print('###### total rand_phas_incs:', rand_phas_incs_mask.shape[0])
#     print('###### rand_phas_incs_mask sum:', rand_phas_incs_mask.sum())
#
#     rand_phas_incs *= rand_phas_incs_mask

    if rfft_flag:
        sim_phas = np.concatenate((
             np.atleast_2d(data_phas[0,:]),
             data_phas[1:(n_data_steps // 2)] + rand_phas_incs.reshape(-1, 1),
             np.atleast_2d(data_phas[(n_data_steps // 2),:])))

    else:
        sim_phas = np.concatenate((
            np.atleast_2d(data_phas[0,:]),
            data_phas[1:(n_data_steps // 2)] + rand_phas_incs.reshape(-1, 1),
            np.atleast_2d(data_phas[(n_data_steps // 2),:]),
            -(data_phas[1:(n_data_steps // 2)] + rand_phas_incs.reshape(-1, 1),)[::-1]))

#     if rfft_flag:
#         sim_phas = np.concatenate((
#              np.atleast_2d(data_phas[0, :]),
#              data_phas[1:(n_data_steps // 2)],
#              np.atleast_2d(data_phas[(n_data_steps // 2), :]))) * 1
#
#     else:
#         sim_phas = np.concatenate((
#             np.atleast_2d(data_phas[0, :]),
#             data_phas[1:(n_data_steps // 2)],
#             np.atleast_2d(data_phas[(n_data_steps // 2), :]),
#             -(data_phas[1:(n_data_steps // 2)])[::-1])) * 1

#     if rfft_flag:
#         sim_phas = np.concatenate((
#             np.atleast_2d(data_phas[0, :]),
#             np.roll(data_phas[1:(n_data_steps // 2)], 1, axis=0),
#             np.atleast_2d(data_phas[(n_data_steps // 2), :])))
#
#     else:
#         sim_phas = np.concatenate((
#             np.atleast_2d(data_phas[0, :]),
#             np.roll(data_phas[1:(n_data_steps // 2)], 1, axis=0),
#             np.atleast_2d(data_phas[(n_data_steps // 2), :]),
#             -(np.roll(data_phas[1:(n_data_steps // 2)], 1, axis=0))[::-1]))

#
#     if not j:
#         rand_phas = data_phas[1:(n_data_steps // 2), j].mean() + (data_phas[1:(n_data_steps // 2), j] * data_phas[1:(n_data_steps // 2), j].var())
#
#     else:
#         rand_phas = get_rand_corr_phas(rand_phas_mean, rand_phas_var, data_phas_corr)
#         # rand_phas = (rand_phas % np.pi) * np.sign(rand_phas)

    for j in range(data_phas.shape[1]):
#         if not j:
#             rand_phas = data_phas[1:(n_data_steps // 2), j].mean() + (data_phas[1:(n_data_steps // 2), j] * data_phas[1:(n_data_steps // 2), j].var())
#
#         else:
#             rand_phas = get_rand_corr_phas(rand_phas_mean, rand_phas_var, data_phas_corr)
#             # rand_phas = (rand_phas % np.pi) * np.sign(rand_phas)
#
#         if rfft_flag:
#             rand_phas = np.concatenate(([data_phas[0, j]], rand_phas, [data_phas[(n_data_steps // 2), j]]))
#
#         else:
#             rand_phas = np.concatenate(([data_phas[0, j]], rand_phas, [data_phas[(n_data_steps // 2), j]], -rand_phas[::-1]))
#
#         sim_phas[:, j] = rand_phas

        sims_ft[:, j].real = data_mags[:, j] * np.cos(sim_phas[:, j])
        sims_ft[:, j].imag = data_mags[:, j] * np.sin(sim_phas[:, j])

        if rfft_flag:
            sims[:, j] = np.fft.irfft(sims_ft[:, j])

        else:
            sims[:, j] = np.fft.ifft(sims_ft[:, j])

    sim_phas_corr = np.corrcoef(sim_phas[:, 0], sim_phas[:, 1])[0, 1]

    prob_sim_df = pd.DataFrame(
        index=prob_df.index,
        data=sims,
        columns=prob_df.columns).rank() / (n_data_steps + 1.0)

#     prob_sim_df = pd.DataFrame(
#         index=prob_df.index,
#         data=sims,
#         columns=prob_df.columns)

    sim_vals_sp_corr = np.corrcoef(
        prob_sim_df.values[:, 0],
        prob_sim_df.values[:, 1]).real[0, 1]

    #==========================================================================
    # Simulated Empirical copula
    #==========================================================================

    plt.figure(figsize=(fig_size))

    plt.scatter(
        prob_sim_df.iloc[:, 0].values,
        prob_sim_df.iloc[:, 1].values,
        alpha=0.3,
        color='b',
        marker='o')

    plt.xlabel(prob_df.columns[0])
    plt.ylabel(prob_df.columns[1])

    plt.grid()

    plt.title(f'Simulated Empirical Copula (SCorr: {sim_vals_sp_corr:0.3f})')

    plt.savefig('sim_ecop.png', bbox_inches='tight')
#     plt.show()
    plt.close('all')

    #==========================================================================
    # Empirical copula compare
    #==========================================================================
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.scatter(
        prob_df.iloc[:, 0].values,
        prob_df.iloc[:, 1].values,
        alpha=0.1,
        color='b',
        marker='o')

    ax1.set_xlabel(prob_df.columns[0])
    ax1.set_ylabel(prob_df.columns[1])

    ax1.grid()

    ax1.set_title(
        f'Reference Empirical Copula (SCorr: {data_vals_sp_corr:0.3f})')

    ax2.scatter(
        prob_sim_df.iloc[:, 0].values,
        prob_sim_df.iloc[:, 1].values,
        alpha=0.1,
        color='b',
        marker='o')

    ax2.set_xlabel(prob_df.columns[0])
    ax2.set_ylabel(prob_df.columns[1])

    ax2.grid()

    ax2.set_title(
        f'Simulated Empirical Copula (SCorr: {sim_vals_sp_corr:0.3f})')

    plt.savefig('ref_sim_ecop_cmpr.png', bbox_inches='tight')
#     plt.show()
    plt.close('all')

    #=========================================================================
    # Phase scatter compare
    #=========================================================================
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.scatter(
        data_phas[:, 0],
        data_phas[:, 1],
        alpha=0.1,
        color='b',
        marker='o')

    ax1.set_xlabel(prob_df.columns[0])
    ax1.set_ylabel(prob_df.columns[1])

    ax1.grid()

    ax1.set_title('Reference Phase Scatter')

    ax2.scatter(
        sim_phas[:, 0],
        sim_phas[:, 1],
        alpha=0.1,
        color='b',
        marker='o')

    ax2.set_xlabel(prob_df.columns[0])
    ax2.set_ylabel(prob_df.columns[1])

    ax2.grid()

    ax2.set_title('Simulated Phase Scatter')

    ax1.set_xlim(
        min(data_phas.min(), sim_phas.min()),
        max(data_phas.max(), sim_phas.max()))

    ax1.set_ylim(*ax1.get_xlim())

    ax2.set_xlim(*ax1.get_xlim())
    ax2.set_ylim(*ax1.get_ylim())

    plt.savefig('ref_sim_phas_cmpr.png', bbox_inches='tight')
#     plt.show()
    plt.close('all')

    #=========================================================================
    # Folded Phase scatter compare
    #=========================================================================
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    folded_data_phas = np.abs(data_phas).copy()
    folded_data_phas = folded_data_phas % np.pi
    folded_data_phas[folded_data_phas > (0.5 * np.pi)] = (
        np.pi - folded_data_phas[folded_data_phas > (0.5 * np.pi)])

    folded_data_phas *= np.sign(data_phas)

    folded_data_phas_pcorr = np.corrcoef(
        folded_data_phas[:, 0], folded_data_phas[:, 1])[0, 1]

    folded_sim_phas = np.abs(sim_phas).copy()
    folded_sim_phas = folded_sim_phas % np.pi
    folded_sim_phas[folded_sim_phas > (0.5 * np.pi)] = (
        np.pi - folded_sim_phas[folded_sim_phas > (0.5 * np.pi)])

    folded_sim_phas *= np.sign(sim_phas)

    folded_sim_phas_pcorr = np.corrcoef(
        folded_sim_phas[:, 0], folded_sim_phas[:, 1])[0, 1]

    ax1.scatter(
        folded_data_phas[:, 0],
        folded_data_phas[:, 1],
        alpha=0.1,
        color='b',
        marker='o')

    ax1.set_xlabel(prob_df.columns[0])
    ax1.set_ylabel(prob_df.columns[1])

    ax1.grid()

    ax1.set_title(
        f'Reference Folded Phase Scatter (pcorr: '
        f'{folded_data_phas_pcorr:0.3f})')

    ax2.scatter(
        folded_sim_phas[:, 0],
        folded_sim_phas[:, 1],
        alpha=0.1,
        color='b',
        marker='o')

    ax2.set_xlabel(prob_df.columns[0])
    ax2.set_ylabel(prob_df.columns[1])

    ax2.grid()

    ax2.set_title(
        f'Simulated Folded Phase Scatter (pcorr: '
        f'{folded_sim_phas_pcorr:0.3f})')

    ax1.set_xlim(
        min(folded_data_phas.min(), folded_sim_phas.min()),
        max(folded_data_phas.max(), folded_sim_phas.max()))

    ax1.set_ylim(*ax1.get_xlim())

    ax2.set_xlim(*ax1.get_xlim())
    ax2.set_ylim(*ax1.get_ylim())

    plt.savefig('ref_sim_folded_phas_cmpr.png', bbox_inches='tight')
#     plt.show()
    plt.close('all')

    #==========================================================================
    # Simulated phase series
    #==========================================================================
    plt.figure(figsize=(fig_size))

    plt.plot(sim_phas[:, 0], label=prob_df.columns[0], alpha=0.6, color='r')
    plt.plot(sim_phas[:, 1], label=prob_df.columns[1], alpha=0.6, color='g')

    plt.xlabel('Frequency')
    plt.ylabel('Phase')

    plt.title(
        f'Simulated Fourier Phase Spectrum (PCorr: {sim_phas_corr:0.3f})')

    plt.grid()
    plt.legend()

#     plt.show()
    plt.savefig('sim_four_phas.png', bbox_inches='tight')
    plt.close('all')

    #==========================================================================
    # Lag SCorr
    #==========================================================================
    n_lag_corr_steps = 30

    lag_corrs = np.full((n_lag_corr_steps + 1, 2), 1.0)

    stn_idx = 1

    for i in range(1, n_lag_corr_steps + 1):
        lag_corrs[i, 0] = np.corrcoef(
            prob_df.iloc[:, stn_idx].values,
            np.roll(prob_df.iloc[:, stn_idx].values, i))[0, 1]

        lag_corrs[i, 1] = np.corrcoef(
            prob_sim_df.iloc[:, stn_idx].values,
            np.roll(prob_sim_df.iloc[:, stn_idx].values, i))[0, 1]

    lag_corrs_x_crds = np.arange(n_lag_corr_steps + 1)

    plt.figure(figsize=(fig_size))

    plt.plot(
        lag_corrs_x_crds,
        lag_corrs[:, 0],
        label='ref',
        alpha=0.6,
        color='r')

    plt.plot(
        lag_corrs_x_crds,
        lag_corrs[:, 1],
        label='sim',
        alpha=0.6,
        color='g')

    plt.xlabel('Lag (steps)')
    plt.ylabel('SCorr.')

    plt.title(f'Reference vs. Simulated Spearman autocorrelation')

    plt.grid()
    plt.legend()

    plt.savefig('ref_sim_scorr_cmpr.png', bbox_inches='tight')
#     plt.show()
    plt.close('all')

    #==========================================================================
    # Lag  PCorr
    #==========================================================================
    lag_corrs = np.full((n_lag_corr_steps + 1, 2), 1.0)

    sim_data = np.sort(data_df.iloc[:, stn_idx].values)[
        np.argsort(np.argsort(prob_sim_df.iloc[:, stn_idx].values))]

    for i in range(1, n_lag_corr_steps + 1):
        lag_corrs[i, 0] = np.corrcoef(
            data_df.iloc[:, stn_idx].values,
            np.roll(data_df.iloc[:, stn_idx].values, i))[0, 1]

        lag_corrs[i, 1] = np.corrcoef(
            sim_data,
            np.roll(sim_data, i))[0, 1]

    lag_corrs_x_crds = np.arange(n_lag_corr_steps + 1)

    plt.figure(figsize=(fig_size))

    plt.plot(
        lag_corrs_x_crds,
        lag_corrs[:, 0],
        label='ref',
        alpha=0.6,
        color='r')

    plt.plot(
        lag_corrs_x_crds,
        lag_corrs[:, 1],
        label='sim',
        alpha=0.6,
        color='g')

    plt.xlabel('Lag (steps)')
    plt.ylabel('PCorr.')

    plt.title(f'Reference vs. Simulated Pearson autocorrelation')

    plt.grid()
    plt.legend()

    plt.savefig('ref_sim_pcorr_cmpr.png', bbox_inches='tight')
#     plt.show()
    plt.close('all')

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
