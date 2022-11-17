'''
@author: Faizan-Uni-Stuttgart

'''
import os
import timeit
import time
import pickle
from pathlib import Path
from math import factorial
from itertools import combinations
from multiprocessing import Pool

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from matplotlib.dates import YearLocator, DateFormatter
from pandas.plotting import register_matplotlib_converters

plt.ioff()
register_matplotlib_converters()


def ret_mp_idxs(n_vals, n_cpus):

    idxs = np.linspace(
        0, n_vals, min(n_vals + 1, n_cpus + 1), endpoint=True, dtype=np.int64)

    idxs = np.unique(idxs)

    assert idxs.shape[0]

    if idxs.shape[0] == 1:
        idxs = np.concatenate((np.array([0]), idxs))

    assert (idxs[0] == 0) & (idxs[-1] == n_vals), idxs
    return idxs


def cmpt_corr_sers(args):

    print(args)
    beg_j, end_j, in_file, min_corr_steps, comb_size, out_dir = args

    out_pkl = Path(out_dir) / f'ft_corr_series_{beg_j}_{end_j}.pkl'

    df = pd.read_pickle(in_file)

    print_pctge = 0.05

    n_cols = df.shape[1]
    n_steps = df.shape[0]

    assert n_cols >= comb_size

    combs = combinations(df.columns, comb_size)

    n_vals = end_j - beg_j

    assert n_vals > 0

    n_vals_ctr = 0
    corr_sers = {}
    for j, (stn_a, stn_b) in enumerate(combs, start=1):
        if not (beg_j <= j < end_j):
            continue

        n_vals_ctr += 1

        if not j % int(print_pctge * n_vals):
            print(
                f'Done at {print_pctge * n_vals * 100:0.1f} value '
                f'from indices {beg_j, end_j}!')

        ser_a = df.loc[:, stn_a]
        ser_b = df.loc[:, stn_b]

        ser_a_finite = np.isfinite(ser_a)
        ser_b_finite = np.isfinite(ser_b)

        not_ser_ab_finite = ~(ser_a_finite & ser_b_finite)

        ser_ab_cumsum = np.zeros(n_steps, dtype=int)

        break_idxs = np.where(not_ser_ab_finite)[0]

        take_idxs = []
        for i in range(1, break_idxs.shape[0]):
            beg_idx = break_idxs[i - 1] + 1
            end_idx = break_idxs[i]

            idx_diff = (end_idx - beg_idx)

            if idx_diff > 0:
                ser_ab_cumsum[beg_idx:end_idx] = idx_diff

                if idx_diff >= min_corr_steps:
                    take_idxs.append(
                        [df.index[beg_idx], df.index[end_idx - 1]])

            if idx_diff < 0:
                raise RuntimeError(
                    f'idx_diff cannot be less than zero! '
                    f'({idx_diff}, {beg_idx}, {end_idx})')

        if not take_idxs:
            print(f'Not enough values for {stn_a} and {stn_b}!')
            continue

        take_idxs = np.atleast_2d(np.array(take_idxs))

        ft_corrs = []
        for i in range(take_idxs.shape[0]):
            sub_arr = df.loc[
                take_idxs[i, 0]:take_idxs[i, 1], [stn_a, stn_b]].values

            assert np.all(np.isfinite(sub_arr))

            ft = np.fft.rfft(sub_arr, axis=0)[1:, :]

            ft_mags = np.abs(ft)
            ft_phas = np.angle(ft)

            ft_cov = (
                ft_mags[:, 0] *
                ft_mags[:, 1] *
                np.cos(ft_phas[:, 0] - ft_phas[:, 1]))

            ft_corr_denom = (
                (ft_mags[:, 0] ** 2).sum() *
                (ft_mags[:, 1] ** 2).sum()) ** 0.5

            ft_corr = np.cumsum(ft_cov) / ft_corr_denom

            ft_corrs.append(ft_corr)

        corr_sers[(stn_a, stn_b)] = [take_idxs, ft_corrs]

    with open(out_pkl, 'wb') as pkl_hdl:
        pickle.dump(corr_sers, pkl_hdl)

    return


def main():

    main_dir = Path(
        r'P:\Synchronize\IWS\Testings\fourtrans_practice\similar'
        r'_dissimilar_series')

    os.chdir(main_dir)

    in_file = r'combined_discharge_df.pkl'

#     out_pkl = r'ft_corr_series.pkl'
#
#     sep = ';'
#     time_fmt = '%Y-%m-%d'
#
#     corr_div_freq = 365
#     take_corr_fiv_freq_flag = False

    n_cpus = 7

    min_corr_steps = (5 * 365)

    #==========================================================================

    if min_corr_steps % 2:
        print('reduced min_corr_steps by 1!')
        min_corr_steps -= 1

    assert min_corr_steps >= 2

    df = pd.read_pickle(in_file)

    comb_size = 2

    n_cols = df.shape[1]
#     n_steps = df.shape[0]

    assert n_cols >= comb_size

#     combs = combinations(df.columns, comb_size)

    n_combs = int(
        factorial(n_cols) /
        (factorial(comb_size) * factorial(n_cols - comb_size)))

    print(f'n_combs: {n_combs}')

    mp_pool = Pool(n_cpus)

    mp_idxs = ret_mp_idxs(n_combs, n_cpus)

    args = (
        (mp_idxs[i],
         mp_idxs[i + 1],
         in_file,
         min_corr_steps,
         comb_size,
         os.getcwd())
        for i in range(mp_idxs.shape[0] - 1))

    mp_pool.map(cmpt_corr_sers, args)

#     stop_idx = 4000
# #     stop_idx = n_combs
#
#     corr_sers = {}
#     for j, (stn_a, stn_b) in enumerate(combs, start=1):
# #         print('\n')
# #         print(f'Going through combination {j}: {stn_a}, {stn_b}')
#         if not j % int(0.05 * n_combs):
#             print(f'Done till combination {j}: {stn_a}, {stn_b}')
#
#         ser_a = df.loc[:, stn_a]
#         ser_b = df.loc[:, stn_b]
#
#         ser_a_finite = np.isfinite(ser_a)
#         ser_b_finite = np.isfinite(ser_b)
#
#         not_ser_ab_finite = ~(ser_a_finite & ser_b_finite)
#
#         ser_ab_cumsum = np.zeros(n_steps, dtype=int)
#
#         break_idxs = np.where(not_ser_ab_finite)[0]
#
#         take_idxs = []
#         for i in range(1, break_idxs.shape[0]):
#             beg_idx = break_idxs[i - 1] + 1
#             end_idx = break_idxs[i]
#
#             idx_diff = (end_idx - beg_idx)
#
#             if idx_diff > 0:
#                 ser_ab_cumsum[beg_idx:end_idx] = idx_diff
#
# #                 take_str = ''
#                 if idx_diff >= min_corr_steps:
#                     take_idxs.append(
#                         [df.index[beg_idx], df.index[end_idx - 1]])
#
# #                     take_str = '### TAKEN ###'
# #
# #                 valid_sum = np.isfinite(
# #                     df.loc[:, [stn_a, stn_b]].iloc[beg_idx:end_idx]
# #                     ).all(axis=1).sum()
# #
# #                 print(
# #                     i,
# #                     idx_diff,
# #                     beg_idx,
# #                     end_idx,
# #                     valid_sum,
# #                     take_str)
#
#             if idx_diff < 0:
#                 raise RuntimeError(
#                     f'idx_diff cannot be less than zero! '
#                     f'({idx_diff}, {beg_idx}, {end_idx})')
#
#         if not take_idxs:
#             print(f'Not enough values for {stn_a} and {stn_b}!')
#             continue
#
#         take_idxs = np.atleast_2d(np.array(take_idxs))
# #         print(take_idxs.shape)
#
#         ft_corrs = []
#         for i in range(take_idxs.shape[0]):
#             sub_arr = df.loc[
#                 take_idxs[i, 0]:take_idxs[i, 1], [stn_a, stn_b]].values
#
#             assert np.all(np.isfinite(sub_arr))
#
#             ft = np.fft.rfft(sub_arr, axis=0)[1:, :]
#
#             ft_mags = np.abs(ft)
#             ft_phas = np.angle(ft)
#
#             ft_cov = (
#                 ft_mags[:, 0] *
#                 ft_mags[:, 1] *
#                 np.cos(ft_phas[:, 0] - ft_phas[:, 1]))
#
#             ft_corr_denom = (
#                 (ft_mags[:, 0] ** 2).sum() *
#                 (ft_mags[:, 1] ** 2).sum()) ** 0.5
#
#             ft_corr = np.cumsum(ft_cov) / ft_corr_denom
#
#             ft_corrs.append(ft_corr)
#
#         corr_sers[(stn_a, stn_b)] = [take_idxs, ft_corrs]
#
#         if j > stop_idx:
#             break
#
#     with open(out_pkl, 'wb') as pkl_hdl:
#         pickle.dump(corr_sers, pkl_hdl)
#
#     lowest_corr = 1.0
#     highest_corr = 0.0
#     lowest_corr_stns = None
#     highest_corr_stns = None
#
#     plt.figure(figsize=(15, 7))
#
#     for (stn_a, stn_b), values in corr_sers.items():
#
#         for i in range(values[0].shape[0]):
#             n_ft_steps = values[1][i].shape[0]
#             ft_corr = values[1][i]
#
#             div_idx = n_ft_steps // corr_div_freq
#
#             beg_corr = ft_corr[div_idx - 1]
#             end_corr = ft_corr[-1] - ft_corr[div_idx]
#
#             print(stn_a, stn_b, beg_corr, end_corr, n_ft_steps)
#
#             plt.plot(ft_corr, lw=0.5, alpha=0.7, color='k')
#
#     plt.grid()
#     plt.xlim(0, 300)
#     plt.show()
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
