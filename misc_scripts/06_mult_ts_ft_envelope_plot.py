'''
@author: Faizan-Uni-Stuttgart

10 Mar 2020

13:55:03

'''
import os
import time
import pickle
import timeit
from pathlib import Path
from multiprocessing import Pool
from itertools import combinations

import matplotlib as mpl

mpl.rc('font', size=14)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters

plt.ioff()
register_matplotlib_converters()

DPI = 300

DEBUG_FLAG = False


def ret_mp_idxs(n_vals, n_cpus):

    idxs = np.linspace(
        0, n_vals, min(n_vals + 1, n_cpus + 1), endpoint=True, dtype=np.int64)

    idxs = np.unique(idxs)

    assert idxs.shape[0]

    if idxs.shape[0] == 1:
        idxs = np.concatenate((np.array([0]), idxs))

    assert (idxs[0] == 0) & (idxs[-1] == n_vals), idxs
    return idxs


def get_input_data(in_files, min_abs_corr, min_freq_steps, max_freq_steps):

    assert in_files, 'No files!'

    corrs = []
    steps = []  # steps are ft freqs
    stns = []
    periods = []
    beg_end_times = []

    ft_corr_sers = {}
    for in_file in in_files:
        print('Reading file:', in_file)
        with open(in_file, 'rb') as pkl_hdl:
            ft_corr_ser = pickle.load(pkl_hdl)

        sel_ctr = 0
        for stn_comb, (idxs, ft_corrs) in ft_corr_ser.items():
            for i in range(len(idxs)):
                if min_freq_steps is not None:
                    end_idx = ft_corrs[i].size // min_freq_steps

                else:
                    end_idx = ft_corrs[i].size

                if max_freq_steps is not None:
                    beg_idx = ft_corrs[i].size // max_freq_steps

                else:
                    beg_idx = 0

#                 if beg_idx == end_idx == 0:
#                     continue

                ft_corr_band = ft_corrs[i][beg_idx:end_idx + 1]

                if beg_idx > 0:
                    ft_corr_band -= ft_corrs[i][beg_idx - 1]

                if abs(ft_corr_band[-1]) < min_abs_corr:
                    continue

#                 # keep
#                  ft_corr_val = ft_corr_band[-1]
#                  ft_corr_band = (ft_corr_band - 0) / (ft_corr_band[-1] - ft_corr_band[0])

                corrs.append(ft_corr_band[-1])
                steps.append(ft_corr_band.shape[0])
                stns.append(stn_comb)
                periods.append(i)
                beg_end_times.append(idxs[i])

                ft_corr_sers[(stn_comb, i)] = ft_corr_band

                sel_ctr += 1

        print(f'Accepted {sel_ctr} pairs!')

        ft_corr_ser = None

    print('\n')
    print(f'Read a total of {len(ft_corr_sers)} pairs!')

    corrs = np.array(corrs)
    steps = np.array(steps)
    periods = np.array(periods)

    print(f'Accepted a total of {corrs.size} pairs!')

    return ft_corr_sers, corrs, steps, stns, periods, beg_end_times

# def get_corr_idx(pidx, idxs, stns, periods, ft_corr_sers, method):
#
#     if method == 'abs_sum':
#         fin_val = 0.0
#         fin_idx = None
#         for idx in np.where(idxs)[0]:
#             diff = (
#                 ft_corr_sers[(stns[pidx], periods[pidx])] -
#                 ft_corr_sers[(stns[idx ], periods[idx ])])
#
#             curr_val = (
#                 np.abs(diff).sum() /
#                 ft_corr_sers[(stns[pidx], periods[pidx])].shape[0])
#
#             if curr_val > fin_val:
#                 fin_val = curr_val
#                 fin_idx = idx
#
#     elif method == 'max':
#         fin_val = 0.0
#         fin_idx = None
#         for idx in np.where(idxs)[0]:
#             diff = (
#                 ft_corr_sers[(stns[pidx], periods[pidx])] -
#                 ft_corr_sers[(stns[idx ], periods[idx ])])
#
#             curr_val = diff.max()
#
#             if curr_val > fin_val:
#                 fin_val = curr_val
#                 fin_idx = idx
#
#     elif method == 'min':
#         fin_val = 1.0
#         fin_idx = None
#         for idx in np.where(idxs)[0]:
#             diff = (
#                 ft_corr_sers[(stns[pidx], periods[pidx])] -
#                 ft_corr_sers[(stns[idx ], periods[idx ])])
#
#             curr_val = diff.min()
#
#             if curr_val < fin_val:
#                 fin_val = curr_val
#                 fin_idx = idx
#
#     else:
#         raise ValueError(f'Incorrect method: {method}!')
#
#     return fin_val, fin_idx

# def get_corrs_data(
#         ft_corr_sers,
#         corrs,
#         steps,
#         stns,
#         periods,
#         corrs_abs_diff,
#         steps_abs_diff_ratio,
#         corr_selection_method='abs_sum',
#         vbs=True):
#
#     if vbs:
#         print('Getting correlation data...\n')
#
#     # corr_selection_method: abs_sum, max, min
#
#     done_steps = np.zeros(steps.shape[0], dtype=int)
#
#     ctr = 0
#     ctr_lim = np.inf
#     corrs_data = []
#     for i, (n_step, corr) in enumerate(zip(steps, corrs)):
#
#         if done_steps[i]:
#             continue
#
#         steps_le_flags = steps >= (n_step - (steps_abs_diff_ratio * n_step))
#         steps_ge_flags = steps <= (n_step + (steps_abs_diff_ratio * n_step))
#
#         valid_n_steps = steps_le_flags & steps_ge_flags
#         valid_n_steps[i] = False
#
#         corrs_le_flags = corrs >= (corr - corrs_abs_diff)
#         corrs_ge_flags = corrs <= (corr + corrs_abs_diff)
#
#         valid_corrs = corrs_le_flags & corrs_ge_flags
#         valid_corrs[i] = False
#
#         common_idxs = (valid_n_steps & valid_corrs)
#
#         if common_idxs.sum():
#             fin_val, fin_idx = get_corr_idx(
#                 i,
#                 common_idxs,
#                 stns,
#                 periods,
#                 ft_corr_sers,
#                 corr_selection_method)
#
#             if fin_idx is None:
#                 if vbs:
#                     print('max_corr_diff_idx is None')
#                 continue
#
#             if vbs:
#                 print('stns[i], stns[fin_idx]:', stns[i], stns[fin_idx])
#                 print('valid_n_steps sum:', valid_n_steps.sum())
#                 print('valid_corrs sum:  ', valid_corrs.sum())
#                 print('common_idxs sum:  ', common_idxs.sum())
#                 print('periods[i]:       ', periods[i])
#                 print('fin_val:          ', np.round(fin_val, 3))
#
#             corrs_data.append([
#                 stns[i],
#                 stns[fin_idx],
#                 fin_val,
#                 periods[i],
#                 periods[fin_idx]])
#
#             done_steps[fin_idx] = 1
#             done_steps[i] = 1
#
#             ctr += 1
#
#             if vbs:
#                 print('\n')
#
#         if ctr >= ctr_lim:
#             break
#
#     if vbs:
#         print('Done getting correlation data!\n')
#
#     return corrs_data


def plot_corr_series(args):

    (ft_corr_sers,
     fig_name_suff,
     out_dir,
     fig_size,
     vbs,
     in_df_pkl_file,
     plot_corr_ecops_flag) = args

    out_dir = Path(out_dir)

    out_dir.mkdir(exist_ok=True)

    df = pd.read_pickle(in_df_pkl_file)

    # should be equal intervals
    ecop_ll_lims = (0.0, 0.05)
    ecop_uu_lims = (0.95, 1.0)

    alpha = 0.03

    if vbs:
        print('Plotting...\n')

    ll_rel_freqs = []
    uu_rel_freqs = []

    for ((stn_as, period_idx_a),
         (ser_rank, ft_corr_ser, beg_end_times)) in ft_corr_sers.items():

        n_stn_as = len(stn_as)

        if vbs:
            print(
                'Plotting:',
                stn_as,
                period_idx_a,
                )

        n_cops = (n_stn_as * (n_stn_as - 1)) // 2
        if plot_corr_ecops_flag:

            n_cols = n_cops
            n_rows = 4

            fig_shape = (n_rows, n_cols)

            plt.figure(figsize=fig_size)

            corrs_ax = plt.subplot2grid((fig_shape), (0, 0), 1, n_cols)

            # ft correlations
            corrs_ax.plot(
                ft_corr_ser,
                color='C0',
                alpha=0.7,
                label=''.join([f'{stn} & ' for stn in stn_as])[:-3])

    #         min_corr = min(
    #             ft_corr_sers[(stn_as, period_idx_a)][-1],
    #             ft_corr_sers[(stn_bs, period_idx_b)][-1])
    #
    #         max_corr = max(
    #             ft_corr_sers[(stn_as, period_idx_a)][-1],
    #             ft_corr_sers[(stn_bs, period_idx_b)][-1])
    #
    #         if (min_corr > 0) and (max_corr >= 0):
    #             min_y_lim = +0
    #             max_y_lim = +1
    #
    #         elif (min_corr < 0) and (max_corr >= 0):
    #             min_y_lim = -1
    #             max_y_lim = +1
    #
    #         elif (min_corr < 0) and (max_corr < 0):
    #             min_y_lim = -1
    #             max_y_lim = +0
    #
    #         else:
    #             raise ValueError(f'Unknown situation: {min_corr, max_corr}')

    #         corrs_ax.set_ylim(min_y_lim, max_y_lim)

            corrs_ax.set_aspect('auto', 'box', 'C')

            corrs_ax.set_xlabel('Frequency [-]')
            corrs_ax.set_ylabel('Correlation [-]')

            corrs_ax.legend(loc=4)
            corrs_ax.grid()

        # ecops
        dfs_a = df.loc[beg_end_times[0]: beg_end_times[1], stn_as]

        dfs_a_probs = dfs_a.rank(axis=0) / (dfs_a.shape[0] + 1.0)

        if plot_corr_ecops_flag:
            clrs = plt.get_cmap('jet')(
                np.arange(1.0, (dfs_a.shape[0] + 1.0)) /
                (dfs_a.shape[0] + 1.0))

        combs = combinations(stn_as, 2)
        for i, comb in enumerate(combs):
            sub_df = dfs_a_probs.loc[:, comb]

            ll_idxs = np.min(
                sub_df <= ecop_ll_lims[1], axis=1).values.astype(bool)

            uu_idxs = np.min(
                sub_df >= ecop_uu_lims[0], axis=1).values.astype(bool)

            ll_rel_freq = ll_idxs.sum() / ll_idxs.size
            uu_rel_freq = uu_idxs.sum() / uu_idxs.size

            ll_rel_freqs.append(ll_rel_freq)
            uu_rel_freqs.append(uu_rel_freq)

            if plot_corr_ecops_flag:

                ecop_ll_ax = plt.subplot2grid((fig_shape), (1, i), 1, 1)
                ecop_ax = plt.subplot2grid((fig_shape), (2, i), 1, 1)
                ecop_uu_ax = plt.subplot2grid((fig_shape), (3, i), 1, 1)

                ecop_ll_ax.scatter(
                    dfs_a_probs.loc[ll_idxs, comb[0]].values,
                    dfs_a_probs.loc[ll_idxs, comb[1]].values,
                    alpha=min(alpha / (ecop_ll_lims[1] - ecop_ll_lims[0]), 1),
                    marker='o',
                    c=clrs[ll_idxs])

                ecop_ax.scatter(
                    dfs_a_probs.loc[:, comb[0]].values,
                    dfs_a_probs.loc[:, comb[1]].values,
                    alpha=alpha,
                    marker='o')

                ecop_uu_ax.scatter(
                    dfs_a_probs.loc[uu_idxs, comb[0]].values,
                    dfs_a_probs.loc[uu_idxs, comb[1]].values,
                    alpha=min(alpha / (ecop_uu_lims[1] - ecop_uu_lims[0]), 1),
                    marker='o',
                    c=clrs[uu_idxs])

                ecop_uu_ax.set_xlabel(comb[0])
                ecop_ax.set_ylabel(comb[1])

                ecop_ll_ax.set_xlim(*ecop_ll_lims)
                ecop_ax.set_xlim(0, 1)
                ecop_uu_ax.set_xlim(*ecop_uu_lims)

                ecop_ll_ax.set_ylim(*ecop_ll_lims)
                ecop_ax.set_ylim(0, 1)
                ecop_uu_ax.set_ylim(*ecop_uu_lims)

                ecop_ll_ax.grid()
                ecop_ax.grid()
                ecop_uu_ax.grid()

                ecop_ll_ax.set_aspect('equal', 'box', 'SE')
                ecop_ax.set_aspect('equal', 'box', 'SE')
                ecop_uu_ax.set_aspect('equal', 'box', 'SE')

        # IO stuff
        all_stns_str = '_'.join(stn_as)

        if plot_corr_ecops_flag:
            fig_name = (
                f'{fig_name_suff}_{ser_rank}_{all_stns_str}_{period_idx_a}.png')

            plt.savefig(str(out_dir / fig_name), bbox_inches='tight', dpi=DPI)

            plt.close()

    if vbs:
        print('Done plotting!\n')

    return ll_rel_freqs, uu_rel_freqs


def plot_rel_freqs(ll_rel_freqs, uu_rel_freqs, out_fig_path):

    print('Plotting relative frequencies...')

    axes = plt.subplots(1, 2, squeeze=False, figsize=(15, 8))[1]

    ll_ax = axes[0, 0]
    uu_ax = axes[0, 1]

    ll_ax.hist(ll_rel_freqs, alpha=0.7, label='ll')

    ll_ax.set_xlabel('Relative frequency of ll')
    ll_ax.set_ylabel('Relative frequency')

    uu_ax.hist(uu_rel_freqs, alpha=0.7, label='uu')

    uu_ax.set_xlabel('Relative frequency of uu')

    plt.savefig(out_fig_path, bbox_inches='tight')

    plt.close()

    print('Done plotting relative frequencies.')
    return


def main():

    main_dir = Path(
        r'P:\Synchronize\IWS\Testings\fourtrans_practice\similar_dissimilar_'
        r'series_multuplets')

    os.chdir(main_dir)

    n_combs = 3

    data_dir = Path(f'combs_of_{n_combs}')

    in_files_patt = r'ft_corr_series_*_*.pkl'

    in_data_file = r'combined_discharge_df.pkl'

    steps_abs_diff_ratio = 0  # 0.005

    # differential correlation
    min_abs_corr = 0.001

    fig_size = (13, 15)

    # None for all
    max_n_files = None
    max_plot_figs = 500

    plot_corr_ecops_flag = False

    n_cpus = 1

    vbs_flag = False

    # band-pass selection
    min_freq_steps = 350  #  for the shortest frequency to keep
    max_freq_steps = None  #  for the longest frequency to keep

    out_dir = data_dir / (
        f'figs_rel_freqs__fmin_{min_freq_steps}__fmax_{max_freq_steps}')

    fig_name_suff = f'corrs'

    out_rel_freq_fig_path = out_dir / 'rel_freqs.png'

    data_pkl_name = (
        f'ft_corr_data'
        f'st{int(steps_abs_diff_ratio*1000):04d}.pkl')

    assert min_abs_corr >= 0

    assert (min_freq_steps is None) or (min_freq_steps > 0)
    assert (max_freq_steps is None) or (max_freq_steps > 0)

    if (min_freq_steps is not None) and (max_freq_steps is not None):
        assert max_freq_steps >= min_freq_steps

    assert sum([(min_freq_steps is not None),
                (max_freq_steps is not None)]) >= 1

    in_files = list(data_dir.glob(in_files_patt))

    print('\n\n')
    print('#' * 50)
    print('Found %d files!' % len(in_files))
    print('#' * 50)
    print('\n\n')

    ft_corr_sers, corrs, steps, stns, periods, beg_end_times = get_input_data(
        in_files[:max_n_files], min_abs_corr, min_freq_steps, max_freq_steps)

    assert len(corrs), 'Nothing selected!'

#     corrs_data = get_corrs_data(
#         ft_corr_sers,
#         corrs,
#         steps,
#         stns,
#         periods,
#         corrs_abs_diff,
#         steps_abs_diff_ratio,
#         corr_selection_method,
#         vbs_flag)
#
#     print('\n\n')
#     print('#' * 50)
#     print('corrs_data length:', len(corrs_data))
#     print('#' * 50)
#     print('\n\n')

    out_dir.mkdir(exist_ok=True)

#     with open(out_dir / data_pkl_name, 'wb') as pkl_hdl:
#         pickle.dump(corrs_data, pkl_hdl)

    with open(out_dir / data_pkl_name, 'wb') as pkl_hdl:
        pickle.dump((ft_corr_sers, corrs, steps, stns, periods), pkl_hdl)

#     fin_vals = np.array([corr_data[2] for corr_data in corrs_data])
    fin_vals = np.array(corrs)

    fin_val_sort_idxs = np.argsort(fin_vals)[::-1]

    mp_idxs = ret_mp_idxs(len(fin_val_sort_idxs[:max_plot_figs]), n_cpus)

    print('mp_idxs size:', mp_idxs.size)

    args = (({(stns[j], periods[j]):
              (i, ft_corr_sers[(stns[j], periods[j])], beg_end_times[j])
              for i, j in enumerate(fin_val_sort_idxs[mp_idxs[i]:mp_idxs[i + 1]])},
             fig_name_suff,
             out_dir,
             fig_size,
             vbs_flag,
             in_data_file,
             plot_corr_ecops_flag)

             for i in range(mp_idxs.size - 1))

    ll_rel_freqs = []
    uu_rel_freqs = []

    if n_cpus > 1:
        mp_pool = Pool(n_cpus)

        rets = mp_pool.map(plot_corr_series, args, chunksize=1)

        mp_pool.close()
        mp_pool.join()

    else:
        rets = list(map(plot_corr_series, args))

    for ret in rets:
        ll_rel_freqs.extend(ret[0])
        uu_rel_freqs.extend(ret[1])

    ll_rel_freqs = np.array(ll_rel_freqs)
    uu_rel_freqs = np.array(uu_rel_freqs)

    plot_rel_freqs(ll_rel_freqs, uu_rel_freqs, out_rel_freq_fig_path)

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
