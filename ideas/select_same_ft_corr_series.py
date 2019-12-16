'''
@author: Faizan-Uni-Stuttgart

'''
import os
import timeit
import time
import pickle
from pathlib import Path

import matplotlib as mpl
mpl.rc('font', size=14)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters

plt.ioff()
register_matplotlib_converters()

DPI = 300


def get_input_data(in_files):

    ft_corr_sers = {}
    for in_file in in_files[:1]:
        with open(in_file, 'rb') as pkl_hdl:
            ft_corr_ser = pickle.load(pkl_hdl)

        ft_corr_sers.update(ft_corr_ser)
        ft_corr_ser = None

    print('Read %d pairs!' % len(ft_corr_sers))

    corrs = []
    steps = []
    stns = []
    periods = []
    for (stn_a, stn_b), value in ft_corr_sers.items():
        for i in range(value[0].shape[0]):
            corrs.append(value[1][i][-1])
            steps.append(value[1][i].shape[0])
            stns.append((stn_a, stn_b))
            periods.append(i)

    corrs = np.array(corrs)
    steps = np.array(steps)
    periods = np.array(periods)

    return ft_corr_sers, corrs, steps, stns, periods


def get_corr_idx(pidx, idxs, stns, periods, ft_corr_sers, method):

    if method == 'abs_sum':
        fin_val = 0.0
        fin_idx = None
        for idx in np.where(idxs)[0]:
            diff = (
                ft_corr_sers[stns[pidx]][1][periods[pidx]] -
                ft_corr_sers[stns[idx ]][1][periods[idx]])

            curr_val = (
                np.abs(diff).sum() /
                ft_corr_sers[stns[pidx]][1][periods[pidx]].shape[0])

            if curr_val > fin_val:
                fin_val = curr_val
                fin_idx = idx

    elif method == 'max':
        fin_val = 0.0
        fin_idx = None
        for idx in np.where(idxs)[0]:
            diff = (
                ft_corr_sers[stns[pidx]][1][periods[pidx]] -
                ft_corr_sers[stns[idx ]][1][periods[idx]])

            curr_val = diff.max()

            if curr_val > fin_val:
                fin_val = curr_val
                fin_idx = idx

    elif method == 'min':
        fin_val = 1.0
        fin_idx = None
        for idx in np.where(idxs)[0]:
            diff = (
                ft_corr_sers[stns[pidx]][1][periods[pidx]] -
                ft_corr_sers[stns[idx ]][1][periods[idx]])

            curr_val = diff.min()

            if curr_val < fin_val:
                fin_val = curr_val
                fin_idx = idx

    else:
        raise ValueError(f'Incorrect method: {method}!')

    return fin_val, fin_idx


def get_corrs_data(
        ft_corr_sers,
        corrs,
        steps,
        stns,
        periods,
        corrs_abs_diff,
        steps_abs_diff_ratio,
        corr_selection_method='abs_sum',
        vbs=True):

    if vbs:
        print('Getting correlation data...\n')

    # corr_selection_method: abs_sum, max, min

    done_steps = np.zeros(steps.shape[0], dtype=int)

    ctr = 0
    ctr_lim = np.inf
    corrs_data = []
    for i, (n_step, corr) in enumerate(zip(steps, corrs)):

        if done_steps[i]:
            continue

        steps_le_flags = steps >= (n_step - (steps_abs_diff_ratio * n_step))
        steps_ge_flags = steps <= (n_step + (steps_abs_diff_ratio * n_step))

        valid_n_steps = steps_le_flags & steps_ge_flags
        valid_n_steps[i] = False

        corrs_le_flags = corrs >= (corr - corrs_abs_diff)
        corrs_ge_flags = corrs <= (corr + corrs_abs_diff)

        valid_corrs = corrs_le_flags & corrs_ge_flags
        valid_corrs[i] = False

        common_idxs = (valid_n_steps & valid_corrs)

        if common_idxs.sum():
            fin_val, fin_idx = get_corr_idx(
                i,
                common_idxs,
                stns,
                periods,
                ft_corr_sers,
                corr_selection_method)

            if fin_idx is None:
                if vbs:
                    print('max_corr_diff_idx is None')
                continue

            if vbs:
                print('stns[i], stns[fin_idx]:', stns[i], stns[fin_idx])
                print('valid_n_steps sum:', valid_n_steps.sum())
                print('valid_corrs sum:  ', valid_corrs.sum())
                print('common_idxs sum:  ', common_idxs.sum())
                print('periods[i]:       ', periods[i])
                print('fin_val:          ', np.round(fin_val, 3))

            corrs_data.append([
                stns[i],
                stns[fin_idx],
                fin_val,
                periods[i],
                periods[fin_idx]])

            done_steps[fin_idx] = 1
            done_steps[i] = 1

            ctr += 1

            if vbs:
                print('\n')

        if ctr >= ctr_lim:
            break

    if vbs:
        print('Done getting correlation data!\n')

    return corrs_data


def plot_corr_series(
        plot_idxs,
        plot_data,
        ft_corr_sers,
        data_df,
        fig_name_suff,
        out_dir,
        fig_size,
        vbs=True):

#     cm = plt.get_cmap('gist_rainbow')

    out_dir = Path(out_dir)

    out_dir.mkdir(exist_ok=True)

    if vbs:
        print('Plotting...\n')

    fig_cols = 2
    fig_rows = 5

    alpha = 0.03

    fig_shape = (fig_rows, fig_cols)

    unq_stns = []
    for i, plot_idx in enumerate(plot_idxs):
        (stn_as,
         stn_bs,
         data_val,
         period_idx_a,
         period_idx_b) = plot_data[plot_idx]

        if vbs:
            print(
                'Plotting:',
                i,
                stn_as,
                stn_bs,
                np.round(data_val, 3))

        if stn_as not in unq_stns:
            unq_stns.append(stn_as)

        plt.figure(figsize=fig_size)

        corrs_ax = plt.subplot2grid((fig_shape), (0, 0), 1, 2)

        ecop_ll_a_ax = plt.subplot2grid((fig_shape), (1, 0), 2, 1)
        ecop_ll_b_ax = plt.subplot2grid((fig_shape), (1, 1), 2, 1)

        ecop_uu_a_ax = plt.subplot2grid((fig_shape), (3, 0), 2, 1)
        ecop_uu_b_ax = plt.subplot2grid((fig_shape), (3, 1), 2, 1)

        # ft correlations
        corrs_ax.plot(
            ft_corr_sers[stn_as][1][period_idx_a],
            color='C0',
            alpha=0.7,
            label=f'{stn_as[0]} & {stn_as[1]}')

        if stn_bs not in unq_stns:
            unq_stns.append(stn_bs)

        corrs_ax.plot(
            ft_corr_sers[stn_bs][1][period_idx_b],
            color='C1',
            alpha=0.7,
            label=f'{stn_bs[0]} & {stn_bs[1]}')

        corrs_ax.set_ylim(0, 1)

        corrs_ax.set_xlabel('Frequency [-]\n')
        corrs_ax.set_ylabel('Correlation [-]')

        corrs_ax.legend(loc=4)
        corrs_ax.grid()

        # ecops

        beg_time_a, end_time_a = ft_corr_sers[stn_as][0][period_idx_a, :]
        beg_time_b, end_time_b = ft_corr_sers[stn_bs][0][period_idx_b, :]

        sers_a = data_df.loc[beg_time_a: end_time_a, stn_as]
        sers_b = data_df.loc[beg_time_b: end_time_b, stn_bs]

        ser_a_probs = sers_a.rank(axis=0) / (sers_a.shape[0] + 1.0)
        ser_b_probs = sers_b.rank(axis=0) / (sers_b.shape[0] + 1.0)

        # upper-upper ecop
        uu_ecop_lims = (0.95, 1.0)

        ecop_uu_a_ax.scatter(
            ser_a_probs.values[:, 0],
            ser_a_probs.values[:, 1],
            alpha=min(alpha / (uu_ecop_lims[1] - uu_ecop_lims[0]), 1),
            color='C0',
            marker='o')

        ecop_uu_b_ax.scatter(
            ser_b_probs.values[:, 0],
            ser_b_probs.values[:, 1],
            alpha=min(alpha / (uu_ecop_lims[1] - uu_ecop_lims[0]), 1),
            color='C1',
            marker='o')

        ecop_uu_a_ax.set_xlim(*uu_ecop_lims)
        ecop_uu_b_ax.set_xlim(*ecop_uu_a_ax.get_xlim())

        ecop_uu_a_ax.set_ylim(*ecop_uu_a_ax.get_xlim())
        ecop_uu_b_ax.set_ylim(*ecop_uu_a_ax.get_xlim())

        ecop_uu_a_ax.set_xticks(
            np.linspace(uu_ecop_lims[0], uu_ecop_lims[1], 4))

        ecop_uu_a_ax.set_xticklabels(
            [f'{tick:3.2f}' for tick in ecop_uu_a_ax.get_xticks()],
            rotation=90)

        ecop_uu_b_ax.set_xticks(ecop_uu_a_ax.get_xticks())

        ecop_uu_b_ax.set_xticklabels(
            [f'{tick:3.2f}' for tick in ecop_uu_a_ax.get_xticks()],
            rotation=90)

        ecop_uu_a_ax.set_yticks(ecop_uu_a_ax.get_xticks())

        ecop_uu_a_ax.set_yticklabels(
            [f'{tick:3.2f}' for tick in ecop_uu_a_ax.get_yticks()])

        ecop_uu_b_ax.set_yticks(ecop_uu_a_ax.get_yticks())

        ecop_uu_b_ax.set_yticklabels([])

        ecop_uu_a_ax.grid()
        ecop_uu_b_ax.grid()

        ecop_uu_a_ax.set_xlabel('Probability [-]')
        ecop_uu_b_ax.set_xlabel('Probability [-]')

        ecop_uu_a_ax.set_ylabel('Probability [-]')

        # lower-lower ecop
        ll_ecop_lims = (0.0, 1.0)

        ecop_ll_a_ax.scatter(
            ser_a_probs.values[:, 0],
            ser_a_probs.values[:, 1],
            alpha=min(alpha / (ll_ecop_lims[1] - ll_ecop_lims[0]), 1),
            color='C0',
            marker='o')

        ecop_ll_b_ax.scatter(
            ser_b_probs.values[:, 0],
            ser_b_probs.values[:, 1],
            alpha=min(alpha / (ll_ecop_lims[1] - ll_ecop_lims[0]), 1),
            color='C1',
            marker='o')

        ecop_ll_a_ax.set_xlim(*ll_ecop_lims)
        ecop_ll_b_ax.set_xlim(*ecop_ll_a_ax.get_xlim())

        ecop_ll_a_ax.set_ylim(*ecop_ll_a_ax.get_xlim())
        ecop_ll_b_ax.set_ylim(*ecop_ll_a_ax.get_xlim())

        ecop_ll_a_ax.set_xticks(
            np.linspace(ll_ecop_lims[0], ll_ecop_lims[1], 4))

        ecop_ll_a_ax.set_xticklabels(
            [f'{tick:3.2f}' for tick in ecop_ll_a_ax.get_xticks()],
            rotation=90)

        ecop_ll_b_ax.set_xticks(ecop_ll_a_ax.get_xticks())

        ecop_ll_b_ax.set_xticklabels(
            [f'{tick:3.2f}' for tick in ecop_ll_a_ax.get_xticks()],
            rotation=90)

        ecop_ll_a_ax.set_yticks(ecop_ll_a_ax.get_xticks())

        ecop_ll_a_ax.set_yticklabels(
            [f'{tick:3.2f}' for tick in ecop_ll_a_ax.get_yticks()])

        ecop_ll_b_ax.set_yticks(ecop_ll_a_ax.get_yticks())

        ecop_ll_b_ax.set_yticklabels([])

        ecop_ll_a_ax.grid()
        ecop_ll_b_ax.grid()

        ecop_ll_a_ax.set_ylabel('Probability [-]')

        corrs_ax.set_aspect('auto', 'box', 'C')
        ecop_ll_a_ax.set_aspect('equal', 'box', 'SE')
        ecop_ll_b_ax.set_aspect('equal', 'box', 'SE')
        ecop_uu_a_ax.set_aspect('equal', 'box', 'SE')
        ecop_uu_b_ax.set_aspect('equal', 'box', 'SE')

        fig_name = (
            f'{fig_name_suff}_{i}_{stn_as[0]}_{stn_as[1]}_{period_idx_a}_'
            f'{stn_bs[0]}_{stn_bs[1]}_{period_idx_b}.png')

        plt.subplots_adjust(0, 0, 1.0, 1.0, 0.1, 0.2)

#         plt.tight_layout(pad=0.0)

        plt.savefig(str(out_dir / fig_name), bbox_inches='tight', dpi=DPI)

        plt.close()

    if vbs:
        print('Done plotting!\n')

    return


def main():

    main_dir = Path(
        r'P:\Synchronize\IWS\Testings\fourtrans_practice\similar_'
        r'dissimilar_series')

    os.chdir(main_dir)

    in_files_patt = r'ft_corr_series_*_*.pkl'
    in_data_file = r'combined_discharge_df.pkl'

    corrs_abs_diff = 0.01
    steps_abs_diff_ratio = 0  # 0.005

    corr_selection_method = 'min'
    out_dir = main_dir / corr_selection_method
    fig_name_suff = f'{corr_selection_method}_corrs'

    data_pkl_name = (
        f'ft_corr_data_{corr_selection_method}_cr'
        f'{int(corrs_abs_diff*1000):04d}_'
        f'st{int(steps_abs_diff_ratio*1000):04d}.pkl')

    fig_size = (6, 9)

    # None for all
    max_n_files = None
    max_plot_figs = 100

    vbs_flag = False

    in_files = list(main_dir.glob(in_files_patt))

    print('\n\n')
    print('#' * 50)
    print('Found %d files!' % len(in_files))
    print('#' * 50)
    print('\n\n')

    ft_corr_sers, corrs, steps, stns, periods = get_input_data(
        in_files[:max_n_files])

    corrs_data = get_corrs_data(
        ft_corr_sers,
        corrs,
        steps,
        stns,
        periods,
        corrs_abs_diff,
        steps_abs_diff_ratio,
        corr_selection_method,
        vbs_flag)

    print('\n\n')
    print('#' * 50)
    print('corrs_data length:', len(corrs_data))
    print('#' * 50)
    print('\n\n')

    out_dir.mkdir(exist_ok=True)

    with open(out_dir / data_pkl_name, 'wb') as pkl_hdl:
        pickle.dump(corrs_data, pkl_hdl)

    fin_vals = np.array([corr_data[2] for corr_data in corrs_data])

    fin_val_sort_idxs = np.argsort(fin_vals)[::-1]

    data_df = pd.read_pickle(in_data_file)

    plot_corr_series(
        fin_val_sort_idxs[:max_plot_figs],
        corrs_data,
        ft_corr_sers,
        data_df,
        fig_name_suff,
        out_dir,
        fig_size,
        vbs_flag)

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
