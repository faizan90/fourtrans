'''
@author: Faizan-Uni-Stuttgart

'''
import os
import timeit
import time
from pathlib import Path
from math import factorial
from itertools import combinations

import matplotlib as mpl
mpl.rc('font', size=14)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.ioff()

DPI = 500


def main():

    main_dir = Path(
        r'P:\Synchronize\IWS\Projects\2016_DFG_SPATE\data\simultaneous_'
        r'extremes\multuplets')

    os.chdir(main_dir)

    data_file = r'neckar_daily_discharge_1961_2015.csv'

    out_suff = 'neckar'

    beg_time = '1950-01-01'
    end_time = '2015-10-31'
    fmt_time = '%Y-%m-%d'
    sep = ';'

    n_comb_max = 4
    n_comb_min = 3

    return_period = 1
    steps_per_period = 365
    steps_per_period_label = 'year'

    assert n_comb_max > 1
    assert n_comb_max > n_comb_min
    assert return_period > 0
    assert steps_per_period > 0

    data_df = pd.read_csv(data_file, sep=sep, index_col=0)
    data_df.index = pd.to_datetime(data_df.index, format=fmt_time)
    data_df = data_df.loc[beg_time:end_time]

    n_rows, n_cols = data_df.shape

    print(f'n_rows: {n_rows}, n_cols: {n_cols}')

    assert n_rows > (return_period * steps_per_period)

    assert n_cols >= n_comb_max

    rank_df = data_df.rank(axis=0, ascending=False)

    limit_rank = int(n_rows / (return_period * steps_per_period))
    assert limit_rank > 0

    assert np.all(rank_df.max(axis=0) > limit_rank)

    active_cts = []
    stn_combs = combinations(rank_df.columns, n_comb_max)

    n_combs = int(
        factorial(n_cols) /
        (factorial(n_comb_max) * factorial(n_cols - n_comb_max)))

    print('Number of combinations to test:', n_combs)

    print(f'{0:8d}', time.asctime())
    for i, stn_comb in enumerate(stn_combs, start=1):
        comb_df = rank_df.loc[:, stn_comb]

        status_df = (comb_df <= limit_rank).sum(axis=1)

        min_active_ct = (status_df >= n_comb_min).values.sum()
        max_active_ct = (status_df == n_comb_max).values.sum()

        active_cts.append([min_active_ct, max_active_ct])

        if not (i % 5000):
            print(f'{i:8d}', time.asctime())
#             break

    print(f'{len(active_cts):8d}', time.asctime())

    active_cts = np.array(active_cts) / (rank_df.shape[0] + 1)

    max_prob = min(1, active_cts.max() * 1.1)

    plt.figure(figsize=(7.0, 7.0))

    plt.scatter(active_cts[:, 0], active_cts[:, 1], alpha=0.3)

    plt.xlim(0.0, max_prob)
    plt.ylim(0.0, max_prob)

    ticks = np.linspace(0, max_prob, 5)
    tick_labels = [f'{tick * 100:0.4f}' for tick in ticks]

    plt.xticks(ticks, tick_labels, rotation=90)
    plt.yticks(ticks, tick_labels)

    plt.gca().set_aspect('equal', 'box', 'SE')

    plt.xlabel(
        f'\nPercent probability that at least {n_comb_min} out of '
        f'{n_comb_max} stations have\nan '
        f'extreme value with a return period of '
        f'{return_period}-{steps_per_period_label}(s) or\nmore on '
        f'the same day')

    plt.ylabel(
        f'Percent probability that all '
        f'{n_comb_max} stations have an\n'
        f'extreme value with a return period of\n'
        f'{return_period}-{steps_per_period_label}(s) or more on '
        f'the same day\n')

    plt.title(
        f'Simultaneous occurrence of extreme values\n\n'
        f'Total number of stations: {n_cols}\n'
        f'Extreme value threshold return period: '
        f'{return_period}-{steps_per_period_label}(s)\n'
        f'Total number of steps: {n_rows}, '
        f'Total number of combinations: {n_combs}\n'
        f'Maximum number of possible events in the entire '
        f'time series per station: {limit_rank}\n',
        ha='right',
        loc='right')

    plt.gca().set_axisbelow(True)
    plt.grid()

    fig_name = (
        f'simultext_probs_cmpr_lr{limit_rank}_mn{n_comb_min}_'
        f'mx{n_comb_max}_rp{return_period}_{out_suff}.png')

    plt.savefig(fig_name, bbox_inches='tight', dpi=DPI)

#     plt.show()

    plt.close()
    return

#     # file_1 should have less combination length
#     # than file_2.
#     # Also, combination length in each file should be constant
#     in_h5_file_1 = r'simultexts_db_duplets.hdf5'
#     in_h5_file_2 = r'simultexts_db_triplets.hdf5'
#
#     (stn_combs_ds_1,
#      simult_ext_evts_cts_ds_1,
#      excd_probs_ds_1,
#      time_wins_ds_1) = get_data(in_h5_file_1)
#
#     comb_len_ds_1 = len(stn_combs_ds_1[0])
#     assert all((len(comb) == comb_len_ds_1 for comb in stn_combs_ds_1))
#
#     (stn_combs_ds_2,
#      simult_ext_evts_cts_ds_2,
#      excd_probs_ds_2,
#      time_wins_ds_2) = get_data(in_h5_file_2)
#
#     comb_len_ds_2 = len(stn_combs_ds_2[0])
#     assert all((len(comb) == comb_len_ds_2 for comb in stn_combs_ds_2))
#
# import h5py
#
#
# def get_data(h5_path):
#
#     with h5py.File(h5_path, mode='r', driver=None) as h5_hdl:
#         sim_keys = list(h5_hdl['simultexts_sims'].keys())
#
#         assert len(sim_keys) == 1
#
#         sim_key = sim_keys[0]
#
#         all_stn_combs = h5_hdl[
#             f'simultexts_sims/{sim_key}/all_stn_combs'][...]
#
#         stn_combs = [eval(comb) for comb in all_stn_combs]
#
#         simult_ext_evts_cts = h5_hdl[
#             f'simultexts_sims/{sim_key}/simult_ext_evts_cts'][...]
#
#         excd_probs = h5_hdl['excd_probs'][...]
#         time_windows = h5_hdl['time_windows'][...]
#
#     return stn_combs, simult_ext_evts_cts, excd_probs, time_windows


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
