'''
@author: Faizan-Uni-Stuttgart

10 Mar 2020

12:03:49

'''

import os
import timeit
import time
import pickle
import random
from pathlib import Path
from math import factorial
from itertools import combinations
from multiprocessing import Pool, Manager

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pandas.plotting import register_matplotlib_converters

plt.ioff()
register_matplotlib_converters()

DEBUG_FLAG = True


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
    tid, beg_j, end_j, in_file, min_corr_steps, comb_size, out_dir, lock = args

    out_pkl = Path(out_dir) / f'ft_corr_series_{beg_j}_{end_j}.pkl'

    df = pd.read_pickle(in_file)

    print_pctge = 0.1

    n_cols = df.shape[1]

    assert n_cols >= comb_size

    combs = combinations(df.columns, comb_size)

    n_vals = end_j - beg_j

    assert n_vals > 0

    fnts_df = np.isfinite(df)

    unq_take_idxs_stns = {}

    df_idx = df.index

    n_vals_ctr = 0
    for j, stns in enumerate(combs, start=1):
        if not (beg_j <= j < end_j):
            continue

        n_vals_ctr += 1

        if not (n_vals_ctr % int(print_pctge * n_vals)):
            with lock:
                print('#############################################')
                print(f'Status: {tid}')
                print(f'beg_j: {beg_j}, end_j: {end_j}, j: {j}')
                print(f'n_vals_ctr: {n_vals_ctr}, n_vals: {n_vals}')
                print('#############################################')
                print('\n')

        valid_idxs = np.prod(fnts_df.loc[:, stns].values, axis=1).astype(bool)

#             plt.plot(stn_valid_idxs, label=stn, alpha=0.5)
#
#         plt.plot(valid_idxs, label='fin', alpha=0.5)
#         plt.grid()
#         plt.legend()
#
#         plt.show()

        n_vld_idxs = valid_idxs.sum()

        assert n_vld_idxs >= 0

        if n_vld_idxs < min_corr_steps:
            continue

        n_taken_steps = 0
        n_rjctd_steps = 0

        # easier to handle if first step is invalid
        vld_idxs_int = np.concatenate(([0], valid_idxs.astype(int)))

        diffs = vld_idxs_int[1:] - vld_idxs_int[:-1]

        valid_idxs_wh = np.where(diffs == 1)[0]
        non_valid_idxs_wh = np.where(diffs == -1)[0]

        for i, valid_idx_wh in enumerate(valid_idxs_wh):

            cvld_steps = non_valid_idxs_wh[i] - valid_idx_wh

            if cvld_steps >= min_corr_steps:

                cidxs = (
                    df_idx[valid_idx_wh],
                    df_idx[non_valid_idxs_wh[i] - 1])

#                 assert np.all(np.isfinite(df.loc[cidxs[0]:cidxs[1], stns]))

                n_taken_steps += cvld_steps

                idxs_hash = hash(cidxs)

                if idxs_hash not in unq_take_idxs_stns:
                    unq_take_idxs_stns[idxs_hash] = (cidxs, [], [])

                unq_hash_stns = unq_take_idxs_stns[idxs_hash][1]
                unq_take_idxs_stns[idxs_hash][2].append(stns)

                for stn in stns:
                    if stn in unq_hash_stns:
                        continue

                    unq_hash_stns.append(stn)

            else:
                n_rjctd_steps += cvld_steps

                assert 0 <= cvld_steps < min_corr_steps

        assert (n_taken_steps + n_rjctd_steps) == n_vld_idxs

    with lock:
        print('#############################################')
        print(f'Status: {tid}')
        print('Computing FT...')
        print('#############################################')
        print('\n')

    n_acpt_sers = 0
    n_combs_dist = []
    corr_sers = {}
    for (idxs, stns, acpt_combs) in unq_take_idxs_stns.values():
        n_combs_dist.append(len(acpt_combs))

        comb_df = df.loc[idxs[0]:idxs[1], stns]

        if comb_df.shape[0] % 2:
            ft_df = comb_df.iloc[1:, ].apply(np.fft.rfft, axis=0).iloc[1:, :]

        else:
            ft_df = comb_df.apply(np.fft.rfft, axis=0).iloc[1:, :]

        mag_df = ft_df.apply(np.abs)
        phs_df = ft_df.apply(np.angle)

        for acpt_comb in acpt_combs:
            ft_mags = mag_df.loc[:, acpt_comb].values
            ft_phas = phs_df.loc[:, acpt_comb].values

            mags_prod = np.prod(ft_mags, axis=1)

            assert mags_prod.size == ft_df.shape[0]

            min_phs = ft_phas.min(axis=1)
            max_phs = ft_phas.max(axis=1)

            assert min_phs.size == ft_df.shape[0]
            assert max_phs.size == ft_df.shape[0]

            ft_env_cov = mags_prod * np.cos(max_phs - min_phs)

            mag_sq_sum = np.prod((ft_mags ** 2).sum(axis=0)) ** 0.5

            assert mag_sq_sum > 0

            ft_corr = np.cumsum(ft_env_cov) / mag_sq_sum

            assert np.isfinite(ft_corr[-1])

            if acpt_comb not in corr_sers:
                corr_sers[acpt_comb] = [[idxs], [ft_corr]]

            else:
                corr_sers[acpt_comb][0].append(idxs)
                corr_sers[acpt_comb][1].append(ft_corr)

            n_acpt_sers += 1

    with lock:
        with open(out_pkl, 'wb') as pkl_hdl:
            pickle.dump(corr_sers, pkl_hdl)

        print('#############################################')
        print(f'Status: {tid}')
        print(f'beg_j: {beg_j}, end_j: {end_j}')
        print(f'n_vals_ctr: {n_vals_ctr}')
        print('End of job!')
        print('#############################################')
        print('\n')

    return n_acpt_sers


def main():

    main_dir = Path(
        r'P:\Synchronize\IWS\Testings\fourtrans_practice\similar_dissimilar_series_multuplets')

    os.chdir(main_dir)

    in_file = r'combined_discharge_df.pkl'

    comb_size = 3

    out_dir = Path(f'combs_of_{comb_size}')

    n_cpus = 14

    min_corr_steps = (10 * 365)

    max_chunk_size = 1000000

    max_n_combs = 100000

    #==========================================================================

    out_dir.mkdir(exist_ok=True)

    if min_corr_steps % 2:
        print('reduced min_corr_steps by 1!')
        min_corr_steps -= 1

    assert min_corr_steps >= 2

    df = pd.read_pickle(in_file)

    random.seed(435345435346587655789345)

    shuff_cols = list(df.columns)
    random.shuffle(shuff_cols)

    df = df.loc[:, shuff_cols]

    n_cols = df.shape[1]

    assert n_cols >= comb_size

    del df

    n_combs = int(
        factorial(n_cols) /
        (factorial(comb_size) * factorial(n_cols - comb_size)))

    if max_n_combs is not None:
        n_combs = min(max_n_combs, n_combs)

    print(f'n_combs: {n_combs}')

    if n_combs > max_chunk_size:
        mp_idxs = np.arange(0, n_combs, max_chunk_size + 1)

        if mp_idxs[-1] != max_chunk_size:
            mp_idxs = np.concatenate((mp_idxs, [max_chunk_size]))

    else:
        mp_idxs = ret_mp_idxs(n_combs, n_cpus)

    print('mp_idxs size:', mp_idxs.size)

    lock = Manager().Lock()

    args = (
        (i,
         mp_idxs[i],
         mp_idxs[i + 1],
         in_file,
         min_corr_steps,
         comb_size,
         out_dir,
         lock)
        for i in range(mp_idxs.size - 1))

    if n_cpus > 1:
        mp_pool = Pool(n_cpus)

        n_acpt_sers = sum(mp_pool.map(cmpt_corr_sers, args))

        mp_pool.close()
        mp_pool.join()

    else:
        n_acpt_sers = sum(list(map(cmpt_corr_sers, args)))

    print('Overall accepted series count:', n_acpt_sers)
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
