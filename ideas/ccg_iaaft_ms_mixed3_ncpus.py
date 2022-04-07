'''
@author: Faizan-Uni-Stuttgart

Apr 4, 2022

2:58:43 PM

'''
import os

# Numpy sneakily uses multiple threads sometimes. I don't want that.
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MPI_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

import sys
import time
import timeit
import traceback as tb
from pathlib import Path
from pathos.multiprocessing import ProcessPool

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt; plt.ioff()
from scipy.stats import rankdata

DEBUG_FLAG = False


def get_sim_dict(args):

    (data,
     cols,
     ratio_a,
     ratio_b,
     auto_spec_flag,
     cross_spec_flag,
     n_repeat,
     max_opt_iters,
     sim_idx,
     n_sims) = args

    assert any([ratio_a, ratio_b])
    assert any([auto_spec_flag, cross_spec_flag])

    ref_ft = np.fft.rfft(data, axis=0)

    ref_ft[0,:] = 0

    ref_phs = np.angle(ref_ft)
    ref_mag = np.abs(ref_ft)

    ref_ft_ranks = np.fft.rfft(rankdata(data, axis=0), axis=0)

    ref_ft_ranks[0,:] = 0

    ref_phs_ranks = np.angle(ref_ft_ranks)
    ref_mag_ranks = np.abs(ref_ft_ranks)

    data_sort = np.sort(data, axis=0)

    sims = {cols[k]:{} for k in range(len(cols))}

    sim_zeros_str = len(str(n_sims))

    ref_phs_diffs = ref_phs - ref_phs[:, [0]]
    ref_phs_ranks_diffs = ref_phs_ranks - ref_phs_ranks[:, [0]]

    order_old = np.empty(data_sort.shape, dtype=int)

    data_rand = np.empty_like(data)

    for k in range(len(cols)):
        order_old[:, k] = np.argsort(np.argsort(
            np.random.random(data_sort.shape[0])))

        data_rand[:, k] = data_sort[order_old[:, k], k]

    for _ in range(n_repeat):
        if auto_spec_flag:
            for _ in range(max_opt_iters):
                # Ranks.
                if ratio_b:
                    sim_ft = np.fft.rfft(
                        rankdata(data_rand, axis=0),
                        axis=0)

                    sim_phs = np.angle(sim_ft)

                    sim_ft_new = np.empty_like(sim_ft)

                    sim_ft_new.real[:] = np.cos(sim_phs) * ref_mag_ranks
                    sim_ft_new.imag[:] = np.sin(sim_phs) * ref_mag_ranks

                    sim_ft_new[0,:] = 0

                    sim_ift_b = np.fft.irfft(sim_ft_new, axis=0)

                else:
                    sim_ift_b = 0.0

                # Marginals.
                if ratio_a:
                    sim_ft = np.fft.rfft(data_rand, axis=0)

                    sim_phs = np.angle(sim_ft)

                    sim_ft_new = np.empty_like(sim_ft)

                    sim_ft_new.real[:] = np.cos(sim_phs) * ref_mag
                    sim_ft_new.imag[:] = np.sin(sim_phs) * ref_mag

                    sim_ft_new[0,:] = 0

                    sim_ift_a = np.fft.irfft(sim_ft_new, axis=0)

                else:
                    sim_ift_a = 0.0

                # Their sum.
                sim_ift = (
                    (ratio_a * sim_ift_a) +
                    (ratio_b * sim_ift_b)
                    )

                order_new = np.empty_like(order_old)
                for k in range(len(cols)):
                    order_new[:, k] = np.argsort(np.argsort(sim_ift[:, k]))

                order_sdiff = ((order_old - order_new) ** 2).sum()

                order_old = order_new

                data_rand = np.empty_like(data)

                for k in range(len(cols)):
                    data_rand[:, k] = data_sort[order_old[:, k], k]

                if order_sdiff == 0:
                    break
            #==================================================================

        if cross_spec_flag:
            for _ in range(max_opt_iters):
                # Ranks.
                if ratio_b:
                    sim_ft = np.fft.rfft(
                        rankdata(data_rand, axis=0),
                        axis=0)

                    sim_mag = np.abs(sim_ft)

                    sim_phs = np.angle(sim_ft[:, [0]]) + ref_phs_ranks_diffs

                    sim_phs[0,:] = ref_phs_ranks[0,:]

                    sim_ft_new = np.empty_like(sim_ft)

                    sim_ft_new.real[:] = np.cos(sim_phs) * sim_mag
                    sim_ft_new.imag[:] = np.sin(sim_phs) * sim_mag

                    sim_ft_new[0,:] = 0

                    sim_ift_b = np.fft.irfft(sim_ft_new, axis=0)

                else:
                    sim_ift_b = 0.0

                # Marginals.
                if ratio_a:
                    sim_ft = np.fft.rfft(data_rand, axis=0)

                    sim_mag = np.abs(sim_ft)

                    sim_phs = np.angle(sim_ft[:, [0]]) + ref_phs_diffs

                    sim_phs[0,:] = ref_phs_ranks[0,:]

                    sim_ft_new = np.empty_like(sim_ft)

                    sim_ft_new.real[:] = np.cos(sim_phs) * ref_mag
                    sim_ft_new.imag[:] = np.sin(sim_phs) * ref_mag

                    sim_ft_new[0,:] = 0

                    sim_ift_a = np.fft.irfft(sim_ft_new, axis=0)

                else:
                    sim_ift_a = 0.0

                # Their sum.
                sim_ift = (
                    (ratio_a * sim_ift_a) +
                    (ratio_b * sim_ift_b)
                    )

                order_new = np.empty_like(order_old)
                for k in range(len(cols)):
                    order_new[:, k] = np.argsort(np.argsort(sim_ift[:, k]))

                order_sdiff = ((order_old - order_new) ** 2).sum()

                order_old = order_new

                data_rand = np.empty_like(data)

                for k in range(len(cols)):
                    data_rand[:, k] = data_sort[order_old[:, k], k]

                if order_sdiff == 0:
                    break
            #==================================================================

    for k, col in enumerate(cols):
        sims[col][f'sims_{sim_idx:0{sim_zeros_str}d}'] = data_rand[:, k]

    print('Done with sim_idx:', sim_idx)
    return sims


def main():

    main_dir = Path(
        r'P:\Synchronize\IWS\Testings\fourtrans_practice\iaaft')

    os.chdir(main_dir)

    in_data_file = Path(r'neckar_q_data_combined_20180713_10cps.csv')

    sep = ';'

    beg_time = '1961-01-01'
    end_time = '2015-12-31'
    # end_time = '1970-12-31'

    cols = ['420', '427', '3470', '3465', '3421', 'cp']

    n_cpus = 8

    n_sims = 8 * 4

    ratio_a = 1.0
    ratio_b = 3.0

    auto_spec_flag = True
    cross_spec_flag = True

    # auto_spec_flag = False
    # cross_spec_flag = False

    n_repeat = 3
    max_opt_iters = int(1e5)

    out_dir = Path(r'iaaft_test_ncpus_02_cps')
    #==========================================================================

    assert n_cpus > 0, n_cpus

    out_dir.mkdir(exist_ok=True)

    df_data = pd.read_csv(in_data_file, sep=sep, index_col=0)

    df_data = df_data.loc[beg_time:end_time, cols]

    if df_data.shape[0] % 2:
        df_data = df_data.iloc[:-1,:]

    data = df_data.values.copy()

    args_gen = (
        (data,
         cols,
         ratio_a,
         ratio_b,
         auto_spec_flag,
         cross_spec_flag,
         n_repeat,
         max_opt_iters,
         sim_idx,
         n_sims)
        for sim_idx in range(n_sims))

    all_sims = {cols[k]:{'ref': data[:, k].copy()} for k in range(len(cols))}

    if n_cpus == 1:
        ress = []
        for args in args_gen:
            sims = get_sim_dict(args)
            ress.append(sims)

    else:
        mp_pool = ProcessPool(n_cpus)

        ress = list(mp_pool.imap(get_sim_dict, args_gen))

        mp_pool.close()
        mp_pool.join()

    for sims in ress:
        for col in cols:
            all_sims[col].update(sims[col])

    ress = sims = None

    for col in cols:
        col_df = pd.DataFrame(all_sims[col])

        # print(f'{col} ref_sim_pcorrs:')
        # print(col_df.corr(method='pearson').round(3).values)
        # print('')
        #
        # print(f'{col} ref_sim_scorrs:')
        # print(col_df.corr(method='spearman').round(3).values)
        # print('')

        col_df.to_csv(
            out_dir / f'sims_{col}.csv', sep=';', float_format='%0.6f')

    return


if __name__ == '__main__':
    print('#### Started on %s ####\n' % time.asctime())
    START = timeit.default_timer()

    #==========================================================================
    # When in post_mortem:
    # 1. "where" to show the stack,
    # 2. "up" move the stack up to an older frame,
    # 3. "down" move the stack down to a newer frame, and
    # 4. "interact" start an interactive interpreter.
    #==========================================================================

    if DEBUG_FLAG:
        try:
            main()

        except:
            pre_stack = tb.format_stack()[:-1]

            err_tb = list(tb.TracebackException(*sys.exc_info()).format())

            lines = [err_tb[0]] + pre_stack + err_tb[2:]

            for line in lines:
                print(line, file=sys.stderr, end='')

            import pdb
            pdb.post_mortem()
    else:
        main()

    STOP = timeit.default_timer()
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))
