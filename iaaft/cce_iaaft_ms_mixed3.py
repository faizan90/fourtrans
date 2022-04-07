'''
@author: Faizan-Uni-Stuttgart

Apr 4, 2022

2:58:43 PM

'''
import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm, expon
import matplotlib.pyplot as plt; plt.ioff()

DEBUG_FLAG = False

if True:
    inv_dist = norm

else:
    inv_dist = expon


def get_corrs(data_rand):

    pc = np.corrcoef(data_rand[:, 0], data_rand[:, 1])[0, 1]

    sc = np.corrcoef(rankdata(data_rand[:, 0]), rankdata(data_rand[:, 1]))[0, 1]

    return round(pc, 3), round(sc, 3)


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

    n_sims = int(1)
    n_repeat = 1
    max_opt_iters = int(1e5)

    ratio_a = 1.0
    ratio_b = 2.0
    ratio_c = 0.0

    out_dir = Path(r'iaaft_ms_mixed3_dis_test_03_cps')
    #==========================================================================

    out_dir.mkdir(exist_ok=True)

    df_data = pd.read_csv(in_data_file, sep=sep, index_col=0)

    df_data = df_data.loc[beg_time:end_time, cols]

    if df_data.shape[0] % 2:
        df_data = df_data.iloc[:-1,:]

    data = df_data.values.copy()

    n_steps = data.shape[0]

    # data = rankdata(data, axis=0) / (n_steps + 1.0)
    # data = norm.ppf(data)

    ref_ft = np.fft.rfft(data, axis=0)

    ref_ft[0,:] = 0

    ref_phs = np.angle(ref_ft)
    ref_mag = np.abs(ref_ft)

    ref_ft_ranks = np.fft.rfft(
        rankdata(data, axis=0), axis=0)

    ref_ft_ranks[0,:] = 0

    ref_phs_ranks = np.angle(ref_ft_ranks)
    ref_mag_ranks = np.abs(ref_ft_ranks)

    ref_ft_norms = np.fft.rfft(
        inv_dist.ppf(rankdata(data, axis=0) / (n_steps + 1.0)), axis=0)

    ref_ft_norms[0,:] = 0

    ref_phs_norms = np.angle(ref_ft_norms)
    ref_mag_norms = np.abs(ref_ft_norms)

    data_sort = np.sort(data, axis=0)

    order_ref = np.argsort(np.argsort(data, axis=0), axis=0)

    sims = {cols[k]:{'ref': data[:, k].copy()} for k in range(len(cols))}

    sim_zeros_str = len(str(max_opt_iters))

    ref_phs_diffs = ref_phs - ref_phs[:, [0]]
    ref_phs_ranks_diffs = ref_phs_ranks - ref_phs_ranks[:, [0]]
    ref_phs_norms_diffs = ref_phs_norms - ref_phs_norms[:, [0]]

    for j in range(n_sims):
        print('sim:', j)

        order_old = np.empty(data_sort.shape, dtype=int)

        data_rand = np.empty_like(data)

        for k in range(len(cols)):
            # Fully random.
            order_old[:, k] = np.argsort(np.argsort(
                np.random.random(data_sort.shape[0])))

            data_rand[:, k] = data_sort[order_old[:, k], k]

        print('i', *get_corrs(data_rand))

        for m in range(n_repeat):

            if True:
                sqdiffs = []
                for i in range(max_opt_iters):

                    # ##
                    sim_ft = np.fft.rfft(
                        inv_dist.ppf(rankdata(data_rand, axis=0) / (n_steps + 1.0)),
                        axis=0)

                    sim_phs = np.angle(sim_ft)

                    sim_ft_new = np.empty_like(sim_ft)

                    sim_ft_new.real[:] = np.cos(sim_phs) * ref_mag_norms
                    sim_ft_new.imag[:] = np.sin(sim_phs) * ref_mag_norms

                    sim_ft_new[0,:] = 0

                    sim_ift_c = np.fft.irfft(sim_ft_new, axis=0)

                    # ##
                    sim_ft = np.fft.rfft(
                        rankdata(data_rand, axis=0),
                        axis=0)

                    sim_phs = np.angle(sim_ft)

                    sim_ft_new = np.empty_like(sim_ft)

                    sim_ft_new.real[:] = np.cos(sim_phs) * ref_mag_ranks
                    sim_ft_new.imag[:] = np.sin(sim_phs) * ref_mag_ranks

                    sim_ft_new[0,:] = 0

                    sim_ift_b = np.fft.irfft(sim_ft_new, axis=0)

                    # ##
                    sim_ft = np.fft.rfft(data_rand, axis=0)

                    sim_phs = np.angle(sim_ft)

                    sim_ft_new = np.empty_like(sim_ft)

                    sim_ft_new.real[:] = np.cos(sim_phs) * ref_mag
                    sim_ft_new.imag[:] = np.sin(sim_phs) * ref_mag

                    sim_ft_new[0,:] = 0

                    sim_ift_a = np.fft.irfft(sim_ft_new, axis=0)

                    # ##
                    sim_ift = (
                        (ratio_a * sim_ift_a) +
                        (ratio_b * sim_ift_b) +
                        (ratio_c * sim_ift_c))

                    order_new = np.empty_like(order_old)
                    for k in range(len(cols)):
                        order_new[:, k] = np.argsort(np.argsort(sim_ift[:, k]))

                    order_sdiff = ((order_old - order_new) ** 2).sum()

                    sqdiffs.append(order_sdiff)

                    if False:
                        print(i, int(order_sdiff))

                        # print(order_old[:10])
                        # print(order_new[:10])

                    order_old = order_new

                    data_rand = np.empty_like(data)

                    for k in range(len(cols)):
                        data_rand[:, k] = data_sort[order_old[:, k], k]

                    if order_sdiff == 0:
                        break

                if (i + 1) == max_opt_iters:
                    print('max_opt_iters!')

                print(f'l{m}', *get_corrs(data_rand))
                #==============================================================

            if True:
                sqdiffs = []
                for i in range(max_opt_iters):

                    # ##
                    sim_ft = np.fft.rfft(
                        inv_dist.ppf(rankdata(data_rand, axis=0) / (n_steps + 1.0)),
                        axis=0)

                    sim_mag = np.abs(sim_ft)

                    sim_phs = np.angle(sim_ft[:, [0]]) + ref_phs_norms_diffs

                    sim_phs[0,:] = ref_phs_norms[0,:]

                    sim_ft_new = np.empty_like(sim_ft)

                    sim_ft_new.real[:] = np.cos(sim_phs) * sim_mag
                    sim_ft_new.imag[:] = np.sin(sim_phs) * sim_mag

                    sim_ft_new[0,:] = 0

                    sim_ift_c = np.fft.irfft(sim_ft_new, axis=0)

                    # ##
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

                    # ##
                    sim_ft = np.fft.rfft(data_rand, axis=0)

                    sim_mag = np.abs(sim_ft)

                    sim_phs = np.angle(sim_ft[:, [0]]) + ref_phs_diffs

                    sim_phs[0,:] = ref_phs_ranks[0,:]

                    sim_ft_new = np.empty_like(sim_ft)

                    sim_ft_new.real[:] = np.cos(sim_phs) * ref_mag
                    sim_ft_new.imag[:] = np.sin(sim_phs) * ref_mag

                    sim_ft_new[0,:] = 0

                    sim_ift_a = np.fft.irfft(sim_ft_new, axis=0)

                    # ##
                    sim_ift = (
                        (ratio_a * sim_ift_a) +
                        (ratio_b * sim_ift_b) +
                        (ratio_c * sim_ift_c))

                    order_new = np.empty_like(order_old)
                    for k in range(len(cols)):
                        order_new[:, k] = np.argsort(np.argsort(sim_ift[:, k]))

                    order_sdiff = ((order_old - order_new) ** 2).sum()

                    sqdiffs.append(order_sdiff)

                    if False:
                        print(i, int(order_sdiff))

                        # print(order_old[:10])
                        # print(order_new[:10])

                    order_old = order_new

                    data_rand = np.empty_like(data)

                    for k in range(len(cols)):
                        data_rand[:, k] = data_sort[order_old[:, k], k]

                    if order_sdiff == 0:
                        break

                if (i + 1) == max_opt_iters:
                    print('max_opt_iters!')

                print(f'l{m}', *get_corrs(data_rand))
                #==============================================================

        for k, col in enumerate(cols):
            sims[col][f'sims_{j:0{sim_zeros_str}d}'] = data_rand[:, k]

        # print(
        #     'l',
        #     round(np.corrcoef(data_rand[:, 0], data_rand[:, 1])[0, 1], 3))

        order_sdiff = ((order_old - order_ref) ** 2).sum()

        print(i + 1, int(order_sdiff))

        print('')

    print('r', *get_corrs(data))
    print('')

    for col in cols:
        col_df = pd.DataFrame(sims[col])

        print(f'{col} ref_sim_pcorrs:')
        print(col_df.corr(method='pearson').round(3).values)
        print('')

        print(f'{col} ref_sim_scorrs:')
        print(col_df.corr(method='spearman').round(3).values)
        print('')

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
