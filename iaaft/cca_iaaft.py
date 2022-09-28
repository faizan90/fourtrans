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
from scipy.stats import rankdata
import matplotlib.pyplot as plt; plt.ioff()

DEBUG_FLAG = False


def main():

    main_dir = Path(os.getcwd())
    os.chdir(main_dir)

    in_data_file = Path(
        r"P:\Synchronize\IWS\Discharge_data_longer_series\final_q_data_combined_20180713\neckar_q_data_combined_20180713.csv")

    sep = ';'

    beg_time = '1961-01-01'
    end_time = '2015-12-31'

    col = '420'

    n_sims = int(50)

    max_opt_iters = int(1e1)

    out_dir = Path(r'P:/iaaft_test2')
    #==========================================================================

    out_dir.mkdir(exist_ok=True)

    df_data = pd.read_csv(in_data_file, sep=sep, index_col=0)

    df_data = df_data.loc[beg_time:end_time, col]

    data = df_data.values.copy()

    ref_ft = np.fft.rfft(data)
    ref_mag = np.abs(ref_ft)
    ref_pwr = ref_mag ** 2

    ref_ft_ranks = np.fft.rfft(rankdata(data))
    ref_mag_ranks = np.abs(ref_ft_ranks)
    ref_pwr_ranks = ref_mag_ranks ** 2

    data_sort = np.sort(data)

    # order_ref = np.argsort(np.argsort(data))

    sims = {'ref': data.copy()}

    sim_zeros_str = len(str(max_opt_iters))

    for j in range(n_sims):
        order_old = np.argsort(np.argsort(np.random.random(data_sort.size)))

        print('sim:', j)

        data_rand = data_sort[order_old]

        sqdiffs = []
        for _ in range(max_opt_iters):

            sim_ft = np.fft.rfft(data_rand)

            sim_phs = np.angle(sim_ft)

            sim_ft_new = np.empty_like(sim_ft)

            sim_ft_new.real[:] = np.cos(sim_phs) * ref_mag
            sim_ft_new.imag[:] = np.sin(sim_phs) * ref_mag

            # sim_ft_new.real[:] = np.cos(sim_phs) * ref_mag_ranks
            # sim_ft_new.imag[:] = np.sin(sim_phs) * ref_mag_ranks

            # Incorrect with pwr.
            # sim_ft_new.real[:] = np.cos(sim_phs) * ref_pwr
            # sim_ft_new.imag[:] = np.sin(sim_phs) * ref_pwr

            sim_ift = np.fft.irfft(sim_ft_new)

            order_new = np.argsort(np.argsort(sim_ift))

            order_sdiff = ((order_old - order_new) ** 2).sum()

            sqdiffs.append(order_sdiff)

            # print(i, int(order_sdiff))
            #
            # print(order_old[:10])
            # print(order_new[:10])

            order_old = order_new

            data_rand = data_sort[order_old]

            if order_sdiff == 0:
                break

        sims[f'sims_{j:0{sim_zeros_str}d}'] = data_rand

    pd.DataFrame(sims).to_csv(
        out_dir / r'sims.csv', sep=';', float_format='%0.6f')

    if True:
        # order_sdiff = ((order_old - order_ref) ** 2).sum()
        #
        # print(i + 1, int(order_sdiff))
        # print('Done')

        # plt.semilogy(sqdiffs)

        sim_ft = np.fft.rfft(data_rand)
        sim_mag = np.abs(sim_ft)
        sim_pwr = sim_mag ** 2

        sim_ft_ranks = np.fft.rfft(rankdata(data_rand))
        sim_mag_ranks = np.abs(sim_ft_ranks)
        sim_pwr_ranks = sim_mag_ranks ** 2

        plt.semilogy(ref_pwr, label='ref_pwr_data')
        plt.semilogy(sim_pwr, label='sim_pwr_data')

        plt.semilogy(ref_pwr_ranks, label='ref_pwr_ranks')
        plt.semilogy(sim_pwr_ranks, label='sim_pwr_ranks')

        plt.xlabel('Frequency')
        plt.ylabel('Power')

        plt.grid()
        plt.gca().set_axisbelow(True)

        plt.legend()

        plt.show()

        # plt.savefig(out_dir / r'pwrs.png')
        #
        # plt.close()

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
