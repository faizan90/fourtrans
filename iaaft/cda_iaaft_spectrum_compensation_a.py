'''
@author: Faizan-Uni-Stuttgart

Aug 4, 2022

9:04:40 AM

'''
import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm
import matplotlib.pyplot as plt; plt.ioff()

DEBUG_FLAG = False


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Testings\fourtrans_practice\iaaft')
    os.chdir(main_dir)

    in_data_file = Path(r'neckar_q_data_combined_20180713_10cps.csv')

    sep = ';'

    beg_time = '1961-08-01'
    end_time = '1975-07-31'
    #==========================================================================

    df_data = pd.read_csv(in_data_file, sep=sep, index_col=0)

    data = df_data.loc[beg_time:end_time, '420']

    if data.shape[0] % 2:
        data = data[:-1]

    assert np.all(np.isfinite(data))

    if False:
        probs = rankdata(data) / (data.size + 1.0)

        data = norm.ppf(probs)

    ft = np.fft.rfft(data)

    ref_mag = np.abs(ft)

    pwr = ref_mag[1:] ** 2

    pwr = pwr.cumsum()

    periods = (pwr.size * 2) / np.arange(1, pwr.size + 1)

    assert periods.size == pwr.shape[0]

    plt.semilogx(
        periods,
        pwr,
        alpha=0.75,
        color='r',
        label='ref',
        lw=3.0,
        zorder=1)

    #==========================================================================
    data_sort = np.sort(data)
    data_rand = np.empty_like(data)
    order_old = np.empty(data_sort.shape, dtype=int)
    order_old[:] = np.argsort(np.argsort(np.random.random(data_sort.shape[0])))

    data_rand[:] = data_sort[order_old[:]]

    old_ratio_scale = 1.0
    for i in range(10):

        sim_ft_margs = np.fft.rfft(data_rand, axis=0)

        sim_mag = np.abs(sim_ft_margs)

        if True:
            # mag_ratio = ref_mag / sim_mag
            #
            # ref_mag_sum = ref_mag.sum()
            # ref_mag_ratio_sum = (ref_mag * mag_ratio).sum()
            #
            # mag_ratio *= ref_mag_sum / ref_mag_ratio_sum

            pwr_exp = 0.5

            if old_ratio_scale < 1:
                mag_ratio = ((sim_mag ** pwr_exp) / (ref_mag ** pwr_exp))[1:]

            else:
                mag_ratio = ((ref_mag ** pwr_exp) / (sim_mag ** pwr_exp))[1:]

            ref_mag_sum = (ref_mag[1:] ** pwr_exp).sum()

            ref_mag_ratio_sum = ((ref_mag[1:] ** pwr_exp) * mag_ratio).sum()

            mag_ratio *= ref_mag_sum / ref_mag_ratio_sum

            mag_ratio = np.concatenate(([1.0], mag_ratio))

            # mag_ratio[1:] *= old_ratio_scale

            print(
                i,
                mag_ratio[1:].mean(),
                (ref_mag * mag_ratio)[1:].sum() / ref_mag[1:].sum())

            old_ratio_scale = 1 / mag_ratio[1:].mean()

        else:
            mag_ratio = 1.0

        sim_phs_margs = np.angle(sim_ft_margs)

        sim_ft_new = np.empty_like(sim_ft_margs)

        sim_ft_new.real[:] = np.cos(sim_phs_margs) * ref_mag * mag_ratio
        sim_ft_new.imag[:] = np.sin(sim_phs_margs) * ref_mag * mag_ratio

        sim_ft_new[0] = 0

        sim_ift_a_auto = np.fft.irfft(sim_ft_new, axis=0)

        order_new_a = np.empty_like(order_old)
        order_new_a[:] = np.argsort(np.argsort(sim_ift_a_auto[:]))

        data_rand[:] = data_sort[order_new_a[:]]

        ft_rand = np.fft.rfft(data_rand)
        rand_mag = np.abs(ft_rand)

        pwr_rand = rand_mag[1:] ** 2

        pwr_rand = pwr_rand.cumsum()

        plt.semilogx(
            periods,
            pwr_rand,
            alpha=0.5,
            color=f'C{i}',
            label=i,
            lw=1.5,
            zorder=2)
    #==========================================================================

    plt.legend()

    plt.grid()
    plt.gca().set_axisbelow(True)

    plt.xlabel('Period')
    plt.ylabel('Cummulative power')

    plt.xlim(plt.xlim()[::-1])

    plt.show()
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
