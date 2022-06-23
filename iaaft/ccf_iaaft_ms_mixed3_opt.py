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
import matplotlib.pyplot as plt; plt.ioff()
from scipy.stats import rankdata, norm, expon
from scipy.optimize import differential_evolution, minimize

DEBUG_FLAG = False

if True:
    inv_dist = norm

else:
    inv_dist = expon


def get_mag_phs_spec(arr):

    assert isinstance(arr, np.ndarray)
    assert arr.ndim == 1
    assert np.all(np.isfinite(arr))
    assert arr.size >= 3

    ft = np.fft.rfft(arr)

    mag_spec = np.abs(ft)
    phs_spec = np.angle(ft)

    return mag_spec[1:], phs_spec[1:]


def get_ft_cum_corr(arr_1, arr_2):

    mag_spec_1, phs_spec_1 = get_mag_phs_spec(arr_1)
    mag_spec_2, phs_spec_2 = get_mag_phs_spec(arr_2)

    assert mag_spec_1.size == mag_spec_2.size
    assert phs_spec_1.size == phs_spec_2.size

    mag_specs_prod = mag_spec_1 * mag_spec_2

    denom_corrs = (
        ((mag_spec_1 ** 2).sum() ** 0.5) *
        ((mag_spec_2 ** 2).sum() ** 0.5))

    corr = (mag_specs_prod * np.cos(phs_spec_1 - phs_spec_2)).cumsum()

    return corr, denom_corrs

# def get_corrs(data_rand):
#
#     pc = np.corrcoef(data_rand[:, 0], data_rand[:, 1])[0, 1]
#
#     sc = np.corrcoef(rankdata(data_rand[:, 0]), rankdata(data_rand[:, 1]))[0, 1]
#
#     return round(pc, 3), round(sc, 3)


def get_sim_dict(df_data, cols, ratio_a, ratio_b):

    n_sims = int(50)
    n_repeat = 3
    max_opt_iters = int(1e5)

    data = df_data.values.copy()

    # n_steps = data.shape[0]

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

    data_sort = np.sort(data, axis=0)

    order_ref = np.argsort(np.argsort(data, axis=0), axis=0)

    sims = {cols[k]:{'ref': data[:, k].copy()} for k in range(len(cols))}

    sim_zeros_str = len(str(max_opt_iters))

    ref_phs_diffs = ref_phs - ref_phs[:, [0]]
    ref_phs_ranks_diffs = ref_phs_ranks - ref_phs_ranks[:, [0]]
    for j in range(n_sims):
        # print('sim:', j)

        order_old = np.empty(data_sort.shape, dtype=int)

        data_rand = np.empty_like(data)

        for k in range(len(cols)):
            # Fully random.
            order_old[:, k] = np.argsort(np.argsort(
                np.random.random(data_sort.shape[0])))

            data_rand[:, k] = data_sort[order_old[:, k], k]

        # print('i', *get_corrs(data_rand))

        for _ in range(n_repeat):

            if True:
                sqdiffs = []
                for _ in range(max_opt_iters):

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
                        (ratio_b * sim_ift_b)
                        )

                    order_new = np.empty_like(order_old)
                    for k in range(len(cols)):
                        order_new[:, k] = np.argsort(np.argsort(sim_ift[:, k]))

                    order_sdiff = ((order_old - order_new) ** 2).sum()

                    sqdiffs.append(order_sdiff)

                    order_old = order_new

                    data_rand = np.empty_like(data)

                    for k in range(len(cols)):
                        data_rand[:, k] = data_sort[order_old[:, k], k]

                    if order_sdiff == 0:
                        break

                # if (i + 1) == max_opt_iters:
                #     print('max_opt_iters!')

                # print(f'l{m}', *get_corrs(data_rand))
                #==============================================================

            if True:
                sqdiffs = []
                for _ in range(max_opt_iters):

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
                        (ratio_b * sim_ift_b)
                        )

                    order_new = np.empty_like(order_old)
                    for k in range(len(cols)):
                        order_new[:, k] = np.argsort(np.argsort(sim_ift[:, k]))

                    order_sdiff = ((order_old - order_new) ** 2).sum()

                    sqdiffs.append(order_sdiff)

                    order_old = order_new

                    data_rand = np.empty_like(data)

                    for k in range(len(cols)):
                        data_rand[:, k] = data_sort[order_old[:, k], k]

                    if order_sdiff == 0:
                        break

                # if (i + 1) == max_opt_iters:
                #     print('max_opt_iters!')

                # print(f'l{m}', *get_corrs(data_rand))
                #==============================================================

        for k, col in enumerate(cols):
            sims[col][f'sims_{j:0{sim_zeros_str}d}'] = data_rand[:, k]

        # print(
        #     'l',
        #     round(np.corrcoef(data_rand[:, 0], data_rand[:, 1])[0, 1], 3))

        order_sdiff = ((order_old - order_ref) ** 2).sum()

        # print(i + 1, int(order_sdiff))

        # print('')

    # print('r', *get_corrs(data))
    # print('')

    return sims


def obj_ftn(prms, df_data, cols):

    # ratio_a , ratio_b, ratio_c = prms
    # ratio_a , ratio_b = prms
    ratio_a, = prms

    if not (0.2 < ratio_a < 0.6):
        return 1e7 * (1 + np.random.random())

    ratio_b = 1 - ratio_a
    #==========================================================================

    sims = get_sim_dict(df_data, cols, ratio_a, ratio_b)
    #==========================================================================

    obj_val = 0.0

    # Auto pearson.
    for col in cols:
        ref_ft_corr, ref_ft_corr_denom = get_ft_cum_corr(
            *([sims[col]['ref']] * 2))

        ref_ft_corr /= ref_ft_corr_denom

        for sim_lab, sim in sims[col].items():
            if sim_lab == 'ref':
                continue

            else:
                sim_ft_corr = get_ft_cum_corr(*([sim] * 2))[0]

                sim_ft_corr /= ref_ft_corr_denom

                obj_val += ((ref_ft_corr - sim_ft_corr) ** 2).sum()

    # Auto spearman.
    for col in cols:
        ref_ft_corr, ref_ft_corr_denom = get_ft_cum_corr(
            *([rankdata(sims[col]['ref'])] * 2))

        ref_ft_corr /= ref_ft_corr_denom

        for sim_lab, sim in sims[col].items():
            if sim_lab == 'ref':
                continue

            else:
                sim_ft_corr = get_ft_cum_corr(*([rankdata(sim)] * 2))[0]

                sim_ft_corr /= ref_ft_corr_denom

                obj_val += ((ref_ft_corr - sim_ft_corr) ** 2).sum()

    # Cross pearson.
    ref_ft_corr_cross, ref_ft_corr_cross_denom = get_ft_cum_corr(
        sims[cols[0]]['ref'], sims[cols[1]]['ref'])

    ref_ft_corr_cross /= ref_ft_corr_cross_denom

    for sim_lab in sims[cols[0]].keys():
        if sim_lab == 'ref':
            continue

        sim_ft_corr_cross = get_ft_cum_corr(
            sims[cols[0]][sim_lab], sims[cols[1]][sim_lab])[0]

        sim_ft_corr_cross /= ref_ft_corr_cross_denom

        obj_val += ((ref_ft_corr_cross - sim_ft_corr_cross) ** 2).sum()

    # Cross spearman.
    ref_ft_corr_cross, ref_ft_corr_cross_denom = get_ft_cum_corr(
        rankdata(sims[cols[0]]['ref']),
        rankdata(sims[cols[1]]['ref']))

    ref_ft_corr_cross /= ref_ft_corr_cross_denom

    for sim_lab in sims[cols[0]].keys():
        if sim_lab == 'ref':
            continue

        sim_ft_corr_cross = get_ft_cum_corr(
            rankdata(sims[cols[0]][sim_lab]),
            rankdata(sims[cols[1]][sim_lab]))[0]

        sim_ft_corr_cross /= ref_ft_corr_cross_denom

        obj_val += ((ref_ft_corr_cross - sim_ft_corr_cross) ** 2).sum()

    print('iter prms:', prms)
    print('obj_val:', obj_val)
    print('')
    return obj_val


def main():

    main_dir = Path(
        r'P:\Synchronize\IWS\Testings\fourtrans_practice\iaaft')

    os.chdir(main_dir)

    # in_data_file = Path(r'neckar_q_data_combined_20180713.csv')
    #
    # sep = ';'
    #
    # beg_time = '1961-01-01'
    # # end_time = '2015-12-31'
    # end_time = '1970-12-31'
    #
    # cols = ['3465', '420']

    # From Prof.
    in_data_file = Path(r'BW_dwd_stns_60min_1995_2020_data.csv')

    # All these have no missing values in BW fro 2010 to 2014.
    # cols = [
    #     'P00071', 'P00257', 'P00279', 'P00498', 'P00684', 'P00757', 'P00931',
    #     'P01089', 'P01216', 'P01224', 'P01255', 'P01290', 'P01584', 'P01602', ]
    #     'P01711', 'P01937', 'P02388', 'P02575', 'P02638', 'P02787', 'P02814',
    #     'P02880', 'P03278', 'P03362', 'P03519', 'P03761', 'P03925', 'P03927',
    #     'P04160', 'P04175', 'P04294', 'P04300', 'P04315', 'P04349', 'P04623',
    #     'P04710', 'P04881', 'P04928', 'P05229', 'P05664', 'P05711', 'P05724',
    #     'P05731', 'P06258', 'P06263', 'P06275', 'P07138', 'P07187', 'P07331',
    #     'P13672', 'P13698', 'P13965']

    # cols = 'P13698;P07331;P13672;P02575;P02814;P00279;P06275;P02787;P05711;P03278;P03761'.split(';')
    cols = 'P13698;P07331'.split(';')

    beg_time = '2010-01-01 00:00:00'
    end_time = '2010-12-31 23:00:00'

    sep = ';'

    # out_dir = Path(r'iaaft_test_ms_mixed3_42')
    out_dir = Path(r'iaaft_ppt_hourly_test_02_opt')
    #==========================================================================

    out_dir.mkdir(exist_ok=True)

    df_data = pd.read_csv(in_data_file, sep=sep, index_col=0)

    df_data = df_data.loc[beg_time:end_time, cols]

    if df_data.shape[0] % 2:
        df_data = df_data.iloc[:-1,:]

    bds = np.array([
        [0.2, 0.6],
        # [0.0, 1.0],
        # [0.0, 0.0],
        ])

    if True:
        opt_ress = differential_evolution(obj_ftn, bds, args=(df_data, cols), workers=8, maxiter=10, updating='deferred')

    else:
        opt_ress = minimize(obj_ftn, 0.5, args=(df_data, cols), method='BFGS')

    prms = opt_ress.x
    obj_val = opt_ress.fun

    print(prms)
    print(obj_val)
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
