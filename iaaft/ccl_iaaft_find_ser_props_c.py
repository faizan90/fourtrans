'''
@author: Faizan-Uni-Stuttgart

Jul 29, 2022

4:28:57 PM

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
from itertools import combinations, product

import numpy as np
import pandas as pd
from scipy.stats import rankdata, expon, norm
import matplotlib.pyplot as plt; plt.ioff()
from pathos.multiprocessing import ProcessPool

from fcopulas import (
    get_asymms_sample,
    fill_bi_var_cop_dens,
    get_asymm_1_max,
    get_asymm_2_max,
    get_etpy_min,
    get_etpy_max,
    )

from zb_cmn_ftns_plot import roll_real_2arrs_with_nan

DEBUG_FLAG = False


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Testings\fourtrans_practice\iaaft')

    os.chdir(main_dir)

    #==========================================================================
    # Daily discharge.
    #==========================================================================

    in_data_file = Path(r'neckar_q_data_combined_20180713_10cps.csv')

    sep = ';'

    beg_time = '1961-08-01'
    end_time = '1975-07-31'

    cols = [
        '420',  # '427'
        #
        # '406', '411',  # '420',
        # '4408', '406', '411', '420', '422', '427', '1438', '1439', '1452', '2431',
        # '2446', '2477', '4427', '44603', '76121', '76179', '434', '473',
        # '475', '1470', '3421', '3465', '3470', '3498', '4428', '36056', '454',
        # '1496', '2444', '4414', '4415', '4435', '62722', '76183', '464', '465',
        ]

    # All valid values from 1961 to 1965.
    # cols = [
    #     '406', '409', '411', '420', '422', '427', '463', '1438', '1439', '1452',
    #     '1458', '2431', '2446', '2477', '4408', '4427', '40670', '44603',
    #     '76121', '76179', '434', '469', '473', '475', '478', '1411', '1412',
    #     '1433', '1470', '2452', '3421', '3465', '3470', '3498', '4421', '4422',
    #     '4428', '36056', '46358', '454', '1450', '1469', '1496', '2444', '4414',
    #     '4415', '443', '4435', '62722', '76183', '464', '465', 'cp'
    #    ]

    # All valid values from 1961 to 2015.
    # cols = [
    #     '406', '411', '420', '422', '427', '1438', '1439', '1452', '2431',
    #     '2446', '2477', '4408', '4427', '44603', '76121', '76179', '434', '473',
    #     '475', '1470', '3421', '3465', '3470', '3498', '4428', '36056', '454',
    #     '1496', '2444', '4414', '4415', '4435', '62722', '76183', '464', '465',
    #     'cp'
    #     ]

    out_dir = Path(r'test_asymm23_dis_18__30')

    noise_add_flag = True
    noise_add_flag = False
    noise_magnitude = 1e-3
    #==========================================================================

    #==========================================================================
    # Hourly ppt from Prof.
    #==========================================================================

    # in_data_file = Path(r'BW_dwd_stns_60min_1995_2020_data.csv')
    #
    # beg_time = '2010-01-01 00:00:00'
    # end_time = '2013-12-31 23:00:00'
    #
    # sep = ';'
    #
    # out_dir = Path(r'iaaft_ppt_02_no_cps_margs_only_hourly')

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
    # cols = 'P13698;P07331;P13672'.split(';')
    #
    # noise_add_flag = True
    # # noise_add_flag = False
    # noise_magnitude = 1e-4
    #==========================================================================

    #==========================================================================
    # Hourly ppt DWD, mine.
    #==========================================================================

    # in_data_file = Path(r'neckar_1hr_ppt_data_20km_buff_Y2004_2020_10cps.pkl')
    #
    # sep = ';'
    #
    # beg_time = '2009-01-01'
    # end_time = '2009-12-31'
    #
    # out_dir = Path(r'test_spcorr_ppt_04')
    #
    # cols = ['P1176']  # , 'P1290' , 'P13674' , 'P13698', 'P1937', 'P2159', 'P2292']  # , 'cp']
    #
    # noise_add_flag = True
    # noise_add_flag = False
    # noise_magnitude = 1e-4
    #==========================================================================

    #==========================================================================
    # Daily ppt.
    #==========================================================================

    # in_data_file = Path(r'precipitation_bw_1961_2015_10cps.csv')
    #
    # sep = ';'
    #
    # beg_time = '1961-01-01'
    # # end_time = '2015-12-31'
    # end_time = '1965-12-31'
    #
    # out_dir = Path(r'test_spcorr_ppt_19')
    #
    # cols = ['P1162', 'P1197']  # , 'cp']
    #
    # noise_add_flag = True
    # noise_add_flag = False
    # noise_magnitude = 1e-3
    #==========================================================================

    #==========================================================================
    # Daily temperature.
    #==========================================================================

    # in_data_file = Path(r'daily_neckar_tg_Y1961_2015.pkl')
    #
    # sep = ';'
    #
    # beg_time = '1961-01-01'
    # # end_time = '2015-12-31'
    # end_time = '1965-12-31'
    #
    # cols = ['TG1197']  # , 'TG3257', 'TG330']  # , 'TG3402', 'TG3621']
    #
    # # All valid from 1961 to 2015.
    # # cols = [
    # #     'TG1197', 'TG3257', 'TG330', 'TG3402', 'TG3621', 'TG3761', 'TG4287',
    # #     'TG4887', 'TG4931', 'TG5229', 'TG5664']
    #
    # # All valid values from 1961 to 1965.
    # # cols = [
    # #     'TG1018', 'TG1093', 'TG1197', 'TG1254', 'TG1468', 'TG1875', 'TG2074',
    # #     'TG2095', 'TG2349', 'TG257', 'TG2638', 'TG2654', 'TG268', 'TG2775',
    # #     'TG2814', 'TG2879', 'TG2949', 'TG3135', 'TG3257', 'TG330', 'TG3402',
    # #     'TG3425', 'TG3432', 'TG3486', 'TG3621', 'TG3671', 'TG3761', 'TG3924',
    # #     'TG4287', 'TG4300', 'TG4330', 'TG4581', 'TG4703', 'TG4887', 'TG4927',
    # #     'TG4928', 'TG4931', 'TG4933', 'TG5105', 'TG5120', 'TG5155', 'TG5229',
    # #     'TG5429', 'TG5559', 'TG5654', 'TG5664', 'TG5885', 'TG755', 'TG772',
    # #     'TG881']
    #
    # out_dir = Path(r'test_spcorr_tg_21')
    #
    # noise_add_flag = True
    # noise_add_flag = False
    # noise_magnitude = 1e-3
    #==========================================================================

    #==========================================================================
    #    Daily HBV sim
    #==========================================================================
    # in_data_file = Path(r'hbv_sim__1963_2015_2.csv')
    #
    # cols = 'prec;pet;temp;q_sim'.split(';')
    #
    # sep = ';'
    #
    # beg_time = '1996-01-01'
    # end_time = '2000-12-31'
    #
    # out_dir = Path(r'holy_grail_2_04')
    #
    # noise_add_flag = True
    # noise_add_flag = False
    # noise_magnitude = 1e-3
    #==========================================================================

    n_cpus = 8
    n_sims = n_cpus * 4

    ratio_a = 1.0  # For marginals.
    ratio_b = 5.0  # For ranks.

    auto_spec_flag = True
    cross_spec_flag = True

    # auto_spec_flag = False
    cross_spec_flag = False

    take_best_flag = True
    # take_best_flag = False

    asymmetrize_flag = True
    # asymmetrize_flag = False

    # All coefficients with periods longer than and equal to this are kept.
    keep_period = None
    keep_period = 180

    quant_levels = np.arange(5, 101, 5)
    shift_exps = np.linspace(1, 10, 20)
    max_shifts = np.arange(1, 4)
    pre_vals_ratios = np.linspace(0.2, 1.0, 9)

    # quant_levels = np.arange(60, 81, 5)
    # shift_exps = np.linspace(6, 8, 11)
    # max_shifts = np.arange(1, 3)
    # pre_vals_ratios = np.linspace(0.8, 1.0, 9)

    # Column with the name "ref_lab" should not be in cols.
    ref_lab = 'ref'
    sim_lab = 'S'  # Put infront of each simulation number.

    n_repeat = int(len(cols) * 10)

    float_fmt = '%0.2f'

    show_corrs_flag = False
    max_corr_to_show = 6
    #==========================================================================

    # assert n_sims > 0, n_sims
    assert n_cpus > 0, n_cpus

    assert ref_lab not in cols, cols

    out_dir /= 'sim_files'

    out_dir.mkdir(exist_ok=True, parents=True)

    if in_data_file.suffix == '.csv':
        df_data = pd.read_csv(in_data_file, sep=sep, index_col=0)

    elif in_data_file.suffix == '.pkl':
        df_data = pd.read_pickle(in_data_file)

    else:
        raise NotImplementedError(
            f'Unknown extension of in_data_file: '
            f'{in_data_file.suffix}!')

    df_data = df_data.loc[beg_time:end_time, cols]

    if df_data.shape[0] % 2:
        df_data = df_data.iloc[:-1,:]

    assert np.all(np.isfinite(df_data.values))

    assert ref_lab not in df_data.columns

    if cross_spec_flag:
        assert df_data.shape[1] >= 2

    assert np.unique(df_data.columns).size == df_data.shape[1]
    assert np.unique(df_data.index).size == df_data.shape[0]
    #==========================================================================

    if False:
        data_probs = (
            rankdata(df_data.values, axis=0) / (df_data.shape[0] + 1.0))

        data_expon = expon.ppf(data_probs, loc=50, scale=10)

        df_data[:] = data_expon

    if False:
        data_probs = (
            rankdata(df_data.values, axis=0) / (df_data.shape[0] + 1.0))

        data_norm = norm.ppf(data_probs)

        df_data[:] = data_norm
    #==========================================================================

    df_data.to_csv(
        out_dir / f'cross_sims_{ref_lab}.csv', sep=sep, float_format=float_fmt)

    data = df_data.values.copy()

    time_index = df_data.index.copy()
    #==========================================================================

    ref_data_cls = get_ts_data_cls(
        data, noise_add_flag, noise_magnitude, keep_period)

    if asymmetrize_flag:
        asymm_23_propss = product(
            quant_levels,
            shift_exps,
            max_shifts,
            pre_vals_ratios)

        args_gen = (
            (ref_data_cls,
             cols,
             ratio_a,
             ratio_b,
             auto_spec_flag,
             cross_spec_flag,
             n_repeat,
             sim_idx,
             # n_sims,
             sim_lab,
             take_best_flag,
             asymm23_props,
             asymmetrize_flag,
            )
            for sim_idx, asymm23_props in enumerate(asymm_23_propss))

    else:
        args_gen = (
            (ref_data_cls,
             cols,
             ratio_a,
             ratio_b,
             auto_spec_flag,
             cross_spec_flag,
             n_repeat,
             sim_idx,
             # n_sims,
             sim_lab,
             take_best_flag,
             None,
             asymmetrize_flag,
            )
            for sim_idx in range(n_sims))

        n_cpus = min(n_sims, n_cpus)
    #==========================================================================

    if n_cpus == 1:
        ress = []
        for args in args_gen:
            sims = get_sim_dict(args)
            ress.append(sims)

    else:
        mp_pool = ProcessPool(n_cpus)

        ress = list(mp_pool.imap(get_sim_dict, args_gen, chunksize=1))

        mp_pool.close()
        mp_pool.join()
    #==========================================================================

    all_sims = {
        cols[k]: {ref_lab: data[:, k].copy()} for k in range(len(cols))}

    all_order_sdiffs = {}
    all_obj_vals = {}
    all_asymm_props = {}

    sim_labs = []
    for sims, order_sdiffs, obj_vals, asymm_props in ress:
        for col in cols:
            all_sims[col].update(sims[col])

            sim_labs.extend(list(sims[col].keys()))

        all_order_sdiffs.update(order_sdiffs)
        all_obj_vals.update(obj_vals)

        all_asymm_props.update(asymm_props)

    ress = sims = order_sdiffs = obj_vals = None
    sim_labs = tuple(sim_labs)
    #==========================================================================

    print('')

    for col in cols:
        col_df = pd.DataFrame(all_sims[col], index=time_index)

        col_df.to_csv(
            out_dir / f'auto_sims_{col}.csv', sep=sep, float_format=float_fmt)

        if show_corrs_flag:
            print(f'ref_sim_pcorrs ({col}):')

            print(col_df.corr(method='pearson').round(3
                  ).values[:max_corr_to_show,:max_corr_to_show])

            print('')

            print(f'ref_sim_scorrs ({col}):')

            print(col_df.corr(method='spearman').round(3
                  ).values[:max_corr_to_show,:max_corr_to_show])

            print('')
    #==========================================================================

    if show_corrs_flag:
        print(f'{ref_lab}_{ref_lab}_pcorrs:')

        print(df_data.corr(method='pearson').round(3
              ).values[:max_corr_to_show,:max_corr_to_show])

        print('')

        print(f'{ref_lab}_{ref_lab}_scorrs:')

        print(df_data.corr(method='spearman').round(3
              ).values[:max_corr_to_show,:max_corr_to_show])

        print('')

    for sim_lab in sim_labs:
        sim_df = pd.DataFrame(index=time_index, columns=cols, dtype=float)

        for col in cols:
            sim_df.loc[:, col] = all_sims[col][sim_lab]

        sim_df.to_csv(
            out_dir / f'cross_sims_{sim_lab}.csv',
            sep=sep,
            float_format=float_fmt)

        if show_corrs_flag:
            print(f'sim_sim_pcorrs ({sim_lab}):')

            print(sim_df.corr(method='pearson').round(3
                  ).values[:max_corr_to_show,:max_corr_to_show])

            print('')

            print(f'sim_sim_scorrs ({sim_lab}):')

            print(sim_df.corr(method='spearman').round(3
                  ).values[:max_corr_to_show,:max_corr_to_show])
            print('')
    #==========================================================================

    pd.DataFrame(all_order_sdiffs).to_csv(
        out_dir / 'all_order_sdiffs.csv', sep=sep)

    pd.DataFrame(all_obj_vals).to_csv(
        out_dir / 'all_obj_vals.csv', sep=sep)

    pd.DataFrame(
        all_asymm_props,
        index=['n_levels', 'max_shift_exp', 'max_shift', 'pre_vals_ratio']
        ).to_csv(out_dir / 'all_asymm_props.csv', sep=sep)
    #==========================================================================
    return


def get_sim_dict(args):

    (ref_data_cls,
     cols,
     ratio_a,
     ratio_b,
     auto_spec_flag,
     cross_spec_flag,
     n_repeat,
     sim_idx,
     # n_sims,
     sim_lab,
     take_best_flag,
     asymm23_props,
     asymmetrize_flag,
    ) = args
    #==========================================================================

    assert any([ratio_a, ratio_b])
    assert any([auto_spec_flag, cross_spec_flag])
    #==========================================================================

    data = ref_data_cls.data

    if cross_spec_flag:
        assert data.shape[1] >= 2

    # ref_ft = ref_data_cls.ft
    ref_phs = ref_data_cls.phss
    ref_mag = ref_data_cls.mags

    # ref_ft_ranks = ref_data_cls.ft_ranks
    ref_phs_ranks = ref_data_cls.phss_ranks
    ref_mag_ranks = ref_data_cls.mags_ranks

    data_sort = ref_data_cls.data_sort

    keep_period_flags = ref_data_cls.keep_period_flags
    #==========================================================================

    sim_zeros_str = 5
    #==========================================================================

    data_rand = np.empty_like(data)
    order_old = np.empty(data_sort.shape, dtype=int)
    for k in range(len(cols)):
        order_old[:, k] = np.argsort(np.argsort(
            np.random.random(data_sort.shape[0])))

        data_rand[:, k] = data_sort[order_old[:, k], k]

    order_old_ranks = rankdata(order_old)
    #==========================================================================

    # For the cross case only.
    obj_val_global_min = get_obj_val(
        ref_data_cls, data_rand, ratio_a, ratio_b)

    data_best = data_rand.copy()
    i_data_best = -1

    order_sdiffs = np.full(n_repeat, np.nan)
    obj_vals = order_sdiffs.copy()

    i_repeat = 0
    order_sdiff = 0.0

    adjust_mag_diffs_flag = True
    adjust_mag_diffs_flag = False

    stn_ctr = 0
    for i_repeat in range(n_repeat):

        if asymmetrize_flag:
            asymmetrize_data(data_rand, asymm23_props)
        #======================================================================

        if ratio_a:
            sim_ft_margs = np.fft.rfft(data_rand, axis=0)

            if i_repeat and adjust_mag_diffs_flag:
                assert ref_mag.shape[1] == 1

                ref_mag_adj = (2 * ref_mag) - np.abs(sim_ft_margs)

                neg_mag_idxs = ref_mag_adj < 0
                pos_mag_idxs = ref_mag_adj > 0

                neg_mag = ref_mag_adj[neg_mag_idxs].copy() * -1
                pos_mag = ref_mag_adj[pos_mag_idxs].copy()

                pos_mag_ratios = pos_mag / pos_mag.sum()

                ref_mag_adj[neg_mag_idxs] = 0
                ref_mag_adj[pos_mag_idxs] += neg_mag.sum() * pos_mag_ratios

            else:
                ref_mag_adj = ref_mag.copy()

        if ratio_b:
            sim_ft_ranks = np.fft.rfft(rankdata(data_rand, axis=0), axis=0)

            if i_repeat and adjust_mag_diffs_flag:
                assert ref_mag_ranks.shape[1] == 1

                ref_mag_ranks_adj = (2 * ref_mag_ranks) - np.abs(sim_ft_ranks)

                neg_mag_idxs = ref_mag_ranks_adj < 0
                pos_mag_idxs = ref_mag_ranks_adj > 0

                neg_mag = ref_mag_ranks_adj[neg_mag_idxs].copy() * -1
                pos_mag = ref_mag_ranks_adj[pos_mag_idxs].copy()

                pos_mag_ratios = pos_mag / pos_mag.sum()

                ref_mag_ranks_adj[neg_mag_idxs] = 0
                ref_mag_ranks_adj[pos_mag_idxs] += neg_mag.sum() * pos_mag_ratios

            else:
                ref_mag_ranks_adj = ref_mag_ranks.copy()

        # Marginals auto.
        if ratio_a and auto_spec_flag:
            sim_phs_margs = np.angle(sim_ft_margs)

            if keep_period_flags is not None:
                sim_phs_margs[keep_period_flags,:] = (
                    ref_phs[keep_period_flags,:])

            sim_ft_new = np.empty_like(sim_ft_margs)

            sim_ft_new.real[:] = np.cos(sim_phs_margs) * ref_mag_adj
            sim_ft_new.imag[:] = np.sin(sim_phs_margs) * ref_mag_adj

            sim_ft_new[0,:] = 0

            sim_ift_a_auto = np.fft.irfft(sim_ft_new, axis=0)

            order_new_a = np.empty_like(order_old)
            for k in range(len(cols)):
                order_new_a[:, k] = np.argsort(
                    np.argsort(sim_ift_a_auto[:, k]))

            for k in range(len(cols)):
                sim_ift_a_auto[:, k] = data_sort[order_new_a[:, k], k]

            sim_ift_a_auto -= sim_ift_a_auto.mean(axis=0)
            sim_ift_a_auto /= sim_ift_a_auto.std(axis=0)

        else:
            sim_ift_a_auto = 0.0

        # Ranks auto.
        if ratio_b and auto_spec_flag:
            sim_phs = np.angle(sim_ft_ranks)

            if keep_period_flags is not None:
                sim_phs[keep_period_flags,:] = (
                    ref_phs_ranks[keep_period_flags,:])

            sim_ft_new = np.empty_like(sim_ft_ranks)

            sim_ft_new.real[:] = np.cos(sim_phs) * ref_mag_ranks_adj
            sim_ft_new.imag[:] = np.sin(sim_phs) * ref_mag_ranks_adj

            sim_ft_new[0,:] = 0

            sim_ift_b_auto = np.fft.irfft(sim_ft_new, axis=0)
            sim_ift_b_auto = rankdata(sim_ift_b_auto, axis=0)
            sim_ift_b_auto -= sim_ift_b_auto.mean(axis=0)
            sim_ift_b_auto /= sim_ift_b_auto.std(axis=0)

        else:
            sim_ift_b_auto = 0.0
        #==================================================================

        # Marginals cross.
        if ratio_a and cross_spec_flag:
            # sim_mag = np.abs(sim_ft_margs)
            # sim_mag = ref_mag.copy()
            sim_mag = ref_mag_adj.copy()

            sim_phs = (
                np.angle(sim_ft_margs[:, [stn_ctr]]) +
                ref_phs -
                ref_phs[:, [stn_ctr]])

            sim_phs[0,:] = ref_phs[0,:]

            if keep_period_flags is not None:
                sim_phs[keep_period_flags,:] = ref_phs[keep_period_flags,:]
                # sim_mag[keep_period_flags,:] = ref_mag[keep_period_flags,:]

            sim_ft_new = np.empty_like(sim_ft_margs)

            sim_ft_new.real[:] = np.cos(sim_phs) * sim_mag
            sim_ft_new.imag[:] = np.sin(sim_phs) * sim_mag

            sim_ft_new[0,:] = 0

            sim_ift_a_cross = np.fft.irfft(sim_ft_new, axis=0)

            order_new_a = np.empty_like(order_old)
            for k in range(len(cols)):
                order_new_a[:, k] = np.argsort(
                    np.argsort(sim_ift_a_cross[:, k]))

            for k in range(len(cols)):
                sim_ift_a_cross[:, k] = data_sort[order_new_a[:, k], k]

            sim_ift_a_cross -= sim_ift_a_cross.mean(axis=0)
            sim_ift_a_cross /= sim_ift_a_cross.std(axis=0)

        else:
            sim_ift_a_cross = 0.0

        # Ranks cross.
        if ratio_b and cross_spec_flag:
            # sim_mag = np.abs(sim_ft_ranks)
            # sim_mag = ref_mag_ranks.copy()
            sim_mag = ref_mag_ranks_adj.copy()

            sim_phs = (
                np.angle(sim_ft_ranks[:, [stn_ctr]]) +
                ref_phs_ranks -
                ref_phs_ranks[:, [stn_ctr]])

            sim_phs[0,:] = ref_phs_ranks[0,:]

            if keep_period_flags is not None:
                sim_phs[keep_period_flags,:] = (
                    ref_phs_ranks[keep_period_flags,:])

                # sim_mag[keep_period_flags,:] = ref_mag_ranks[
                #     keep_period_flags,:]

            sim_ft_new = np.empty_like(sim_ft_ranks)

            sim_ft_new.real[:] = np.cos(sim_phs) * sim_mag
            sim_ft_new.imag[:] = np.sin(sim_phs) * sim_mag

            sim_ft_new[0,:] = 0

            sim_ift_b_cross = np.fft.irfft(sim_ft_new, axis=0)
            sim_ift_b_cross = rankdata(sim_ift_b_cross, axis=0)
            sim_ift_b_cross -= sim_ift_b_cross.mean(axis=0)
            sim_ift_b_cross /= sim_ift_b_cross.std(axis=0)

        else:
            sim_ift_b_cross = 0.0
        #======================================================================

        # Their sum.
        sim_ift = (
            (ratio_a * sim_ift_a_auto) +
            (ratio_b * sim_ift_b_auto) +
            (ratio_a * sim_ift_a_cross) +
            (ratio_b * sim_ift_b_cross)
            )

        order_new = np.empty_like(order_old)
        for k in range(len(cols)):
            order_new[:, k] = np.argsort(np.argsort(sim_ift[:, k]))

        order_new_ranks = rankdata(order_new)

        # order_sdiff = (
        #     (order_old.astype(float) - order_new.astype(float)) ** 2).sum()

        order_sdiff = 1 - np.corrcoef(order_old_ranks, order_new_ranks)[0, 1]

        # Casting may create problems.
        assert order_sdiff >= 0, order_sdiff

        order_sdiffs[i_repeat] = order_sdiff

        if np.isclose(order_sdiff, 0.0):
            # Nothing changed.
            break

        order_old = order_new
        order_old_ranks = order_new_ranks

        data_rand = np.empty_like(data)
        for k in range(len(cols)):
            data_rand[:, k] = data_sort[order_old[:, k], k]
        #======================================================================

        stn_ctr += 1
        if stn_ctr == data_rand.shape[1]:
            stn_ctr = 0
        #======================================================================

        obj_val = get_obj_val(ref_data_cls, data_rand, ratio_a, ratio_b)

        obj_vals[i_repeat] = obj_val

        if obj_val < obj_val_global_min:
            obj_val_global_min = obj_val

            data_best = data_rand.copy()
            i_data_best = i_repeat
        #======================================================================
    #==========================================================================

    sims = {cols[k]: {} for k in range(len(cols))}
    key = f'{sim_lab}{sim_idx:0{sim_zeros_str}d}'
    for k, col in enumerate(cols):
        if not take_best_flag:
            value = data_rand[:, k]

        else:
            value = data_best[:, k]

        value = data_rand[:, k]

        sims[col][key] = value

    print(
        f'Done with sim_idx: {sim_idx} with i_repeat: {i_repeat} and '
        f'with i_data_best: {i_data_best} and '
        f'with order_sdiff: {order_sdiff:0.3f}')

    order_sdiffs = {key: order_sdiffs}
    obj_vals = {key: obj_vals}

    asymm_props = {key: np.array(asymm23_props)}

    return sims, order_sdiffs, obj_vals, asymm_props


def asymmetrize_data(sim_data, asymm23_props):

    n_levels, max_shift_exp, max_shift, pre_vals_ratio = asymm23_props

    for i in range(sim_data.shape[1]):

        vals = sim_data[:, i].copy()

        vals_sort = np.sort(vals)

        probs = rankdata(vals) / (vals.size + 1.0)

        levels = (probs * n_levels).astype(int)

        asymm_vals = vals.copy()
        for level in range(n_levels):
            asymm_vals_i = asymm_vals.copy()

            max_shift_level = (
                max_shift -
                (max_shift * ((level / n_levels) ** max_shift_exp)))

            max_shift_level = round(max_shift_level)
            max_shift_level = int(max_shift_level)

            for shift in range(1, max_shift_level + 1):
                asymm_vals_i = np.roll(asymm_vals_i, shift)

                idxs_to_shift = levels <= level

                asymm_vals[idxs_to_shift] = (
                    asymm_vals[idxs_to_shift] * pre_vals_ratio +
                    asymm_vals_i[idxs_to_shift] * (1 - pre_vals_ratio))

        asymm_vals += (
            (-1e-6) + (2e-6) * np.random.random(size=asymm_vals.size))

        asymm_vals = vals_sort[np.argsort(np.argsort(asymm_vals))]

        sim_data[:, i] = asymm_vals

    return


class Data:

    def __init__(self):

        return


def get_ts_data_cls(data, noise_add_flag, noise_magnitude, keep_period):

    assert not (data.shape[0] % 2), data.shape[0]

    data_cls = Data()
    #==========================================================================

    if noise_add_flag:
        data_noise = np.random.random(size=(data.shape[0], 1))

        data += data_noise * noise_magnitude

    data_cls.data = data
    #==========================================================================

    ft = np.fft.rfft(data, axis=0)

    ft[0,:] = 0

    phss = np.angle(ft)
    mags = np.abs(ft)

    data_cls.ft = ft
    data_cls.phss = phss
    data_cls.mags = mags
    #==========================================================================

    ft_ranks = np.fft.rfft(rankdata(data, axis=0), axis=0)

    ft_ranks[0,:] = 0

    phss_ranks = np.angle(ft_ranks)
    mags_ranks = np.abs(ft_ranks)

    data_cls.ft_ranks = ft_ranks
    data_cls.phss_ranks = phss_ranks
    data_cls.mags_ranks = mags_ranks
    #==========================================================================

    # phss_diffs = phss - phss[:, [0]]
    # phss_ranks_diffs = phss_ranks - phss_ranks[:, [0]]
    #
    # data_cls.phss_diffs = phss_diffs
    # data_cls.phss_ranks_diffs = phss_ranks_diffs
    #==========================================================================

    data_sort = np.sort(data, axis=0)

    data_cls.data_sort = data_sort
    #==========================================================================

    probs = rankdata(data, axis=0) / (data.shape[0] + 1.0)
    probs = probs.copy(order='f')

    data_cls.pcorrs_cross = np.corrcoef(data, rowvar=False)
    data_cls.scorrs_cross = np.corrcoef(probs, rowvar=False)
    #==========================================================================

    scorrs_auto, asymms_1_auto, asymms_2_auto, etpys_auto, pcorrs_auto = (
        get_corrs_asymms_ecop_auto(data, np.arange(1, 31, dtype=np.int64)))

    data_cls.scorrs_auto = scorrs_auto

    data_cls.asymms_1_auto = asymms_1_auto
    data_cls.asymms_1_auto_obj_idxs = np.arange(7, 20)

    data_cls.asymms_2_auto = asymms_2_auto
    data_cls.asymms_2_auto_obj_idxs = np.arange(0, 16)

    data_cls.etpys_auto = etpys_auto
    data_cls.etpys_auto_obj_idxs = np.arange(0, 11)

    data_cls.pcorrs_auto = pcorrs_auto
    #==========================================================================

    asymms_1s_cross, asymms_2s_cross = get_asymms_ecop_cross(
        probs, data_cls.scorrs_cross)

    data_cls.asymms_1s_cross = asymms_1s_cross
    data_cls.asymms_2s_cross = asymms_2s_cross
    #==========================================================================

    if keep_period is not None:
        periods = (ft.shape[0] * 2) / (np.arange(1.0, ft.shape[0] + 1.0))

        keep_period_flags = periods >= keep_period

        assert periods.size == ft.shape[0], (periods.size, ft.shape[0])

        assert periods[+1] == ft.shape[0], periods[+1]
        assert periods[-1] == 2, periods[-1]

    else:
        keep_period_flags = None

    data_cls.keep_period_flags = keep_period_flags
    #==========================================================================

    return data_cls


def get_obj_val(ref_data_cls, sim_data, ratio_a, ratio_b):

    obj_val = 0.0

    if sim_data.shape[0] > 1:
        cross_wt = ref_data_cls.pcorrs_auto.size / sim_data.shape[0]

        pcorrs_cross = np.corrcoef(sim_data, rowvar=False)

        obj_val += ratio_a * cross_wt * (
            (ref_data_cls.pcorrs_cross - pcorrs_cross) ** 2).sum()

        # pcorrs auto in ratio_b part.
    #==========================================================================

    probs = rankdata(sim_data, axis=0) / (sim_data.shape[0] + 1.0)
    probs = probs.copy(order='f')

    if sim_data.shape[0] > 1:
        scorrs_cross = np.corrcoef(probs, rowvar=False)

        asymms_1s_cross, asymms_2s_cross = get_asymms_ecop_cross(
            probs, scorrs_cross)

        obj_val += ratio_b * cross_wt * (
            (ref_data_cls.scorrs_cross - scorrs_cross) ** 2).sum()

        obj_val += ratio_b * cross_wt * (
            (ref_data_cls.asymms_1s_cross - asymms_1s_cross) ** 2).sum()

        obj_val += ratio_b * cross_wt * (
            (ref_data_cls.asymms_2s_cross - asymms_2s_cross) ** 2).sum()
    #==========================================================================

    scorrs_auto, asymms_1_auto, asymms_2_auto, etpys_auto, pcorrs_auto = (
        get_corrs_asymms_ecop_auto(
            sim_data, np.arange(1, 31, dtype=np.int64)))

    obj_val += ratio_b * (
        (ref_data_cls.scorrs_auto - scorrs_auto) ** 2).sum()

    obj_val += ratio_b * (
        (ref_data_cls.asymms_1_auto - asymms_1_auto) ** 2)[
            ref_data_cls.asymms_1_auto_obj_idxs].sum()

    obj_val += ratio_b * (
        (ref_data_cls.asymms_2_auto - asymms_2_auto) ** 2)[
            ref_data_cls.asymms_2_auto_obj_idxs].sum()

    obj_val += ratio_b * (
        (ref_data_cls.etpys_auto - etpys_auto) ** 2)[
            ref_data_cls.etpys_auto_obj_idxs].sum()

    obj_val += ratio_b * (
        (ref_data_cls.pcorrs_auto - pcorrs_auto) ** 2).sum()

    return obj_val


def get_asymms_ecop_cross(probs, scorrs):

    combs = combinations(list(range(probs.shape[1])), 2)

    asymms_1s = np.zeros_like(scorrs)
    asymms_2s = asymms_1s.copy()

    for comb in combs:
        scorr = scorrs[comb[0], comb[1]]

        asymm_1, asymm_2 = get_asymms_sample(
            probs[:, comb[0]], probs[:, comb[1]])

        asymm_1 /= get_asymm_1_max(scorr)
        asymm_2 /= get_asymm_2_max(scorr)

        asymms_1s[comb[0], comb[1]] = asymm_1
        asymms_2s[comb[0], comb[1]] = asymm_2

        asymms_1s[comb[1], comb[0]] = asymm_1
        asymms_2s[comb[1], comb[0]] = asymm_2

    return asymms_1s, asymms_2s


def get_corrs_asymms_ecop_auto(data, lag_steps):

    scorrs = []
    asymms_1 = []
    asymms_2 = []
    etpys = []
    pcorrs = []

    ecop_bins = 30

    etpy_min = get_etpy_min(ecop_bins)
    etpy_max = get_etpy_max(ecop_bins)

    ecop_dens_arrs = np.full((ecop_bins, ecop_bins), np.nan, dtype=np.float64)

    for col_i in range(data.shape[1]):

        for lag_step in lag_steps:
            probs_i, rolled_probs_i = roll_real_2arrs_with_nan(
                data[:, col_i], data[:, col_i], lag_step, True)

            data_i, rolled_data_i = roll_real_2arrs_with_nan(
                data[:, col_i], data[:, col_i], lag_step, False)

            # scorr.
            scorr = np.corrcoef(probs_i, rolled_probs_i)[0, 1]
            scorrs.append(scorr)

            # asymms.
            asymm_1, asymm_2 = get_asymms_sample(probs_i, rolled_probs_i)

            asymm_1 /= get_asymm_1_max(scorr)

            asymm_2 /= get_asymm_2_max(scorr)

            asymms_1.append(asymm_1)
            asymms_2.append(asymm_2)

            # ecop etpy.
            fill_bi_var_cop_dens(probs_i, rolled_probs_i, ecop_dens_arrs)

            non_zero_idxs = ecop_dens_arrs > 0

            dens = ecop_dens_arrs[non_zero_idxs]

            etpy_arr = -(dens * np.log(dens))

            etpy = etpy_arr.sum()

            etpy = (etpy - etpy_min) / (etpy_max - etpy_min)

            etpys.append(etpy)

            # pcorr.
            pcorr = np.corrcoef(data_i, rolled_data_i)[0, 1]
            pcorrs.append(pcorr)

    scorrs = np.array(scorrs)
    asymms_1 = np.array(asymms_1)
    asymms_2 = np.array(asymms_2)
    etpys = np.array(etpys)
    pcorrs = np.array(pcorrs)
    return scorrs, asymms_1, asymms_2, etpys, pcorrs


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
