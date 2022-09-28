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

import numpy as np
import pandas as pd
from scipy.stats import rankdata, expon, norm
import matplotlib.pyplot as plt; plt.ioff()
from pathos.multiprocessing import ProcessPool

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
    end_time = '1963-07-31'

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

    out_dir = Path(r'test_wk_33')

    # noise_add_flag = True
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
    n_sims = n_cpus * 20

    auto_spec_flag = True
    cross_spec_flag = True

    # auto_spec_flag = False
    cross_spec_flag = False

    # All coefficients with periods longer than and equal to this are kept.
    keep_period = None
    keep_period = 180

    marg_ratio_bds = np.array([0.0, 1.0])

    # Column with the name "ref_lab" should not be in cols.
    ref_lab = 'ref'
    sim_lab = 'S'  # Put infront of each simulation number.

    n_repeat = int(len(cols) * 100)

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

    data = df_data.values.copy()

    time_index = df_data.index.copy()
    #==========================================================================

    ref_data_cls = get_ts_data_cls(
        data, noise_add_flag, noise_magnitude, keep_period)
    #==========================================================================

    data_sort = ref_data_cls.data_sort.copy()
    data_rand = np.empty_like(data, order='f')
    data_probs = data_rand.copy(order='f')
    order_old = np.empty(data_sort.shape, dtype=int)
    for k in range(len(cols)):
        order_old[:, k] = np.argsort(np.argsort(
            np.random.random(data_sort.shape[0])))

        data_rand[:, k] = data_sort[order_old[:, k], k]
    #==========================================================================

    # n_cols = len(cols)

    marg_ratios = (
            marg_ratio_bds[0] +
            ((marg_ratio_bds[1] - marg_ratio_bds[0]) *
             np.random.random(size=(n_sims, 2))))

    marg_ratios[:, 1] = 1 - marg_ratios[:, 0]
    #==========================================================================

    args_gen = (
        (ref_data_cls,
         cols,
         marg_ratios[sim_idx,:],
         auto_spec_flag,
         cross_spec_flag,
         n_repeat,
         sim_idx,
         sim_lab,
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

    sim_labs = []
    for sims, order_sdiffs in ress:
        for col in cols:
            all_sims[col].update(sims[col])

            sim_labs.extend(list(sims[col].keys()))

        all_order_sdiffs.update(order_sdiffs)

    ress = sims = order_sdiffs = None
    sim_labs = tuple(sim_labs)
    #==========================================================================

    print('')

    df_data.to_csv(
        out_dir / f'cross_sims_{ref_lab}.csv', sep=sep, float_format=float_fmt)

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
    #==========================================================================
    return


def get_sim_dict(args):

    (ref_data_cls,
     cols,
     marg_ratios,
     auto_spec_flag,
     cross_spec_flag,
     n_repeat,
     sim_idx,
     sim_lab,
    ) = args

    #==========================================================================

    ratio_a, ratio_b = marg_ratios[0], marg_ratios[1]

    assert any([ratio_a, ratio_b])

    assert any([auto_spec_flag, cross_spec_flag])
    #==========================================================================

    for variable in dir(ref_data_cls):
        if not isinstance(getattr(ref_data_cls, variable), np.ndarray):
            continue

        else:
            getattr(ref_data_cls, variable).flags.writeable = False
    #==========================================================================

    data = ref_data_cls.data
    # probs = ref_data_cls.probs

    if cross_spec_flag:
        assert data.shape[1] >= 2

    # ref_ft = ref_data_cls.ft
    ref_phs = ref_data_cls.phss
    ref_mag = ref_data_cls.mags

    # ref_ft_ranks = ref_data_cls.ft_ranks
    ref_phs_ranks = ref_data_cls.phss_ranks
    ref_mag_ranks = ref_data_cls.mags_ranks

    data_sort = ref_data_cls.data_sort
    probs_sort = ref_data_cls.probs_sort

    keep_period_flags = ref_data_cls.keep_period_flags
    #==========================================================================

    sim_zeros_str = 5
    #==========================================================================

    data_rand = np.empty_like(data, order='f')
    probs_rand = np.empty_like(data, order='f')
    order_old = np.empty(data_sort.shape, dtype=int)
    for k in range(len(cols)):
        order_old[:, k] = np.argsort(np.argsort(
            np.random.random(data_sort.shape[0])))

        data_rand[:, k] = data_sort[order_old[:, k], k]
        probs_rand[:, k] = probs_sort[order_old[:, k], k]

    order_old_ranks = rankdata(order_old)
    #==========================================================================

    ref_pwr = ref_mag ** 2
    ref_pwr[0] = 0

    ref_pcorrs = np.fft.irfft(ref_pwr, axis=0)
    ref_pcorrs /= ref_pcorrs[0]

    ref_pcorrs = np.concatenate(
        (ref_pcorrs, ref_pcorrs[[0],:]), axis=0)

    ref_pwr = ref_mag_ranks ** 2
    ref_pwr[0] = 0

    ref_scorrs = np.fft.irfft(ref_pwr, axis=0)
    ref_scorrs /= ref_scorrs[0]

    ref_scorrs = np.concatenate(
        (ref_scorrs, ref_scorrs[[0],:]), axis=0)

    spec_crctn_cnst = 2.0
    #==========================================================================

    order_sdiffs = np.full(n_repeat, np.nan)

    i_repeat = 0
    order_sdiff = 0.0

    # ref_mag_srtd = np.sort(ref_mag, axis=0)

    spec_norm_vals_data = ref_mag[1:,:].sum(axis=0)
    spec_norm_vals_ranks = ref_mag_ranks[1:,:].sum(axis=0)

    ref_mag_adj = ref_mag.copy()
    ref_mag_ranks_adj = ref_mag_ranks.copy()

    phs_spec_data = ref_phs.copy()
    phs_spec_ranks = ref_phs_ranks.copy()

    adj_iters = 2

    for adj_iter in range(adj_iters):

        stn_ctr = 0
        for i_repeat in range(n_repeat):

            sim_ft_margs = np.fft.rfft(data_rand, axis=0)
            sim_ft_ranks = np.fft.rfft(rankdata(data_rand, axis=0), axis=0)

            sim_phs_margs = np.angle(sim_ft_margs)
            sim_phs_ranks = np.angle(sim_ft_ranks)

            if (adj_iter) and (i_repeat == 0):

                ref_mag_adj = get_adj_mag_spec(
                    sim_ft_margs,
                    spec_crctn_cnst,
                    ref_pcorrs,
                    spec_norm_vals_data)

                phs_spec_data = (2 * ref_phs) - sim_phs_margs

                ref_mag_ranks_adj = get_adj_mag_spec(
                    sim_ft_ranks,
                    spec_crctn_cnst,
                    ref_scorrs,
                    spec_norm_vals_ranks)

                phs_spec_ranks = (2 * ref_phs_ranks) - sim_phs_ranks

            # Marginals auto.
            if ratio_a and auto_spec_flag:
                if keep_period_flags is not None:
                    sim_phs_margs[keep_period_flags,:] = (
                        phs_spec_data[keep_period_flags,:])

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
                if keep_period_flags is not None:
                    sim_phs_ranks[keep_period_flags,:] = (
                        phs_spec_ranks[keep_period_flags,:])

                sim_ft_new = np.empty_like(sim_ft_ranks)

                sim_ft_new.real[:] = np.cos(sim_phs_ranks) * ref_mag_ranks_adj
                sim_ft_new.imag[:] = np.sin(sim_phs_ranks) * ref_mag_ranks_adj

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

                sim_phs = (
                    sim_phs_margs[:, [stn_ctr]] +
                    phs_spec_data -
                    phs_spec_data[:, [stn_ctr]])

                sim_phs[0,:] = phs_spec_data[0,:]

                if keep_period_flags is not None:
                    sim_phs[keep_period_flags,:] = phs_spec_data[
                        keep_period_flags,:]

                sim_ft_new = np.empty_like(sim_ft_margs)

                sim_ft_new.real[:] = np.cos(sim_phs) * ref_mag_adj
                sim_ft_new.imag[:] = np.sin(sim_phs) * ref_mag_adj

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
                sim_phs = (
                    sim_phs_ranks[:, [stn_ctr]] +
                    phs_spec_ranks -
                    phs_spec_ranks[:, [stn_ctr]])

                sim_phs[0,:] = phs_spec_ranks[0,:]

                if keep_period_flags is not None:
                    sim_phs[keep_period_flags,:] = (
                        phs_spec_ranks[keep_period_flags,:])

                sim_ft_new = np.empty_like(sim_ft_ranks)

                sim_ft_new.real[:] = np.cos(sim_phs) * ref_mag_ranks_adj
                sim_ft_new.imag[:] = np.sin(sim_phs) * ref_mag_ranks_adj

                sim_ft_new[0,:] = 0

                sim_ift_b_cross = np.fft.irfft(sim_ft_new, axis=0)
                sim_ift_b_cross = rankdata(sim_ift_b_cross, axis=0)
                sim_ift_b_cross -= sim_ift_b_cross.mean(axis=0)
                sim_ift_b_cross /= sim_ift_b_cross.std(axis=0)

            else:
                sim_ift_b_cross = 0.0
            #==================================================================

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

            order_sdiff = 1 - np.corrcoef(
                order_old_ranks, order_new_ranks)[0, 1]

            # Casting may create problems.
            assert order_sdiff >= 0, order_sdiff

            order_sdiffs[i_repeat] = order_sdiff

            if np.isclose(order_sdiff, 0.0):
                # Nothing changed.
                break

            order_old = order_new
            order_old_ranks = order_new_ranks

            for k in range(len(cols)):
                data_rand[:, k] = data_sort[order_old[:, k], k]
                probs_rand[:, k] = probs_sort[order_old[:, k], k]
            #==================================================================

            stn_ctr += 1
            if stn_ctr == data_rand.shape[1]:
                stn_ctr = 0
            #==================================================================
    #==========================================================================

    sims = {cols[k]: {} for k in range(len(cols))}
    key = f'{sim_lab}{sim_idx:0{sim_zeros_str}d}'
    for k, col in enumerate(cols):
        value = data_rand[:, k]

        sims[col][key] = value

    print(
        f'Done with sim_idx: {sim_idx} with i_repeat: {i_repeat} and '
        f'with order_sdiff: {order_sdiff:0.5f}.')

    order_sdiffs = {key: order_sdiffs}

    return sims, order_sdiffs


def get_adj_mag_spec(ft, spec_crctn_cnst, ref_corrs, norm_vals):

    pwr = np.abs(ft) ** 2
    pwr[0,:] = 0

    corrs = np.fft.irfft(pwr, axis=0)
    corrs /= corrs[0]

    corrs = np.concatenate((corrs, corrs[[0],:]), axis=0)

    if False:
        ref_sim_corrs_diff = spec_crctn_cnst * (ref_corrs - corrs)

        wk_pft = np.fft.rfft(corrs + ref_sim_corrs_diff, axis=0)

    else:
        wk_pft = np.fft.rfft((2 * ref_corrs) - corrs, axis=0)

    pwr_adj = np.abs(wk_pft)
    pwr_adj[0,:] = 0

    if False:
        pwr_adj[1:,:] *= (norm_vals / pwr_adj[1:,:].sum(axis=0))

        mag_adj = pwr_adj ** 0.5

    else:
        mag_adj = pwr_adj ** 0.5

        mag_adj[1:,:] *= (norm_vals / mag_adj[1:,:].sum(axis=0))

    return mag_adj


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

    data_sort = np.sort(data, axis=0)

    data_cls.data_sort = data_sort
    #==========================================================================

    probs = rankdata(data, axis=0) / (data.shape[0] + 1.0)

    probs = probs.copy(order='f')

    probs_sort = np.sort(probs, axis=0)

    data_cls.probs = probs
    data_cls.probs_sort = probs_sort
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
