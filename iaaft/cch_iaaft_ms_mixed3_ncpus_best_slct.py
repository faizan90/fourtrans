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
from scipy.stats import rankdata, norm

DEBUG_FLAG = False


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Testings\fourtrans_practice\iaaft')

    os.chdir(main_dir)

    #==========================================================================
    # Daily discharge.
    #==========================================================================

    in_data_file = Path(r'neckar_q_data_combined_20180713_10cps.csv')

    sep = ';'

    beg_time = '1961-01-01'
    # end_time = '2015-12-31'
    end_time = '1965-12-31'

    cols = ['3470', '3421', '420', ]  # '427', '3465']

    out_dir = Path(r'test_spcorr_71')

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
    # out_dir = Path(r'iaaft_ppt_04_no_cps_ranks_only_hourly')
    #
    # cols = ['P1176', 'P1290' , 'P13674' , 'P13698', 'P1937', 'P2159', 'P2292']  # , 'cp']
    #
    # noise_add_flag = True
    # # noise_add_flag = False
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
    # end_time = '2015-12-31'
    #
    # out_dir = Path(r'iaaft_ppt_12_no_cps_5_1_daily_noise')
    #
    # cols = ['P1162', 'P1197']  # , 'cp']
    #
    # noise_add_flag = True
    # # noise_add_flag = False
    # noise_magnitude = 1e-3
    #==========================================================================

    n_cpus = 8

    n_sims = 8

    ratio_a = 1.0  # For marginals.
    ratio_b = 1.0  # For ranks.

    auto_spec_flag = True
    cross_spec_flag = True

    auto_spec_flag = False
    # cross_spec_flag = False

    # Column with the name "ref_lab" should not be in cols.
    ref_lab = 'ref'
    sim_lab = 'S'  # Put infront of each simulation number.

    n_repeat = len(cols) * 5000
    max_opt_iters = int(1e1)

    float_fmt = '%0.1f'

    show_corrs_flag = False
    max_corr_to_show = 6
    #==========================================================================

    assert n_sims > 0, n_sims
    assert n_cpus > 0, n_cpus

    assert ref_lab not in cols, cols

    out_dir.mkdir(exist_ok=True)

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

    df_data.to_csv(
        out_dir / f'cross_sims_{ref_lab}.csv', sep=sep, float_format=float_fmt)

    data = df_data.values.copy()

    time_index = df_data.index.copy()
    #==========================================================================

    ref_data_cls = get_ts_data_cls(data, noise_add_flag, noise_magnitude)

    args_gen = (
        (ref_data_cls,
         cols,
         ratio_a,
         ratio_b,
         auto_spec_flag,
         cross_spec_flag,
         n_repeat,
         max_opt_iters,
         sim_idx,
         n_sims,
         sim_lab,
        )
        for sim_idx in range(n_sims))
    #==========================================================================

    n_cpus = min(n_sims, n_cpus)

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

    all_sims = {cols[k]:{ref_lab: data[:, k].copy()} for k in range(len(cols))}

    sim_labs = []
    for sims in ress:
        for col in cols:
            all_sims[col].update(sims[col])

            sim_labs.extend(list(sims[col].keys()))

    ress = sims = None
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
    return


class Data:

    def __init__(self):

        return


def get_ts_data_cls(data, noise_add_flag, noise_magnitude):

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

    phss_diffs = phss - phss[:, [0]]
    phss_ranks_diffs = phss_ranks - phss_ranks[:, [0]]

    data_cls.phss_diffs = phss_diffs
    data_cls.phss_ranks_diffs = phss_ranks_diffs
    #==========================================================================

    data_sort = np.sort(data, axis=0)

    data_cls.data_sort = data_sort
    #==========================================================================

    data_cls.pcorrs = np.corrcoef(data, rowvar=False)
    data_cls.scorrs = np.corrcoef(rankdata(data, axis=0), rowvar=False)
    #==========================================================================

    return data_cls


def get_obj_val(ref_data_cls, sim_data):

    obj_val = 0.0

    obj_val += (
        (ref_data_cls.pcorrs -
         np.corrcoef(sim_data, rowvar=False)) ** 2).sum()

    obj_val += (
        (ref_data_cls.scorrs -
         np.corrcoef(rankdata(sim_data, axis=0), rowvar=False)) ** 2).sum()

    return obj_val


def get_sim_dict(args):

    (ref_data_cls,
     cols,
     ratio_a,
     ratio_b,
     auto_spec_flag,
     cross_spec_flag,
     n_repeat,
     max_opt_iters,
     sim_idx,
     n_sims,
     sim_lab,
    ) = args
    #==========================================================================

    assert any([ratio_a, ratio_b])
    assert any([auto_spec_flag, cross_spec_flag])
    #==========================================================================

    data = ref_data_cls.data

    # ref_ft = ref_data_cls.ft
    ref_phs = ref_data_cls.phss
    ref_mag = ref_data_cls.mags

    # ref_ft_ranks = ref_data_cls.ft_ranks
    ref_phs_ranks = ref_data_cls.phss_ranks
    ref_mag_ranks = ref_data_cls.mags_ranks

    # ref_phs_diffs = ref_data_cls.phss_diffs
    # ref_phs_ranks_diffs = ref_data_cls.phss_ranks_diffs

    data_sort = ref_data_cls.data_sort
    #==========================================================================

    sim_zeros_str = len(str(n_sims))
    #==========================================================================

    data_rand = np.empty_like(data)
    order_old = np.empty(data_sort.shape, dtype=int)
    for k in range(len(cols)):
        order_old[:, k] = np.argsort(np.argsort(
            np.random.random(data_sort.shape[0])))

        data_rand[:, k] = data_sort[order_old[:, k], k]
    #==========================================================================

    if cross_spec_flag:
        obj_val_global_min = get_obj_val(ref_data_cls, data_rand)
        data_best = data_rand.copy()

    stn_ctr = 0
    for i_repeat in range(n_repeat):
        break_flag_auto = False
        if auto_spec_flag:
            for _ in range(max_opt_iters):
                # Marginals.
                if ratio_a:
                    sim_ft = np.fft.rfft(data_rand, axis=0)

                    sim_phs = np.angle(sim_ft)

                    sim_ft_new = np.empty_like(sim_ft)

                    sim_ft_new.real[:] = np.cos(sim_phs) * ref_mag
                    sim_ft_new.imag[:] = np.sin(sim_phs) * ref_mag

                    sim_ft_new[0,:] = 0

                    sim_ift_a = np.fft.irfft(sim_ft_new, axis=0)

                    sim_ift_a /= sim_ift_a.std(axis=0)

                else:
                    sim_ift_a = 0.0

                # Ranks.
                if ratio_b:
                    sim_ft = np.fft.rfft(rankdata(data_rand, axis=0), axis=0)

                    sim_phs = np.angle(sim_ft)

                    sim_ft_new = np.empty_like(sim_ft)

                    sim_ft_new.real[:] = np.cos(sim_phs) * ref_mag_ranks
                    sim_ft_new.imag[:] = np.sin(sim_phs) * ref_mag_ranks

                    sim_ft_new[0,:] = 0

                    sim_ift_b = np.fft.irfft(sim_ft_new, axis=0)

                    sim_ift_b /= sim_ift_b.std(axis=0)

                else:
                    sim_ift_b = 0.0

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
                    print(f'### Break auto ({i_repeat})!')
                    break_flag_auto = True
                    break
            #==================================================================

        if (not cross_spec_flag) and break_flag_auto:
            print(f'### Break auto ({i_repeat}) full!')
            break

        break_flag_cross = False
        if cross_spec_flag:
            for _ in range(max_opt_iters):
                # Marginals.
                if ratio_a:
                    sim_ft = np.fft.rfft(data_rand, axis=0)

                    sim_mag = np.abs(sim_ft)

                    # sim_phs = np.angle(sim_ft[:, [0]]) + ref_phs_diffs

                    sim_phs = (
                        np.angle(sim_ft[:, [stn_ctr]]) +
                        ref_phs -
                        ref_phs[:, [stn_ctr]])

                    sim_phs[0,:] = ref_phs[0,:]

                    sim_ft_new = np.empty_like(sim_ft)

                    sim_ft_new.real[:] = np.cos(sim_phs) * sim_mag
                    sim_ft_new.imag[:] = np.sin(sim_phs) * sim_mag

                    sim_ft_new[0,:] = 0

                    sim_ift_a = np.fft.irfft(sim_ft_new, axis=0)

                    sim_ift_a /= sim_ift_a.std(axis=0)

                else:
                    sim_ift_a = 0.0

                # Ranks.
                if ratio_b:
                    sim_ft = np.fft.rfft(rankdata(data_rand, axis=0), axis=0)

                    sim_mag = np.abs(sim_ft)

                    # sim_phs = np.angle(sim_ft[:, [0]]) + ref_phs_ranks_diffs

                    sim_phs = (
                        np.angle(sim_ft[:, [stn_ctr]]) +
                        ref_phs_ranks -
                        ref_phs_ranks[:, [stn_ctr]])

                    sim_phs[0,:] = ref_phs_ranks[0,:]

                    sim_ft_new = np.empty_like(sim_ft)

                    sim_ft_new.real[:] = np.cos(sim_phs) * sim_mag
                    sim_ft_new.imag[:] = np.sin(sim_phs) * sim_mag

                    sim_ft_new[0,:] = 0

                    sim_ift_b = np.fft.irfft(sim_ft_new, axis=0)

                    sim_ift_b /= sim_ift_b.std(axis=0)

                else:
                    sim_ift_b = 0.0

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
                    print(f'### Break cross ({i_repeat})!')
                    break_flag_cross = True
                    break
            #==================================================================
        #======================================================================

        if cross_spec_flag:
            obj_val = get_obj_val(ref_data_cls, data_rand)

            if obj_val < obj_val_global_min:
                obj_val_global_min = obj_val

                data_best = data_rand.copy()
        #======================================================================

        if (not auto_spec_flag) and break_flag_cross:
            print(f'### Break cross ({i_repeat}) full!')
            break

        if break_flag_auto and break_flag_cross:
            print(f'### Break auto and cross ({i_repeat}) full!')
            break
        #======================================================================

        stn_ctr += 1
        if stn_ctr == data_rand.shape[1]:
            stn_ctr = 0
    #==========================================================================

    sims = {cols[k]:{} for k in range(len(cols))}
    for k, col in enumerate(cols):
        if cross_spec_flag:
            sims[col][
                f'{sim_lab}{sim_idx:0{sim_zeros_str}d}'] = data_best[:, k]

        else:
            sims[col][
                f'{sim_lab}{sim_idx:0{sim_zeros_str}d}'] = data_rand[:, k]

    print(f'Done with sim_idx: {sim_idx} with i_repeat: {i_repeat}')
    return sims


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
