'''
@author: Faizan-Uni-Stuttgart

Mar 4, 2022

9:14:23 AM

'''
import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path
from math import factorial
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import rankdata
import matplotlib.pyplot as plt; plt.ioff()

DEBUG_FLAG = True


class DTVar:

    def __init__(self):

        self.data = None
        self.data_srtd = None

        self.ft = None
        self.mags_spec = None
        self.phss_spec = None

        self.auto_cumm_corrs = None
        self.auto_cumm_corrs_norm_vals = None

        self.pair_cumm_corrs = None
        self.pair_cumm_corrs_norm_vals = None
        return


class Rltzn:

    def __init__(self):

        self.data_dt_vars = DTVar()
        self.probs_dt_vars = DTVar()
        return


def get_ms_cross_pair_ft(
        mags_spec, phss_spec, vtype, data_type, ref_rltzn_cls):

    '''
    Pairwise cross cummulative correlation spectrum with phases.
    '''

    assert mags_spec.ndim == 2

    n_recs, n_cols = mags_spec.shape

    mags_spec = mags_spec[1:,:].copy(order='f')
    phss_spec = phss_spec[1:,:].copy(order='f')

    comb_size = 2

    if vtype == 'ref':
        norm_vals = []
        pwr_spec_sum_sqrt = (mags_spec ** 2).sum(axis=0) ** 0.5

    combs = combinations(np.arange(n_cols), comb_size)

    n_combs = int(
        factorial(n_cols) /
        (factorial(comb_size) *
         factorial(n_cols - comb_size)))

    pair_cumm_corrs = np.empty((n_recs - 1, n_combs), order='f')

    for i, comb in enumerate(combs):
        col_idxs = [col for col in comb]

        if len(comb) != 2:
            raise NotImplementedError('Configured for pairs only!')

        numr = (
            mags_spec[:, col_idxs[0]] *
            mags_spec[:, col_idxs[1]] *
            np.cos(phss_spec[:, col_idxs[0]] -
                   phss_spec[:, col_idxs[1]])
            )

        pair_cumm_corrs[:, i] = numr

        if vtype == 'ref':
            demr = (
                pwr_spec_sum_sqrt[col_idxs[0]] *
                pwr_spec_sum_sqrt[col_idxs[1]])

            norm_vals.append(demr)

    if vtype == 'ref':
        norm_vals = np.array(norm_vals).reshape(1, -1)

        n_combs = int(
            factorial(n_cols) /
            (factorial(comb_size) *
             factorial(n_cols - comb_size)))

        assert norm_vals.size == n_combs, (norm_vals.size, n_combs)

        if data_type == 'data':
            ref_rltzn_cls.data_dt_vars.pair_cumm_corrs_norm_vals = norm_vals

        elif data_type == 'probs':
            ref_rltzn_cls.probs_dt_vars.pair_cumm_corrs_norm_vals = norm_vals

        else:
            raise NotImplementedError

    elif vtype == 'sim':
        if data_type == 'data':
            norm_vals = ref_rltzn_cls.data_dt_vars.pair_cumm_corrs_norm_vals

        elif data_type == 'probs':
            norm_vals = ref_rltzn_cls.probs_dt_vars.pair_cumm_corrs_norm_vals

        else:
            raise NotImplementedError

    else:
        raise NotImplementedError

    pair_cumm_corrs = np.cumsum(pair_cumm_corrs, axis=0)

    pair_cumm_corrs /= norm_vals

    return pair_cumm_corrs


def get_auto_cumm_corrs_ft(mags_spec, vtype, data_type, ref_rltzn_cls):

    '''
    Auto cumm. corr.

    mag_spec can be of the data or probs.
    '''

    mags_spec = mags_spec[1:,:]

    cumm_pwrs = (mags_spec ** 2).cumsum(axis=0)

    if (vtype == 'sim') and (data_type == 'data'):
        norm_vals = ref_rltzn_cls.data_dt_vars.auto_cumm_corrs_norm_vals

    elif (vtype == 'sim') and (data_type == 'probs'):
        norm_vals = ref_rltzn_cls.probs_dt_vars.auto_cumm_corrs_norm_vals

    elif (vtype == 'ref') and (data_type == 'data'):
        norm_vals = cumm_pwrs[-1,:].copy().reshape(1, -1)
        ref_rltzn_cls.data_dt_vars.auto_cumm_corrs_norm_vals = norm_vals

    elif (vtype == 'ref') and (data_type == 'probs'):
        norm_vals = cumm_pwrs[-1,:].copy().reshape(1, -1)
        ref_rltzn_cls.probs_dt_vars.auto_cumm_corrs_norm_vals = norm_vals

    else:
        raise NotImplementedError

    cumm_pwrs /= norm_vals

    return cumm_pwrs


def cmpt_vars(vtype, ref_rltzn_cls, sim_rltzn_cls, data_type):

    '''
    NOTE: For simulated time series, the data should be original reshuffled
    data.
    For sim_rltzn_cls, only the data attribute of the data_dt_vars should
    exist.
    '''

    if vtype == 'ref':
        rltzn_cls = ref_rltzn_cls

    elif vtype == 'sim':
        rltzn_cls = sim_rltzn_cls

    else:
        raise NotImplementedError(vtype)
    #==========================================================================

    data = rltzn_cls.data_dt_vars.data

    if data_type == 'data':
        rltzn_dtvar = rltzn_cls.data_dt_vars

        if vtype == 'ref':
            rltzn_dtvar.data_srtd = np.sort(data, axis=0)

        ft_data = data

    elif data_type == 'probs':
        probs = rankdata(data, axis=0) / (data.shape[0] + 1.0)

        rltzn_dtvar = rltzn_cls.probs_dt_vars

        rltzn_dtvar.data = probs

        ft_data = probs

    else:
        raise NotImplementedError((vtype, data_type))
    #==========================================================================

    ft = np.fft.rfft(ft_data, axis=0)

    rltzn_dtvar.ft = ft

    rltzn_dtvar.mags_spec = np.abs(rltzn_dtvar.ft)
    rltzn_dtvar.phss_spec = np.angle(rltzn_dtvar.ft)

    rltzn_dtvar.auto_cumm_corrs = get_auto_cumm_corrs_ft(
        rltzn_dtvar.mags_spec, vtype, data_type, ref_rltzn_cls)

    rltzn_dtvar.pair_cumm_corrs = get_ms_cross_pair_ft(
        rltzn_dtvar.mags_spec,
        rltzn_dtvar.phss_spec,
        vtype,
        data_type,
        ref_rltzn_cls)

    return


def phs_rand_sim(ref_rltzn_cls, sim_rltzn_cls, rand_idxs, tfm_type):

    assert np.all(rand_idxs > 0)

    if tfm_type == 'data':
        ref_rltzn_dtvar = ref_rltzn_cls.data_dt_vars

    elif tfm_type == 'probs':
        ref_rltzn_dtvar = ref_rltzn_cls.probs_dt_vars

    else:
        raise NotImplementedError(tfm_type)

    ft = ref_rltzn_dtvar.ft.copy()
    mags_spec = ref_rltzn_dtvar.mags_spec[rand_idxs,:].copy()
    phss_spec = ref_rltzn_dtvar.phss_spec[rand_idxs,:].copy()

    rand_phss = -np.pi + ((2 * np.pi) * np.random.random(rand_idxs.shape))

    phss_spec += rand_phss[:, None]
    # phss_spec = np.arccos(np.cos(phss_spec))  # This didn't work as I expected.

    ft_sub = np.empty_like(mags_spec, dtype=complex)
    ft_sub.real = mags_spec * np.cos(phss_spec)
    ft_sub.imag = mags_spec * np.sin(phss_spec)

    ft[rand_idxs,:] = ft_sub

    ift = np.fft.irfft(ft, axis=0)

    data = np.empty_like(ref_rltzn_cls.data_dt_vars.data)

    for i in range(data.shape[1]):
        data[:, i] = ref_rltzn_cls.data_dt_vars.data_srtd[
            np.argsort(np.argsort(ift[:, i])), i]

    # sim_rltzn_dtvar.data = data
    sim_rltzn_cls.data_dt_vars.data = data
    return


def get_obj_val(ref_rltzn_cls, sim_rltzn_cls):

    obj_val = 0.0

    if True:
        obj_val += ((
            ref_rltzn_cls.data_dt_vars.auto_cumm_corrs -
            sim_rltzn_cls.data_dt_vars.auto_cumm_corrs) ** 2).sum()

    if True:
        obj_val += ((
            ref_rltzn_cls.probs_dt_vars.auto_cumm_corrs -
            sim_rltzn_cls.probs_dt_vars.auto_cumm_corrs) ** 2).sum()

    if True:
        obj_val += ((
            ref_rltzn_cls.data_dt_vars.pair_cumm_corrs -
            sim_rltzn_cls.data_dt_vars.pair_cumm_corrs) ** 2).sum()

    if True:
        obj_val += ((
            ref_rltzn_cls.probs_dt_vars.pair_cumm_corrs -
            sim_rltzn_cls.probs_dt_vars.pair_cumm_corrs) ** 2).sum()

    return obj_val


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Testings\fourtrans_practice\phs_rand_effects_on_obj_val')
    os.chdir(main_dir)

    in_file_path = Path(r'neckar_1hr_ppt_data_20km_buff_Y2004_2020_10cps.pkl')

    labels = ['P1176', 'P1290']  #  , 'P2159', 'P2292', ]

    time_fmt = '%Y-%m-%d'

    steps_per_day = 24

    beg_time = '2009-01-01'
    end_time = '2009-12-31'

    sep = ';'

    tfm_type = 'probs'

    n_tot_repeats = 1
    n_idxs_to_rand = 1000
    n_idx_rands = 10
    idx_rand_size = 5

    out_dir = Path('test_10')
    #==========================================================================

    out_dir.mkdir(exist_ok=True)

    n_cols = len(labels)

    if in_file_path.suffix == '.csv':
        in_df = pd.read_csv(in_file_path, sep=sep, index_col=0)
        in_df.index = pd.to_datetime(in_df.index, format=time_fmt)

    elif in_file_path.suffix == '.pkl':
        in_df = pd.read_pickle(in_file_path)

    else:
        raise NotImplementedError(
            f'Unknown extension of in_data_file: '
            f'{in_file_path.suffix}!')

    sub_df = in_df.loc[beg_time:end_time, labels]

    data = sub_df.values

    if data.shape[0] % 2:
        data = data[:-1,:]

    assert np.all(np.isfinite(data))

    ref_rltzn_cls = Rltzn()
    sim_rltzn_cls = None

    ref_rltzn_cls.data_dt_vars.data = data

    cmpt_vars('ref', ref_rltzn_cls, sim_rltzn_cls, 'data')
    cmpt_vars('ref', ref_rltzn_cls, sim_rltzn_cls, 'probs')

    ft_size = ref_rltzn_cls.data_dt_vars.ft.shape[0] - 1

    plt.figure(figsize=(10, 7))

    phs_obj_vals_sum = np.zeros(ft_size)

    all_rand_idxs = np.arange(n_idxs_to_rand)

    for j in range(n_tot_repeats):
        # print(j)
        obj_vals = []
        for i in range(n_idxs_to_rand):
            for k in range(n_idx_rands):
                sim_rltzn_cls = Rltzn()

                # rand_idxs = np.array([i + 1])
                rand_idxs = np.random.choice(
                    all_rand_idxs, idx_rand_size, replace=False)

                phs_rand_sim(
                    ref_rltzn_cls,
                    sim_rltzn_cls,
                    (rand_idxs + 1),
                    tfm_type)

                cmpt_vars('sim', ref_rltzn_cls, sim_rltzn_cls, 'data')
                cmpt_vars('sim', ref_rltzn_cls, sim_rltzn_cls, 'probs')

                obj_val = round(get_obj_val(ref_rltzn_cls, sim_rltzn_cls), 3)

                print(j, i, k, obj_val)

                obj_vals.append(obj_val)

                phs_obj_vals_sum[rand_idxs] += obj_val

        obj_vals = np.array(obj_vals)
        obj_vals.sort()

        obj_probs = rankdata(obj_vals, method='max') / (obj_vals.size + 1.0)

        plt.plot(obj_vals, obj_probs, alpha=0.7, c='k')

    plt.grid()
    plt.gca().set_axisbelow(True)

    plt.xlabel('obj_val')
    plt.ylabel('Non-exceedence probability')

    plt.title(
        f'n_tot_repeats: {n_tot_repeats}, '
        f'n_idxs_to_rand: {n_idxs_to_rand}, '
        f'n_idx_rands: {n_idx_rands}\n'
        f'n_vals: {data.shape[0]}, '
        f'tfm_type: {tfm_type}, '
        f'n_cols: {n_cols}, '
        f'idx_rand_size: {idx_rand_size}'
        )

    # plt.show()

    fig_name = (
        f'obj_val_dists_{n_tot_repeats}_{n_idxs_to_rand}_'
        f'{n_idx_rands}_{data.shape[0]}_{tfm_type}_{n_cols}_'
        f'{idx_rand_size}.png')

    plt.savefig(str(out_dir / fig_name), bbox_inches='tight')
    plt.clf()
    #==========================================================================

    phs_obj_vals_sum /= n_tot_repeats
    phs_obj_vals_sum /= n_idx_rands

    phs_obj_vals_sum = phs_obj_vals_sum.cumsum()
    # phs_obj_vals_sum /= phs_obj_vals_sum[-1]

    ref_periods = (ft_size * 2) / (np.arange(1.0, ft_size + 1.0))

    ref_periods /= steps_per_day

    plt.semilogx(
        ref_periods[:n_idxs_to_rand],
        phs_obj_vals_sum[:n_idxs_to_rand],
        alpha=0.7,
        c='k')

    plt.grid()
    plt.gca().set_axisbelow(True)

    plt.xlabel('Period [days]')
    plt.ylabel('Cumm. Obj. vals. sum')

    plt.xlim(plt.xlim()[::-1])

    # plt.ylim(-0.02, +1.02)

    plt.title(
        f'n_tot_repeats: {n_tot_repeats}, '
        f'n_idxs_to_rand: {n_idxs_to_rand}, '
        f'n_idx_rands: {n_idx_rands}\n'
        f'n_vals: {data.shape[0]}, '
        f'tfm_type: {tfm_type}, '
        f'n_cols: {n_cols}, '
        f'idx_rand_size: {idx_rand_size}'
        )

    fig_name = (
        f'phs_obj_val_sum_{n_tot_repeats}_{n_idxs_to_rand}_'
        f'{n_idx_rands}_{data.shape[0]}_{tfm_type}_{n_cols}_'
        f'{idx_rand_size}.png')

    plt.savefig(str(out_dir / fig_name), bbox_inches='tight')

    plt.close()
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
