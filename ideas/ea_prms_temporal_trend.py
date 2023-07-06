# -*- coding: utf-8 -*-

'''
@author: Faizan-Uni-Stuttgart

Feb 17, 2023

10:21:10 AM

Keywords: Trend

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
import pickle
import timeit
import traceback as tb
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
# from scipy.optimize import differential_evolution

from fdiffevo import fde_host
from cc_cumm_ft_corr import get_cumm_ft_corr_auto

DEBUG_FLAG = False


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Testings\opts\trend_prms_ts')
    os.chdir(main_dir)

    opt_label = 'trend_prms_ts_04'

    trend_max_len = 365 * 40

    trend_thresh_steps = 365

    trend_half_len = 365

    min_mag = 0.0
    max_mag = 3e6

    min_prm = 1.0
    max_prm = trend_half_len * 2

    n_cpus = 8
    alg_cfg_type = 1

    pop_size = int(5e2)
    max_iters = int(1e4)
    max_cont_iters = int(3e3)

    # pop_size = 50
    # max_iters = 100
    # max_cont_iters = 100

    min_thresh_obj_val = np.inf
    save_opt_snapshots_flag = False
    ret_opt_args_flag = False
    verbose = False

    out_opt_prms_path = Path(f'opt_prms_{opt_label}.csv')
    out_opt_pckl_path = Path(f'opt_snapshot_{opt_label}.pkl')

    thresh_idx = int(trend_max_len / trend_thresh_steps)

    trend_ref = np.arange(trend_half_len)
    trend_ref = np.concatenate((trend_ref, trend_ref[::-1]))

    bds_mean = np.array([[min_prm, max_prm]])
    bds_mags = np.tile([min_mag, max_mag], (thresh_idx, 1))
    bds_phss = np.tile([-np.pi, +np.pi], (thresh_idx, 1))

    opt_bds = np.concatenate((bds_mean, bds_mags, bds_phss), axis=0)
    #==========================================================================

    if (trend_ref.size % 2):
        trend_ref = trend_ref[:-1]

    opt_ts_beg_idx = (trend_max_len - trend_ref.size) // 2
    opt_ts_end_idx = opt_ts_beg_idx + trend_ref.size

    opt_args_cls = OPTARGS()
    opt_args_cls.ref_ts = trend_ref
    opt_args_cls.opt_ts_beg_idx = opt_ts_beg_idx
    opt_args_cls.opt_ts_end_idx = opt_ts_end_idx
    opt_args_cls.len_trend_ts = trend_max_len
    opt_args_cls.trend_ft = None
    opt_args_cls.min_prm = min_prm
    opt_args_cls.max_prm = max_prm

    # opt_ress = differential_evolution(
    #     obj_ftn,
    #     bounds=opt_bds,
    #     args=(opt_args_cls,),
    #     popsize=50,
    #     workers=8)
    #
    # prms = opt_ress.x

    # fde_host signature.
    # best_prms, best_obj_val, state_snapshots, ret_ftn_args = fde_host(
    #     obj_ftn,
    #     obj_ftn_args,
    #     prm_bds,
    #     n_prm_vecs,
    #     n_cpus,
    #     mu_sc_fac_bds,
    #     cr_cnst_bds,
    #     max_iters,
    #     max_cont_iters,
    #     prm_pcnt_tol,
    #     new_obj_acpt_ratio,
    #     obj_ftn_tol,
    #     min_thresh_obj_val,
    #     save_snapshots_flag,
    #     ret_opt_args_flag,
    #     alg_cfg_type,
    #     verbose)

    prms, best_obj_val, state_snapshots, ret_opt_args = fde_host(
        obj_ftn,
        opt_args_cls,
        opt_bds,
        pop_size,
        n_cpus,
        # (0.01, 0.2),
        # (0.10, 0.2),
        (0.01, 0.5),
        (0.7, 1.0),
        max_iters,
        max_cont_iters,
        0.05,
        1e-5,
        1e-8,
        min_thresh_obj_val,
        save_opt_snapshots_flag,
        ret_opt_args_flag,
        alg_cfg_type,
        verbose)

    _ = ret_opt_args

    np.savetxt(out_opt_prms_path, prms, fmt='%0.8f')

    if save_opt_snapshots_flag:
        with open(out_opt_pckl_path, 'wb') as pkl_hdl:
            pickle.dump(state_snapshots, pkl_hdl)

    mean = prms[0]

    n_mags = (prms.size - 1) // 2

    mags = prms[1:n_mags + 1]
    phss = prms[n_mags + 1:]

    print('\n\n')
    print(best_obj_val)
    print(mean)
    print(mags)
    print(phss)

    trend_sim = simulate_trend_ts(
        mean,
        mags,
        phss,
        trend_max_len,
        opt_args_cls)

    trend_ref_xcrds = np.arange(
        -(trend_ref.size // 2), (trend_ref.size // 2))

    trend_sim_xcrds = np.arange(
        1 - (trend_max_len // 2), (trend_max_len // 2) - 1)

    cumm_ft_corr, periods = get_cumm_ft_corr_auto(trend_ref)

    if False:
        plt.semilogx(
            periods,
            cumm_ft_corr,
            alpha=0.75,
            color='r',
            label='REF',
            lw=4,
            zorder=1)

        plt.scatter(
            periods,
            cumm_ft_corr,
            alpha=0.75,
            color='r',
            zorder=2)

        plt.xlim(plt.xlim()[::-1])

        plt.xlabel('Period')
        plt.ylabel('Cummulative power')

    else:
        plt.plot(
            trend_ref_xcrds,
            trend_ref,
            alpha=0.75,
            color='r',
            label='REF',
            lw=4,
            zorder=2)

        plt.plot(
            trend_sim_xcrds,
            trend_sim,
            alpha=0.75,
            color='b',
            label='SIM',
            lw=2,
            zorder=3)

        plt.xlabel('Timestep')
        plt.ylabel('Value')

    plt.legend()

    plt.grid()
    plt.gca().set_axisbelow(True)

    plt.show()

    # plt.savefig(f'{in_file.stem}_{out_fig_name_pwr}', bbox_inches='tight')
    plt.close()
    return


def obj_ftn(prms, args):

    ref_ts = args.ref_ts
    opt_ts_beg_idx = args.opt_ts_beg_idx
    opt_ts_end_idx = args.opt_ts_end_idx
    len_trend_ts = args.len_trend_ts

    min_prm = -args.max_prm
    max_prm = +args.max_prm

    mean = prms[0]

    n_mags = (prms.size - 1) // 2

    mags = prms[1:n_mags + 1]
    phss = prms[n_mags + 1:]

    sim_ts = simulate_trend_ts(mean, mags, phss, len_trend_ts, args)

    obj_val = 0.0
    obj_val += ((
        ref_ts -
        sim_ts[opt_ts_beg_idx:opt_ts_end_idx]) ** 2).sum()

    obj_val += np.abs((sim_ts[sim_ts < min_prm])).sum()
    obj_val += np.abs((sim_ts[sim_ts > max_prm])).sum()

    if False:
        print(round(obj_val, 2))

    return obj_val


def simulate_trend_ts(mean, mags, phss, len_ts, args):

    if args.trend_ft is None:
        args.trend_ft = np.zeros(len_ts // 2, dtype=np.complex128)

    args.trend_ft[0] = mean * len_ts

    n_mags = mags.size

    args.trend_ft[1:n_mags + 1].real = mags * np.cos(phss)
    args.trend_ft[1:n_mags + 1].imag = mags * np.sin(phss)

    trend_ts = np.fft.irfft(args.trend_ft)

    return trend_ts


class OPTARGS:

    def __init__(self):

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
