# -*- coding: utf-8 -*-

'''
@author: Faizan-Uni-Stuttgart

Dec 15, 2022

1:49:15 PM

Keywords: Contribution of different quantiles to total variance.

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
from scipy.stats import rankdata
import matplotlib.pyplot as plt; plt.ioff()
# from matplotlib.colors import ListedColormap, BoundaryNorm

from kde import KERNEL_FTNS_DICT

from za_cmn_ftns import (
    get_dis_stn_subsets_daily_1961_2015,
    set_mpl_prms,
    )

DEBUG_FLAG = False


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Projects\2016_DFG_SPATE\data')
    os.chdir(main_dir)

    # How much in the upper tail.
    upp_thresh = 0.005

    var_type = 'margs'
    # var_type = 'probs'

    time_label = 'Time step'
    time_units = 'day'

    if var_type == 'margs':
        var_name = 'Discharge'
        var_units = '$m^3.s^{-1}$'

    elif var_type == 'probs':
        var_name = 'p'
        var_units = '-'

    else:
        raise NotImplementedError(f'Unknown var_type: {var_type}!')

    period_thresh_long = 370
    period_thresh_shrt = 360

    take_thresh_mean_flag = True
    # take_thresh_mean_flag = False

    kernel = KERNEL_FTNS_DICT['silv']

    prms_dict = {
        'figure.figsize': (15, 20),
        'figure.dpi': 300,
        'font.size': 12,
        }

    out_dir = Path('peaks_behavior_plots')
    #==========================================================================

    out_dir.mkdir(exist_ok=True)

    set_mpl_prms(prms_dict)

    _, tss_path, _, subsets = (
        get_dis_stn_subsets_daily_1961_2015())

    for subset_stns in subsets:

        print(subset_stns)

        plot_peaks_behavior(
            (tss_path,
             subset_stns,
             upp_thresh,
             out_dir,
             kernel,
             var_type,
             time_label,
             time_units,
             var_name,
             var_units,
             period_thresh_long,
             period_thresh_shrt,
             take_thresh_mean_flag,
             ))

        break

    return


def plot_peaks_behavior(args):

    (tss_path,
     subset_stns,
     upp_thresh,
     out_dir,
     kernel,
     var_type,
     time_label,
     time_units,
     var_name,
     var_units,
     period_thresh_long,
     period_thresh_shrt,
     take_thresh_mean_flag,
     ) = args

    # tss_path, subset_stns, upp_thresh, out_dir, kernel, var_type = args

    assert var_type in ('margs', 'probs'), var_type

    in_df = pd.read_pickle(tss_path)

    if subset_stns is not None:
        in_df = in_df.loc[:, subset_stns].copy()

    else:
        subset_stns = in_df.columns

    if var_type == 'probs':
        in_df = in_df.rank(axis=0, method='max') / (in_df.shape[0] + 1.0)

    line_alpha = 0.9
    scat_alpha = 0.1
    n_scatt = 20

    clr_ref = 'r'
    clr_upp = 'b'
    clr_low = 'g'
    clr_vrc = 'k'
    clr_pdl = 'orange'
    clr_pds = 'cyan'

    label_ref = 'ref'
    label_upp = 'upr'
    label_low = 'lwr'
    label_pdl = 'long'
    label_pds = 'short'

    lw_ref = 2.0
    lw_upp = 1.5
    lw_low = 1.5

    axes = plt.subplots(9, 1, squeeze=False)[1].ravel()

    for i in range(in_df.shape[1]):

        print(in_df.columns[i])

        #======================================================================
        # Time series.
        #======================================================================

        j = 0
        data = in_df.iloc[:, i].values.copy()

        probs = in_df.iloc[:, i].rank().values / (1.0 + in_df.shape[0])

        upp_lim = data[np.argmin((probs - (1 - upp_thresh)) ** 2)]

        if take_thresh_mean_flag:
            data_mean_upp = data[data >= upp_lim].mean()
            data_mean_low = data[data < upp_lim].mean()

        else:
            data_mean_upp = 0.0
            data_mean_low = 0.0

        axes[j].plot(
            data,
            alpha=line_alpha,
            color=clr_ref,
            label=label_ref,
            lw=lw_ref,
            zorder=1)

        axes[j].axhline(
            upp_lim,
            c=clr_upp,
            label=f'{label_upp} threshold',
            zorder=2,
            alpha=line_alpha,
            lw=lw_upp)

        axes[j].legend()

        axes[j].grid()
        axes[j].set_axisbelow(True)

        axes[j].set_xlabel(f'{time_label} [{time_units}]')
        axes[j].set_ylabel(f'{var_name} [{var_units}]')

        #======================================================================
        # Data CDF.
        #======================================================================

        data = in_df.iloc[:, i].values.copy()

        data = np.sort(data)

        probs = rankdata(data, method='max') / (1.0 + data.size)

        j += 1

        axes[j].plot(
            data,
            probs,
            alpha=line_alpha,
            color=clr_ref,
            lw=lw_ref,
            zorder=2)

        axes[j].scatter(
            data,
            probs,
            alpha=scat_alpha,
            color=clr_ref,
            zorder=1,
            edgecolor='none')

        axes[j].axvline(
            upp_lim,
            c=clr_upp,
            label=label_upp,
            zorder=2,
            alpha=line_alpha,
            lw=lw_upp)

        axes[j].grid()
        axes[j].set_axisbelow(True)

        axes[j].set_ylabel(f'F({var_name}) [-]')

        plt.setp(axes[j].get_xticklabels(), visible=False)

        #======================================================================
        # Data PDF.
        #======================================================================

        density = get_kernel_density(data, kernel)

        density /= density.sum()

        j += 1

        axes[j].plot(
            data,
            density,
            alpha=line_alpha,
            color=clr_ref,
            lw=lw_ref,
            zorder=1)

        axes[j].scatter(
            data,
            density,
            alpha=scat_alpha,
            color=clr_ref,
            zorder=1,
            edgecolor='none')

        axes[j].axvline(
            upp_lim,
            c=clr_upp,
            label=label_upp,
            zorder=2,
            alpha=line_alpha,
            lw=lw_upp)

        axes[j].grid()
        axes[j].set_axisbelow(True)

        axes[j].set_ylabel(f'f({var_name}) [-]')

        plt.setp(axes[j].get_xticklabels(), visible=False)

        axes[j].sharex(axes[j - 1])

        #======================================================================
        # Variance CDF.
        #======================================================================

        vrc_sqs = (data - data.mean()) ** 2

        vrc_sqs_cumsum = vrc_sqs.cumsum()
        vrc_sqs_cumsum /= vrc_sqs_cumsum[-1]

        upp_lim_vrc = vrc_sqs_cumsum[np.argmin((data - upp_lim) ** 2)]

        j += 1

        axes[j].plot(
            data,
            vrc_sqs_cumsum,
            alpha=line_alpha,
            color=clr_ref,
            lw=lw_ref,
            zorder=2)

        axes[j].scatter(
            data,
            vrc_sqs_cumsum,
            alpha=scat_alpha,
            color=clr_ref,
            zorder=1,
            edgecolor='none')

        axes[j].axvline(
            upp_lim,
            c=clr_upp,
            label=label_upp,
            zorder=2,
            alpha=line_alpha,
            lw=lw_upp)

        axes[j].axhline(
            upp_lim_vrc,
            c=clr_vrc,
            zorder=3,
            alpha=line_alpha,
            lw=lw_upp)

        axes[j].text(
            0.98,
            0.1,
            f'V({label_upp})={1-upp_lim_vrc:0.2f}',
            ha='right',
            va='bottom',
            transform=axes[j].transAxes,
            bbox={'facecolor':'white',
                  'boxstyle':f'round,pad=0.2',
                  'edgecolor':'white'})

        axes[j].grid()
        axes[j].set_axisbelow(True)

        axes[j].set_ylabel('F(Variance) [-]')
        axes[j].set_xlabel(f'{var_name} [{var_units}]')

        axes[j].sharex(axes[j - 2])

        #======================================================================
        # Power spectra.
        #======================================================================

        data = in_df.iloc[:, i].values.copy()

        probs = in_df.iloc[:, i].rank().values / (1.0 + in_df.shape[0])

        # Full series.
        ft = np.fft.rfft(data)[1:]

        mag = np.abs(ft)

        pwr = mag ** 2

        pwr = pwr.cumsum()

        ref_pwr = pwr[-1]

        pwr /= ref_pwr

        periods = (pwr.size * 2) / np.arange(1, pwr.size + 1)

        assert periods.size == pwr.shape[0]

        j += 1

        axes[j].semilogx(
            periods,
            pwr,
            alpha=line_alpha,
            color=clr_ref,
            label=label_ref,
            lw=lw_ref,
            zorder=1)

        # Upper tail.
        data_thresh_upp = data.copy()

        thresh_upp_idxs = probs >= (1.0 - upp_thresh)

        data_thresh_upp[~thresh_upp_idxs] = data_mean_low

        ft_thresh_upp = np.fft.rfft(data_thresh_upp)[1:]

        pwr_thresh_upp = np.abs(ft_thresh_upp) ** 2

        pwr_thresh_upp = pwr_thresh_upp.cumsum()
        pwr_thresh_upp /= ref_pwr

        axes[j].semilogx(
            periods,
            pwr_thresh_upp,
            alpha=0.7,
            color=clr_upp,
            label=label_upp,
            lw=lw_upp,
            zorder=2)

        # Lower tail.
        data_thresh_low = data.copy()

        thresh_low_idxs = probs < (1.0 - upp_thresh)

        data_thresh_low[~thresh_low_idxs] = data_mean_low

        ft_thresh_low = np.fft.rfft(data_thresh_low)[1:]

        pwr_thresh_low = np.abs(ft_thresh_low) ** 2

        pwr_thresh_low = pwr_thresh_low.cumsum()
        pwr_thresh_low /= ref_pwr

        axes[j].semilogx(
            periods,
            pwr_thresh_low,
            alpha=0.7,
            color=clr_low,
            label=label_low,
            lw=lw_low,
            zorder=3)

        # Threshold frequencies.
        axes[j].axvline(
            periods[np.argmin((periods - period_thresh_long) ** 2)],
            c=clr_pdl,
            label=label_pdl,
            zorder=4,
            alpha=line_alpha,
            lw=lw_upp)

        axes[j].axvline(
            periods[np.argmin((periods - period_thresh_shrt) ** 2)],
            c=clr_pds,
            label=label_pds,
            zorder=5,
            alpha=line_alpha,
            lw=lw_upp)

        axes[j].legend()

        axes[j].grid()
        axes[j].set_axisbelow(True)

        axes[j].set_xlabel(f'Period [{time_units}s]')
        axes[j].set_ylabel('Cummulative\nperiodogram [-]')

        axes[j].set_xlim(axes[j].get_xlim()[::-1])

        #======================================================================
        # Auto-correlation functions.
        #======================================================================

        data = in_df.iloc[:, i].values.copy()

        probs = in_df.iloc[:, i].rank().values / (1.0 + in_df.shape[0])

        cov_ftn = get_auto_cov_from_ft(data)[:data.size // 2]

        # Full series.
        ft = np.fft.rfft(data)[1:]

        periods = (ft.size * 2) / np.arange(1, ft.size + 1)

        assert periods.size == ft.shape[0]

        lead_steps = np.concatenate(([1.0], np.arange(1, cov_ftn.size + 0)))

        j += 1

        axes[j].semilogx(
            lead_steps,
            cov_ftn / cov_ftn[0],
            alpha=0.9,
            color=clr_ref,
            label=label_ref,
            lw=lw_ref,
            zorder=1)

        axes[j].scatter(
            lead_steps[1:n_scatt],
            (cov_ftn / cov_ftn[0])[1:n_scatt],
            alpha=0.9,
            color=clr_ref,
            zorder=1)

        # Upper tail.
        data_thresh_upp = data.copy()

        thresh_upp_idxs = probs >= (1.0 - upp_thresh)

        data_thresh_upp[~thresh_upp_idxs] = data_mean_low

        cov_ftn_thresh_upp = get_auto_cov_from_ft(
            data_thresh_upp)[:data.size // 2]

        axes[j].semilogx(
            lead_steps,
            cov_ftn_thresh_upp / cov_ftn[0],
            alpha=0.7,
            color=clr_upp,
            label=label_upp,
            lw=lw_upp,
            zorder=2)

        axes[j].scatter(
            lead_steps[1:n_scatt],
            (cov_ftn_thresh_upp / cov_ftn[0])[1:n_scatt],
            alpha=0.7,
            color=clr_upp,
            zorder=2)

        # Lower tail.
        data_thresh_low = data.copy()

        thresh_low_idxs = probs < (1.0 - upp_thresh)

        data_thresh_low[~thresh_low_idxs] = data_mean_low

        cov_ftn_thresh_low = get_auto_cov_from_ft(
            data_thresh_low)[:data.size // 2]

        axes[j].semilogx(
            lead_steps,
            cov_ftn_thresh_low / cov_ftn[0],
            alpha=0.7,
            color=clr_low,
            label=label_low,
            lw=lw_low,
            zorder=3)

        axes[j].scatter(
            lead_steps[1:n_scatt],
            (cov_ftn_thresh_low / cov_ftn[0])[1:n_scatt],
            alpha=0.7,
            color=clr_low,
            zorder=3)

        axes[j].grid()
        axes[j].set_axisbelow(True)

        axes[j].set_xlabel(f'Lead step [{time_units}s]')
        axes[j].set_ylabel('Auto-correlation [-]')

        #======================================================================
        # Long-term behavior.
        #======================================================================

        j += 1

        data = in_df.iloc[:, i].values.copy()

        probs = in_df.iloc[:, i].rank().values / (1.0 + in_df.shape[0])

        ft = np.fft.rfft(data)
        ft[0] = 0.0

        periods = ((ft.size - 1) * 2) / np.arange(1, ft.size)

        long_prd_idxs = periods <= period_thresh_long

        ft[1:][long_prd_idxs] = 0.0

        ift = np.fft.irfft(ft)

        axes[j].plot(
            ift,
            alpha=0.9,
            color=clr_ref,
            label=label_ref,
            lw=lw_ref,
            zorder=1)

        # Upper tail.
        data_thresh_upp = data.copy()

        thresh_upp_idxs = probs >= (1.0 - upp_thresh)

        data_thresh_upp[~thresh_upp_idxs] = data_mean_low

        ft_thresh_upp = np.fft.rfft(data_thresh_upp)
        ft_thresh_upp[0] = 0.0

        ft_thresh_upp[1:][long_prd_idxs] = 0.0

        ift_thresh_upp = np.fft.irfft(ft_thresh_upp)

        axes[j].plot(
            ift_thresh_upp,
            alpha=0.7,
            color=clr_upp,
            label=label_upp,
            lw=lw_upp,
            zorder=2)

        # Lower tail.
        data_thresh_low = data.copy()

        thresh_low_idxs = probs < (1.0 - upp_thresh)

        data_thresh_low[~thresh_low_idxs] = data_mean_low

        ft_thresh_low = np.fft.rfft(data_thresh_low)
        ft_thresh_low[0] = 0.0

        ft_thresh_low[1:][long_prd_idxs] = 0.0

        ift_thresh_low = np.fft.irfft(ft_thresh_low)

        axes[j].plot(
            ift_thresh_low,
            alpha=0.7,
            color=clr_low,
            label=label_low,
            lw=lw_low,
            zorder=3)

        axes[j].grid()
        axes[j].set_axisbelow(True)

        axes[j].set_ylabel(f'Long-range\n{var_name} [{var_units}]')

        plt.setp(axes[j].get_xticklabels(), visible=False)

        #======================================================================
        # Mid-term behavior.
        #======================================================================

        j += 1

        data = in_df.iloc[:, i].values.copy()

        probs = in_df.iloc[:, i].rank().values / (1.0 + in_df.shape[0])

        ft = np.fft.rfft(data)
        ft[0] = 0.0

        periods = ((ft.size - 1) * 2) / np.arange(1, ft.size)

        midd_prd_idxs = (
            ~(periods >= period_thresh_shrt) |
            ~(periods <= period_thresh_long))

        assert midd_prd_idxs.sum()

        ft[1:][midd_prd_idxs] = 0.0

        ift = np.fft.irfft(ft)

        axes[j].plot(
            ift,
            alpha=0.9,
            color=clr_ref,
            label=label_ref,
            lw=lw_ref,
            zorder=1)

        # Upper tail.
        data_thresh_upp = data.copy()

        thresh_upp_idxs = probs >= (1.0 - upp_thresh)

        data_thresh_upp[~thresh_upp_idxs] = data_mean_low

        ft_thresh_upp = np.fft.rfft(data_thresh_upp)
        ft_thresh_upp[0] = 0.0

        ft_thresh_upp[1:][midd_prd_idxs] = 0.0

        ift_thresh_upp = np.fft.irfft(ft_thresh_upp)

        axes[j].plot(
            ift_thresh_upp,
            alpha=0.7,
            color=clr_upp,
            label=label_upp,
            lw=lw_upp,
            zorder=2)

        # Lower tail.
        data_thresh_low = data.copy()

        thresh_low_idxs = probs < (1.0 - upp_thresh)

        data_thresh_low[~thresh_low_idxs] = data_mean_low

        ft_thresh_low = np.fft.rfft(data_thresh_low)
        ft_thresh_low[0] = 0.0

        ft_thresh_low[1:][midd_prd_idxs] = 0.0

        ift_thresh_low = np.fft.irfft(ft_thresh_low)

        axes[j].plot(
            ift_thresh_low,
            alpha=0.7,
            color=clr_low,
            label=label_low,
            lw=lw_low,
            zorder=3)

        axes[j].grid()
        axes[j].set_axisbelow(True)

        axes[j].set_ylabel(f'Mid-range\n{var_name} [{var_units}]')

        plt.setp(axes[j].get_xticklabels(), visible=False)

        axes[j].sharex(axes[j - 1])

        #======================================================================
        # Short-term behavior.
        #======================================================================

        j += 1

        data = in_df.iloc[:, i].values.copy()

        probs = in_df.iloc[:, i].rank().values / (1.0 + in_df.shape[0])

        ft = np.fft.rfft(data)
        ft[0] = 0.0

        periods = ((ft.size - 1) * 2) / np.arange(1, ft.size)

        shrt_prd_idxs = periods >= period_thresh_shrt

        ft[1:][shrt_prd_idxs] = 0.0

        ift = np.fft.irfft(ft)

        axes[j].plot(
            ift,
            alpha=0.9,
            color=clr_ref,
            label=label_ref,
            lw=lw_ref,
            zorder=1)

        # Upper tail.
        data_thresh_upp = data.copy()

        thresh_upp_idxs = probs >= (1.0 - upp_thresh)

        data_thresh_upp[~thresh_upp_idxs] = data_mean_low

        ft_thresh_upp = np.fft.rfft(data_thresh_upp)
        ft_thresh_upp[0] = 0.0

        ft_thresh_upp[1:][shrt_prd_idxs] = 0.0

        ift_thresh_upp = np.fft.irfft(ft_thresh_upp)

        axes[j].plot(
            ift_thresh_upp,
            alpha=0.7,
            color=clr_upp,
            label=label_upp,
            lw=lw_upp,
            zorder=2)

        # Lower tail.
        data_thresh_low = data.copy()

        thresh_low_idxs = probs < (1.0 - upp_thresh)

        data_thresh_low[~thresh_low_idxs] = data_mean_low

        ft_thresh_low = np.fft.rfft(data_thresh_low)
        ft_thresh_low[0] = 0.0

        ft_thresh_low[1:][shrt_prd_idxs] = 0.0

        ift_thresh_low = np.fft.irfft(ft_thresh_low)

        axes[j].plot(
            ift_thresh_low,
            alpha=0.7,
            color=clr_low,
            label=label_low,
            lw=lw_low,
            zorder=3)

        axes[j].grid()
        axes[j].set_axisbelow(True)

        axes[j].set_xlabel(f'{time_label} [{time_units}]')
        axes[j].set_ylabel(f'Short-range\n{var_name} [{var_units}]')

        axes[j].sharex(axes[j - 2])

        #======================================================================
        #======================================================================

        plt.suptitle(
            f'Series label: {subset_stns[i]}, N: {in_df.shape[0]}, '
            f'N_upp: {thresh_upp_idxs.sum()}, N_low: {thresh_low_idxs.sum()}'
            f'\n'
            )

        plt.tight_layout()

        fig_name = (
            f'peaks_behavior_{subset_stns[i]}_{var_type}_th_{upp_thresh}.png')

        plt.savefig(out_dir / fig_name, bbox_inches='tight')

        for j in range(len(axes)):
            axes[j].cla()

        break

    plt.close()
    #==========================================================================
    return


def get_auto_cov_from_ft(data):

    ft = np.fft.rfft(data)

    pwr = np.abs(ft) ** 2
    pwr[0] = 0

    covs = np.fft.irfft(pwr, axis=0)

    covs = np.concatenate((covs, covs[[0]]), axis=0)
    return covs


def get_kernel_density(data, kernel):

    dens = np.zeros((data.size, data.size), dtype=float)

    for i in range(data.size):

        dens[i,:] = kernel(data - data[i])

    return dens.sum(axis=0)


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

