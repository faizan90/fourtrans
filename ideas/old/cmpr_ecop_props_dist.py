'''
@author: Faizan-Uni-Stuttgart

8 Jun 2020

14:35:05

'''
import os
import time
import timeit
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
import matplotlib.pyplot as plt

from phsann.cyth import fill_bi_var_cop_dens
from phsann.misc import roll_real_2arrs

plt.ioff()

DEBUG_FLAG = False


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Testings\fourtrans_practice\phsann')
    os.chdir(main_dir)

    in_file = Path(r'neckar_norm_cop_infill_discharge_1961_2015_20190118.csv')

    lag_steps = np.arange(1, 5, dtype=np.int64)

    ecop_bins = 20

    fig_size = (15, 10)
    plt_alpha = 0.5

    in_ser = pd.read_csv(in_file, sep=';', index_col=0)['420']

    in_ser.index = pd.to_datetime(in_ser.index, format='%Y-%m-%d')

    idx_labs = np.unique(in_ser.index.year)

    ecop_dens_arr = np.full((ecop_bins, ecop_bins), np.nan, dtype=np.float64)

    lag_axes = {
        lag_step:plt.subplots(2, 3, squeeze=False, figsize=fig_size)
        for lag_step in lag_steps}

    cmap = 'jet'
    sim_clrs = plt.get_cmap(cmap)(
        (idx_labs - idx_labs.min()) / (idx_labs.max() - idx_labs.min()))

    sim_clrs = {idx_lab:sim_clr for (idx_lab, sim_clr) in zip(idx_labs, sim_clrs)}

    cmap_mappable_beta = plt.cm.ScalarMappable(cmap=cmap)

    cmap_mappable_beta.set_array([])

    for idx_lab in idx_labs:
        data = in_ser.loc[
            f'{idx_lab}-01-01':f'{idx_lab+5}-12-31'].values

        probs = rankdata(data) / (data.size + 1.0)

#         scorrs = []
#         asymms_1 = []
#         asymms_2 = []
#         etpys = []
#         pcorrs = []
#         cdf_valss = []
#         etpy_cdf_valss = []
        for lag_step in lag_steps:
            probs_i, rolled_probs_i = roll_real_2arrs(probs, probs, lag_step)
            data_i, rolled_data_i = roll_real_2arrs(data, data, lag_step)

            # scorr.
            scorrs = np.sort(rolled_probs_i * probs_i)

            # asymms.
            asymms_1 = np.sort((probs_i + rolled_probs_i - 1.0) ** 3)
            asymms_2 = np.sort((probs_i - rolled_probs_i) ** 3)

            # ecop etpy.
            fill_bi_var_cop_dens(probs_i, rolled_probs_i, ecop_dens_arr)

            non_zero_idxs = ecop_dens_arr > 0

            dens = ecop_dens_arr[non_zero_idxs]

            etpy_arr = -(dens * np.log(dens))

            etpys = np.sort(etpy_arr)

            # pcorr.
            pcorrs = np.sort((rolled_data_i - data_i))

            # cdf vals.
            cdf_vals = np.arange(1.0, data_i.size + 1)
            cdf_vals /= cdf_vals.size + 1.0

            # etpy cdf_vals
            etpy_cdf_vals = np.arange(1.0, etpy_arr.size + 1)
            etpy_cdf_vals /= etpy_cdf_vals.size + 1.0

            # plot
            plt.figure(lag_axes[lag_step][0].number)

            lag_axes[lag_step][1][0, 0].plot(
                scorrs, cdf_vals, alpha=plt_alpha, color=sim_clrs[idx_lab])

            lag_axes[lag_step][1][1, 0].plot(
                asymms_1, cdf_vals, alpha=plt_alpha, color=sim_clrs[idx_lab])

            lag_axes[lag_step][1][1, 1].plot(
                asymms_2, cdf_vals, alpha=plt_alpha, color=sim_clrs[idx_lab])

            lag_axes[lag_step][1][0, 1].plot(
                etpys, etpy_cdf_vals, alpha=plt_alpha, color=sim_clrs[idx_lab])

            lag_axes[lag_step][1][0, 2].plot(
                pcorrs, cdf_vals, alpha=plt_alpha, color=sim_clrs[idx_lab])

            lag_axes[lag_step][1][0, 0].grid()
            lag_axes[lag_step][1][1, 0].grid()
            lag_axes[lag_step][1][1, 1].grid()
            lag_axes[lag_step][1][0, 1].grid()
            lag_axes[lag_step][1][0, 2].grid()
            lag_axes[lag_step][1][1, 2].grid()

            lag_axes[lag_step][1][0, 0].set_ylabel('Spearman correlation')

            lag_axes[lag_step][1][1, 0].set_xlabel('Lag steps')
            lag_axes[lag_step][1][1, 0].set_ylabel('Asymmetry (Type - 1)')

            lag_axes[lag_step][1][1, 1].set_xlabel('Lag steps')
            lag_axes[lag_step][1][1, 1].set_ylabel('Asymmetry (Type - 2)')

            lag_axes[lag_step][1][0, 1].set_ylabel('Entropy')

            lag_axes[lag_step][1][0, 2].set_xlabel('Lag steps')
            lag_axes[lag_step][1][0, 2].set_ylabel('Pearson correlation')

        #     lag_axes[lag_step][1][1, 2].set_xlabel('Nth orders')
        #     lag_axes[lag_step][1][1, 2].set_ylabel('Dist. Sum')

            plt.colorbar(
                mappable=cmap_mappable_beta,
                cax=lag_axes[lag_step][1][1, 2],
                orientation='horizontal',
                label='Relative Timing',
                alpha=plt_alpha,
                drawedges=False)

            lag_axes[lag_step][0].suptitle(f'lag_step:{lag_step}')

    for lag_step in lag_steps:
        plt.figure(lag_axes[lag_step][0].number)
        plt.show(block=True)
        print(lag_step)

    return


if __name__ == '__main__':

    _save_log_ = False
    if _save_log_:
        from datetime import datetime
        from std_logger import StdFileLoggerCtrl

        # save all console activity to out_log_file
        out_log_file = os.path.join(
            r'P:\Synchronize\python_script_logs\\%s_log_%s.log' % (
            os.path.basename(__file__),
            datetime.now().strftime('%Y%m%d%H%M%S')))

        log_link = StdFileLoggerCtrl(out_log_file)

    print('#### Started on %s ####\n' % time.asctime())
    START = timeit.default_timer()

    #==========================================================================
    # When in post_mortem:
    # 1. "where" to show the stack
    # 2. "up" move the stack up to an older frame
    # 3. "down" move the stack down to a newer frame
    # 4. "interact" start an interactive interpreter
    #==========================================================================

    if DEBUG_FLAG:
        try:
            main()

        except:
            import pdb
            pdb.post_mortem()

    else:
        main()

    STOP = timeit.default_timer()
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))

    if _save_log_:
        log_link.stop()
