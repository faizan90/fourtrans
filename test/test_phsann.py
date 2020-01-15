'''
@author: Faizan-Uni-Stuttgart

Dec 30, 2019

1:23:33 PM

'''
import os
import sys
import time
import timeit
from math import ceil
import traceback as tb
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

from fourtrans import PhaseAnnealing

DEBUG_FLAG = False

plt.ioff()

# has to be big enough to accomodate all plotted values
mpl.rcParams['agg.path.chunksize'] = 100000


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Testings\fourtrans_practice\phsann')
    os.chdir(main_dir)

    in_file_path = r'neckar_norm_cop_infill_discharge_1961_2015_20190118.csv'

    stn_no = '427'

    time_fmt = '%Y-%m-%d'

    sep = ';'

    beg_time = '1999-01-01'
    end_time = '1999-12-31'

    verbose = True

    sim_label = '1022'

    plt_show_flag = True
    plt_show_flag = False

    long_test_flag = True
#     long_test_flag = False

    scorr_flag = True
    asymm_type_1_flag = True
    asymm_type_2_flag = True
    ecop_dens_flag = True

#     scorr_flag = False
#     asymm_type_1_flag = False
#     asymm_type_2_flag = False
    ecop_dens_flag = False

    auto_init_temperature_flag = True
#     auto_init_temperature_flag = False

    # TODO: remove this flag!
    normalize_asymms_flag = True
#     normalize_asymms_flag = False

    lag_steps = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    ecop_bins = 20

    n_reals = 7
    outputs_dir = main_dir
    n_cpus = 'auto'

    if long_test_flag:
        initial_annealing_temperature = 0.0001
        temperature_reduction_ratio = 0.992
        update_at_every_iteration_no = 200
        maximum_iterations = int(2e5)
        maximum_without_change_iterations = 1000
        objective_tolerance = 1e-8
        objective_tolerance_iterations = 30

        temperature_lower_bound = 1e-6
        temperature_upper_bound = 1000.0
        max_search_attempts = 100
        n_iterations_per_attempt = 3000
        acceptance_lower_bound = 0.5
        acceptance_upper_bound = 0.8
        target_acpt_rate = 0.7
        ramp_rate = 2.0

    else:
        initial_annealing_temperature = 0.0001
        temperature_reduction_ratio = 0.99
        update_at_every_iteration_no = 20
        maximum_iterations = 100
        maximum_without_change_iterations = 50
        objective_tolerance = 1e-8
        objective_tolerance_iterations = 20

        temperature_lower_bound = 1e-6
        temperature_upper_bound = 1000.0
        max_search_attempts = 50
        n_iterations_per_attempt = 1000
        acceptance_lower_bound = 0.5
        acceptance_upper_bound = 0.8
        target_acpt_rate = 0.7
        ramp_rate = 2.0

    in_df = pd.read_csv(in_file_path, index_col=0, sep=sep)
    in_df.index = pd.to_datetime(in_df.index, format=time_fmt)

    in_ser = in_df.loc[beg_time:end_time, stn_no]

    in_vals = in_ser.values

    phsann_cls = PhaseAnnealing(verbose)

    phsann_cls.set_reference_data(in_vals)

    phsann_cls.set_objective_settings(
        scorr_flag,
        asymm_type_1_flag,
        asymm_type_2_flag,
        ecop_dens_flag,
        lag_steps,
        ecop_bins,
        normalize_asymms_flag)

    phsann_cls.set_annealing_settings(
        initial_annealing_temperature,
        temperature_reduction_ratio,
        update_at_every_iteration_no,
        maximum_iterations,
        maximum_without_change_iterations,
        objective_tolerance,
        objective_tolerance_iterations)

    if auto_init_temperature_flag:
        phsann_cls.set_annealing_auto_temperature_settings(
                temperature_lower_bound,
                temperature_upper_bound,
                max_search_attempts,
                n_iterations_per_attempt,
                acceptance_lower_bound,
                acceptance_upper_bound,
                target_acpt_rate,
                ramp_rate)

    phsann_cls.set_misc_settings(n_reals, outputs_dir, n_cpus)

    phsann_cls.prepare()

    phsann_cls.verify()

    phsann_cls.generate_realizations()

    ref_scorrs = phsann_cls._ref_scorrs
    ref_asymms_1 = phsann_cls._ref_asymms_1
    ref_asymms_2 = phsann_cls._ref_asymms_2

    reals = phsann_cls.get_realizations()

    sim_scorrss = []
    sim_asymmss_1 = []
    sim_asymmss_2 = []
    for i in range(n_reals):
        print(reals[i][11])
        sim_scorrss.append(reals[i][3])

        sim_asymmss_1.append(reals[i][4])

        sim_asymmss_2.append(reals[i][5])

    axes = plt.subplots(2, 2, figsize=(15, 15))[1]

    for i in range(n_reals):
        axes[0, 1].plot(lag_steps, sim_scorrss[i], alpha=0.3, color='k')

        axes[1, 0].plot(lag_steps, sim_asymmss_1[i], alpha=0.3, color='k')

        axes[1, 1].plot(lag_steps, sim_asymmss_2[i], alpha=0.3, color='k')

    axes[0, 1].plot(lag_steps, ref_scorrs, alpha=0.7, color='r')

    axes[1, 0].plot(lag_steps, ref_asymms_1, alpha=0.7, color='r')

    axes[1, 1].plot(lag_steps, ref_asymms_2, alpha=0.7, color='r')

    axes[0, 1].grid()
    axes[1, 0].grid()
    axes[1, 1].grid()

    if plt_show_flag:
        plt.show(block=False)

    else:
        plt.savefig(
            str(outputs_dir / f'{sim_label}_obj_cmpr.png'),
            bbox_inches='tight')

        plt.close()

    rows_cols = int(ceil(lag_steps.size ** 0.5))
    axes = plt.subplots(rows_cols, rows_cols, figsize=(15, 15))[1]

    row = 0
    col = 0
    probs = phsann_cls._ref_rnk / (phsann_cls._ref_rnk.size + 1)
    for i in range(lag_steps.size):
        rolled_probs = np.roll(probs, lag_steps[i])
        axes[row, col].scatter(probs, rolled_probs, alpha=0.4)
        axes[row, col].grid()
        axes[row, col].set_title(f'lag_step: {lag_steps[i]}')

        col += 1
        if not (col % rows_cols):
            row += 1
            col = 0

    if plt_show_flag:
        plt.show(block=False)

    else:
        plt.savefig(
            str(outputs_dir / f'{sim_label}_ref_probs_lags.png'),
            bbox_inches='tight')

        plt.close()

    for j in range(n_reals):
        rows_cols = int(ceil(lag_steps.size ** 0.5))
        axes = plt.subplots(rows_cols, rows_cols, figsize=(15, 15))[1]

        row = 0
        col = 0
        probs = reals[j][1] / (phsann_cls._ref_rnk.size + 1)
        for i in range(lag_steps.size):
            rolled_probs = np.roll(probs, lag_steps[i])
            axes[row, col].scatter(probs, rolled_probs, alpha=0.4)
            axes[row, col].grid()
            axes[row, col].set_title(f'sim: {j}, lag_step: {lag_steps[i]}')

            col += 1
            if not (col % rows_cols):
                row += 1
                col = 0

        if plt_show_flag:
            plt.show(block=False)

        else:
            plt.savefig(
                str(outputs_dir / f'{sim_label}_sim_{j}_probs_lags.png'),
                bbox_inches='tight')

            plt.close()

    plt.figure(figsize=(30, 10))
    for j in range(n_reals):
        plt.plot(reals[j][12], alpha=0.1, color='k')

    plt.ylim(0, plt.ylim()[1])

    plt.grid()

    if plt_show_flag:
        plt.show(block=False)

    else:
        plt.savefig(
            str(outputs_dir / f'{sim_label}_sim_tols.png'),
            bbox_inches='tight')

        plt.close()

    plt.figure(figsize=(30, 10))
    for j in range(n_reals):
        plt.plot(reals[j][13], alpha=0.1, color='k')

    plt.ylim(0, plt.ylim()[1])

    plt.grid()

    if plt_show_flag:
        plt.show(block=False)

    else:
        plt.savefig(
            str(outputs_dir / f'{sim_label}_sim_all_obj_vals.png'),
            bbox_inches='tight')

        plt.close()

    plt.figure(figsize=(30, 10))
    for j in range(n_reals):
        plt.plot(reals[j][15], alpha=0.1, color='k')

    plt.ylim(0, 1.0)

    plt.grid()

    if plt_show_flag:
        plt.show(block=False)

    else:
        plt.savefig(
            str(outputs_dir / f'{sim_label}_sim_acpt_rates.png'),
            bbox_inches='tight')

        plt.close()

    plt.figure(figsize=(30, 10))
    for j in range(n_reals):
        plt.plot(reals[j][16], alpha=0.1, color='k')

    plt.ylim(0, plt.ylim()[1])

    plt.grid()

    if plt_show_flag:
        plt.show(block=False)

    else:
        plt.savefig(
            str(outputs_dir / f'{sim_label}_sim_min_obj_vals.png'),
            bbox_inches='tight')

        plt.close()

    plt.figure(figsize=(30, 10))
    for j in range(n_reals):
        plt.plot(reals[j][17], alpha=0.1, color='k')

    plt.grid()

    if plt_show_flag:
        plt.show(block=False)

    else:
        plt.savefig(
            str(outputs_dir / f'{sim_label}_sim_all_phss.png'),
            bbox_inches='tight')

        plt.close()

    if plt_show_flag:
        plt.show()
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

    if _save_log_:
        log_link.stop()
