'''
@author: Faizan-Uni-Stuttgart

Dec 30, 2019

1:23:33 PM

'''
import os
import time
import timeit
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from fourtrans import PhaseAnnealing

DEBUG_FLAG = False

plt.ioff()


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

    sim_label = '1004'

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

    lag_steps = np.array([1, 2, 3, 4, 5])
    ecop_bins = 20

    n_reals = 1
    outputs_dir = main_dir
    n_cpus = 1  # 'auto'

    if long_test_flag:
        initial_annealing_temperature = 0.0001
        temperature_reduction_ratio = 0.99
        update_at_every_iteration_no = 200
        maximum_iterations = int(1e5)
        maximum_without_change_iterations = 500
        objective_tolerance = 1e-8
        objective_tolerance_iterations = 20

        temperature_lower_bound = 1e-6
        temperature_upper_bound = 100.0
        max_search_attempts = 100
        n_iterations_per_attempt = 3000
        acceptance_lower_bound = 0.1
        acceptance_upper_bound = 0.9

    else:
        initial_annealing_temperature = 0.0001
        temperature_reduction_ratio = 0.99
        update_at_every_iteration_no = 20
        maximum_iterations = 100
        maximum_without_change_iterations = 50
        objective_tolerance = 1e-8
        objective_tolerance_iterations = 20

        temperature_lower_bound = 1e-6
        temperature_upper_bound = 10.0
        max_search_attempts = 20
        n_iterations_per_attempt = 1000
        acceptance_lower_bound = 0.6
        acceptance_upper_bound = 0.8

    in_df = pd.read_csv(in_file_path, index_col=0, sep=sep)
    in_df.index = pd.to_datetime(in_df.index, format=time_fmt)

    in_ser = in_df.loc[beg_time:end_time, stn_no]

    in_vals = in_ser.values

    assert np.all(np.isfinite(in_vals))

    phsann_cls = PhaseAnnealing(verbose)

    phsann_cls.set_reference_data(in_vals)

    phsann_cls.set_objective_settings(
        scorr_flag,
        asymm_type_1_flag,
        asymm_type_2_flag,
        ecop_dens_flag,
        lag_steps,
        ecop_bins)

    phsann_cls.set_annealing_settings(
        initial_annealing_temperature,
        temperature_reduction_ratio,
        update_at_every_iteration_no,
        maximum_iterations,
        maximum_without_change_iterations,
        objective_tolerance,
        objective_tolerance_iterations)

    phsann_cls.set_annealing_auto_temperature_settings(
            auto_init_temperature_flag,
            temperature_lower_bound,
            temperature_upper_bound,
            max_search_attempts,
            n_iterations_per_attempt,
            acceptance_lower_bound,
            acceptance_upper_bound)

    phsann_cls.set_misc_settings(n_reals, outputs_dir, n_cpus)

    phsann_cls.prepare()

    phsann_cls.verify()

    phsann_cls.generate_realizations()

    ref_scorrs = phsann_cls._ref_scorrs
    ref_asymms_1 = phsann_cls._ref_asymms_1
    ref_asymms_2 = phsann_cls._ref_asymms_2

    sim_scorrss = []
    sim_asymmss_1 = []
    sim_asymmss_2 = []

    for i in range(n_reals):
        print(phsann_cls._alg_reals[i][10])
        if scorr_flag:
            sim_scorrss.append(phsann_cls._alg_reals[i][3])

        if asymm_type_1_flag:
            sim_asymmss_1.append(phsann_cls._alg_reals[i][4])

        if asymm_type_2_flag:
            sim_asymmss_2.append(phsann_cls._alg_reals[i][5])

    axes = plt.subplots(2, 2, figsize=(15, 15))[1]

    for i in range(n_reals):
        if scorr_flag:
            axes[0, 1].plot(lag_steps, sim_scorrss[i], alpha=0.3, color='k')

        if asymm_type_1_flag:
            axes[1, 0].plot(lag_steps, sim_asymmss_1[i], alpha=0.3, color='k')

        if asymm_type_2_flag:
            axes[1, 1].plot(lag_steps, sim_asymmss_2[i], alpha=0.3, color='k')

    if scorr_flag:
        axes[0, 1].plot(lag_steps, ref_scorrs, alpha=0.7, color='r')

    if asymm_type_1_flag:
        axes[1, 0].plot(lag_steps, ref_asymms_1, alpha=0.7, color='r')

    if asymm_type_2_flag:
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
#         print(row, col)
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
        probs = phsann_cls._alg_reals[j][1] / (phsann_cls._ref_rnk.size + 1)
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
        try:
            plt.plot(phsann_cls._alg_reals[j][11], alpha=0.1, color='k')

        except:
            pass

    plt.grid()

    if plt_show_flag:
        plt.show(block=False)

    else:
        plt.savefig(
            str(outputs_dir / f'{sim_label}_sim_{j}_tols.png'),
            bbox_inches='tight')

    plt.figure(figsize=(30, 10))
    for j in range(n_reals):
        try:
            plt.plot(phsann_cls._alg_reals[j][12], alpha=0.1, color='k')

        except:
            pass

    plt.grid()

    if plt_show_flag:
        plt.show(block=False)

    else:
        plt.savefig(
            str(outputs_dir / f'{sim_label}_sim_{j}_obj_vals.png'),
            bbox_inches='tight')

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
            import pdb
            pdb.post_mortem()

    else:
        main()

    STOP = timeit.default_timer()
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))

    if _save_log_:
        log_link.stop()
