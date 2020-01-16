'''
@author: Faizan-Uni-Stuttgart

Dec 30, 2019

1:23:33 PM

'''
import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

from fourtrans import PhaseAnnealing, PhaseAnnealingPlot

DEBUG_FLAG = True

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

    sim_label = '1026'

    h5_name = 'phsann.h5'

    gen_rltzns_flag = True
#     gen_rltzns_flag = False

    plt_flag = True
#     plt_flag = False

    long_test_flag = True
    long_test_flag = False

    scorr_flag = True
    asymm_type_1_flag = True
    asymm_type_2_flag = True
    ecop_dens_flag = True

#     scorr_flag = False
#     asymm_type_1_flag = False
#     asymm_type_2_flag = False
    ecop_dens_flag = False

    auto_init_temperature_flag = True
    auto_init_temperature_flag = False

    lag_steps = np.array([1, 2, 3, 4, 5, 6, 7])
    ecop_bins = 20

    n_reals = 3
    outputs_dir = main_dir / sim_label
    n_cpus = 1  # 'auto'

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

    if gen_rltzns_flag:
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
            ecop_bins)

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

        phsann_cls.save_realizations()

    if plt_flag:
        phsann_plt_cls = PhaseAnnealingPlot(verbose)

        phsann_plt_cls.set_input(outputs_dir / h5_name)

        phsann_plt_cls.set_output(outputs_dir)

        phsann_plt_cls.verify()

        phsann_plt_cls.plot_opt_state_vars()

        phsann_plt_cls.plot_comparison()

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
