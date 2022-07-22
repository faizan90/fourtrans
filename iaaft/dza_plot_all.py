'''
@author: Faizan-Uni-Stuttgart

Jun 24, 2022

12:26:22 PM

'''
import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path
from importlib import import_module

from pathos.multiprocessing import ProcessPool

DEBUG_FLAG = False


def main():

    # Has to be the directory of the scripts.
    main_dir = Path(__file__).parents[0]
    os.chdir(main_dir)

    # Whatever the name. It is changed to this.
    sim_dir = r'holy_grail_2_02'

    # There are scripts that require the output of the first as an input.
    # This is specified in sequences_to_run.
    scripts_to_run = (
        Path('daa_cmpr_props.py'),
        Path('dab_cmpr_props_ms.py'),
        Path('dac_plot_cross_corrs_combined.py'),
        Path('dad_plot_cross_asymms_combined.py'),
        Path('dae_cmpr_props_asymms_marginals.py'),
        Path('dba_cmpt_resampled_sers__space.py'),
        Path('dca_cmpt_resampled_sers__time.py'),
        Path('dea_cmpr_ann_cyc.py'),
        Path('dfa_plot_opt.py'),

        Path('dbb_plot_resampled_sers__space.py'),
        Path('dcb_plot_resampled_sers__time.py'),
        )

    sequences_to_run = (
        list(range(len(scripts_to_run) - 2)),
        list(range(len(scripts_to_run) - 2, len(scripts_to_run))),
        )

    n_cpus = 8

    line_pref = 'main_dir /= '
    #==========================================================================

    for script_to_run in scripts_to_run:

        print('Going through:', script_to_run)

        with open(script_to_run, 'r') as script_hdl:
            script_lines = script_hdl.readlines()

            line_idx = None
            for i, script_line in enumerate(script_lines):

                chg_line = script_line.strip()

                if (chg_line[:len(line_pref)] == line_pref):

                    if line_idx is not None:
                        raise AssertionError(
                            f'Found more than one line with the same prefix!')

                    line_idx = i

            assert line_idx is not None, (
                'Could not find a line with the required prefix!')

        print('old:', line_idx, script_lines[line_idx])

        spaces = script_lines[line_idx].split(line_pref)[0]

        script_lines[line_idx] = f"{spaces}{line_pref}r'{sim_dir}'\n"

        print('new:', line_idx, script_lines[line_idx])

        with open(script_to_run, 'w') as script_hdl:
            for script_line in script_lines:
                script_hdl.write(script_line)
    #==========================================================================

    n_cpus = min(len(scripts_to_run), n_cpus)

    if n_cpus == 1:
        pass

    else:
        mp_pool = ProcessPool(n_cpus)

    for sequence_to_run in sequences_to_run:
        args_gen = (
            (main_dir, scripts_to_run[i].name) for i in sequence_to_run)

        if n_cpus == 1:
            for args in args_gen:
                run_script(args)

        else:
            list(mp_pool.imap(run_script, args_gen, chunksize=1))

    mp_pool.close()
    mp_pool.join()
    return


def run_script(args):

    old_cwd = os.getcwd()

    module_dir, script_file_name, = args

    os.chdir(module_dir)

    getattr(import_module(script_file_name.rsplit('.', 1)[0]), 'main')()

    os.chdir(old_cwd)
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
