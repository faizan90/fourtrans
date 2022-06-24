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

DEBUG_FLAG = False


def main():

    main_dir = Path(__file__).parents[0]
    os.chdir(main_dir)

    sim_dir = r'test_spcorr_67'

    # Manual execution of each script.
    # There are scripts that require the output of the first as an input.
    scripts_to_run = [
        Path('daa_cmpr_props.py'),
        Path('dab_cmpr_props_ms.py'),
        Path('dac_plot_cross_corrs_combined.py'),
        Path('dba_cmpt_resampled_sers__space.py'),
        Path('dbb_plot_resampled_sers__space.py'),
        Path('dca_cmpt_resampled_sers__time.py'),
        Path('dcb_plot_resampled_sers__time.py'),
        Path('dea_cmpr_ann_cyc.py'),
        Path('dfa_plot_opt.py'),
        ]

    line_pref = 'main_dir /= '
    #==========================================================================

    for script_to_run in scripts_to_run:

        print('Going through:', script_to_run)

        # script_lines_orig = None
        with open(script_to_run, 'r') as script_hdl:
            script_lines = script_hdl.readlines()

            # script_lines_orig = [line for line in script_lines]

            line_idx = None
            for i, script_line in enumerate(script_lines):

                chg_line = script_line.strip()

                if (chg_line[:len(line_pref)] == line_pref):

                    if line_idx is not None:
                        raise AssertionError(
                            f'Found more than one line with same prefix!')

                    line_idx = i

            assert line_idx is not None, (
                'Could not find a line with the required prefix!')

            print(line_idx, script_lines[line_idx])

        with open(script_to_run, 'w') as script_hdl:
            pass

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
