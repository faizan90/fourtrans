'''
@author: Faizan-Uni-Stuttgart

Jul 21, 2022

11:14:27 AM

'''
import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

DEBUG_FLAG = False


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Testings\hydmod\iaaft_sims')
    os.chdir(main_dir)

    iaaft_sim_dir = Path(r'P:\Synchronize\IWS\Testings\fourtrans_practice\iaaft\test_dis_01')

    # Some hardcodded file_names.
    in_files = iaaft_sim_dir.glob('cross_sims_*.csv')

    beg_date = '2001-01-01'
    end_date = '2015-12-30'
    time_fmt = '%Y-%m-%d'

    sep = ';'

    out_file_name = f'{iaaft_sim_dir.name}.h5'
    out_dir = Path(r'hydmod_tss')
    #==========================================================================

    out_dir.mkdir(exist_ok=True)

    first_sim_flag = True
    for in_file in in_files:

        assert Path(in_file).exists(), f'{in_file} does not exist!'

        print('Going through:', in_file)

        out_file = out_dir / out_file_name

        cnvt_text_to_h5_iaaft(
            in_file,
            out_file,
            beg_date,
            end_date,
            time_fmt,
            sep,
            first_sim_flag)

        first_sim_flag = False

    return


def cnvt_text_to_h5_iaaft(
        in_file, out_file, beg_date, end_date, time_fmt, sep, first_sim_flag):

    in_file, out_file = Path(in_file), Path(out_file)

    in_df = pd.read_csv(in_file, sep=sep, index_col=0)
    in_df.index = pd.to_datetime(in_df.index, format=time_fmt)

    in_df = in_df.loc[beg_date:end_date]

    out_data_dict = {}
    out_rows_dict = {}
    out_cols_dict = {}
    out_area_ratios_dict = {}

    for i, stn in enumerate(in_df.columns):
        out_data_dict[str(stn)] = pd.DataFrame(
            data=in_df[stn].values,
            index=in_df.index,
            columns=[0],
            dtype=np.float64)

        out_rows_dict[str(stn)] = np.array([0], dtype=np.int64)
        out_cols_dict[str(stn)] = np.array([i], dtype=np.int64)
        out_area_ratios_dict[str(stn)] = np.array([1.0], dtype=np.float64)

    if first_sim_flag:
        write_mode = 'w'

    else:
        write_mode = 'a'

    out_hdl = h5py.File(str(out_file), mode=write_mode, driver=None)

    if first_sim_flag:
        h5_times = in_df.index.strftime('%Y%m%dT%H%M%S')
        h5_str_dt = h5py.special_dtype(vlen=str)

        time_grp = out_hdl.create_group('time')
        time_strs_ds = time_grp.create_dataset(
            'time_strs',
            (h5_times.shape[0],),
            dtype=h5_str_dt)

        time_strs_ds[:] = h5_times

    stem_grp = out_hdl.create_group(in_file.stem)

    if first_sim_flag:
        out_hdl.create_group('rows')
        out_hdl.create_group('cols')
        out_hdl.create_group('rel_itsctd_area')

    for stn in in_df.columns:
        stem_grp[str(stn)] = out_data_dict[str(stn)].values

        if first_sim_flag:
            out_hdl[f'rows/{stn}'] = out_rows_dict[str(stn)]
            out_hdl[f'cols/{stn}'] = out_cols_dict[str(stn)]
            out_hdl[f'rel_itsctd_area/{stn}'] = out_area_ratios_dict[str(stn)]

    out_hdl.flush()
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
