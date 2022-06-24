'''
@author: Faizan-Uni-Stuttgart

Apr 7, 2022

4:55:28 PM

'''
import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path
from fnmatch import fnmatch

import pandas as pd

DEBUG_FLAG = False


def main():

    main_dir = Path(
        r'P:\Synchronize\IWS\Testings\fourtrans_practice\iaaft')

    main_dir /= r'test_spcorr_66'

    os.chdir(main_dir)

    data_dir = main_dir

    data_patt = 'cross_sims_*.csv'

    sep = ';'
    float_fmt = '%0.2f'

    ignore_col_patt = '*cp*'

    # Can be .pkl or .csv.
    # out_fmt = '.pkl'
    out_fmt = '.csv'

    # time_fmt = '%Y-%m-%dT%H:%M:%S'
    time_fmt = '%Y-%m-%d %H:%M:%S'

#     resample_types = ['mean']  # , 'min', 'max']
    resample_types = ['sum']

    out_dir = Path('resampled_series__space')
    #==========================================================================

    out_dir.mkdir(exist_ok=True)

    for in_df_path in data_dir.glob(data_patt):

        print('Going through:', in_df_path.name)

        if in_df_path.suffix == '.csv':
            in_df = pd.read_csv(in_df_path, sep=sep, index_col=0)

            in_df.index = pd.to_datetime(in_df.index, format=time_fmt)

        elif in_df_path.suffix == '.pkl':
            in_df = pd.read_pickle(in_df_path)

        else:
            raise NotImplementedError(
                f'Unknown file extension: {in_df_path.suffix}!')

        assert isinstance(in_df, pd.DataFrame)
        assert isinstance(in_df.index, pd.DatetimeIndex)

        drop_cols = [
            col for col in in_df.columns if fnmatch(col, ignore_col_patt)]

        in_df = in_df.drop(drop_cols, axis=1)

        counts_ser = in_df.count(axis=1)

        nan_idxs = counts_ser.values < in_df.shape[1]

        print(f'{nan_idxs.sum()} NaNs out of {in_df.shape[0]} records!')

        for resample_type in resample_types:
            resample_df = getattr(in_df, resample_type)(axis=1)

            resample_df.loc[nan_idxs] = float('nan')

            out_name = f'{in_df_path.stem}__RT{resample_type}{out_fmt}'

            out_path = out_dir / out_name

            if out_fmt == '.csv':
                resample_df.to_csv(
                    out_path,
                    sep=sep,
                    date_format=time_fmt,
                    float_format=float_fmt)

            elif out_fmt == '.pkl':
                resample_df.to_pickle(out_path)

            else:
                raise NotImplementedError(
                    f'Unknown file extension: {out_fmt}!')

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
