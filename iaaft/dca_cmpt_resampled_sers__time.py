'''
@author: Faizan-Uni-Stuttgart

Apr 7, 2022

5:05:05 PM

'''
import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path

import pandas as pd

DEBUG_FLAG = False


def main():

    main_dir = Path(
        r'P:\Synchronize\IWS\Testings\fourtrans_practice\iaaft')

    main_dir /= r'test_spcorr_47'

    os.chdir(main_dir)

    data_dir = main_dir

    # .csv and .pkl allowed.
    in_df_path_patt = 'auto_sims_*.csv'

    sep = ';'
    # time_fmt = '%Y-%m-%d %H:%M:%S'
    time_fmt = '%Y-%m-%d'
    float_fmt = '%0.2f'

    # Can be .pkl or .csv.
    # out_fmt = '.pkl'
    out_fmt = '.csv'

    # min_counts correspond to the resolutions. Each resolution when
    # being resampled should have a min_count to get a non Na value.
    # This is because resample sum does not have a skipna flag.
    resample_ress = ['W']
    min_counts = [7]

    # In case of months, the resampling is slightly different than hours etc.
    # resample_ress = ['m']
    # min_counts = [None]

#     resample_types = ['mean']  # , 'min', 'max']
    resample_types = ['sum']

    # Applied to shift the entire time series by this offset.
    tdelta = pd.Timedelta(0, unit='h')

    out_dir = Path(r'resampled_series__time')
    #==========================================================================

    assert out_fmt in ('.csv', '.pkl')

    assert len(resample_ress) == len(min_counts)

    out_dir.mkdir(exist_ok=True, parents=True)

    for in_df_path in data_dir.glob(f'./{in_df_path_patt}'):

        print('Going through:', in_df_path)

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

        in_df.index += tdelta

        for resample_res, min_count in zip(resample_ress, min_counts):

            counts_df = in_df.resample(resample_res).count().astype(float)

            if resample_res == 'm':
                assert min_count is None, 'For months, min_count must be None!'

                min_count = counts_df.index.days_in_month.values.reshape(-1, 1)

            else:
                pass

            counts_df[counts_df < min_count] = float('nan')
            counts_df[counts_df >= min_count] = 1.0

            # assert counts_df.max().max() <= 1.0, counts_df.max().max()

            for resample_type in resample_types:

                resample_df = getattr(
                    in_df.resample(resample_res), resample_type)()

                resample_df *= counts_df

                # Another, very slow, way of doing this.
    #             resample_df = in_df.resample(resample_res).agg(
    #                 getattr(pd.Series, resample_type), skipna=False)

                out_name = (
                    f'{in_df_path.stem}__'
                    f'RR{resample_res}_RT{resample_type}{out_fmt}')

                out_path = out_dir / out_name

                print('Output:', out_path)

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
