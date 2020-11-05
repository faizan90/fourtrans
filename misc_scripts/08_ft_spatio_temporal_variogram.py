'''
@author: Faizan-Uni-Stuttgart

Nov 4, 2020

5:49:11 PM

'''
import os
import time
import timeit
from pathlib import Path

import numpy as np
import pandas as pd

from spinterps import FitVariograms

DEBUG_FLAG = False


def main():

    main_dir = Path(
        r'P:\Synchronize\IWS\Testings\fourtrans_practice\ft_spatio_temporal_interps')

    os.chdir(main_dir)

    in_data_file = Path(r'precipitation.csv')
    in_crds_file = Path(r'precipitation_coords.csv')

    sep = ';'
    time_fmt = '%Y-%m-%d'

    beg_year = 1989
    end_year = 1992  # NOTE: skips last odd step.

    out_dir = Path(r'precipitation_interpolation_validation')

    min_valid_stns = 10

    # Selected post subsetting.
#     validation_cols = []  # ['T3705', 'T1875', 'T5664', 'T1197']
    validation_cols = ['P3733', 'P3315', 'P3713', 'P3454']

    mdr = 0.7
    perm_r_list = [1, 2]
    fit_vgs = ['Sph', 'Exp']
    fil_nug_vg = 'Nug'
    n_best = 1
    ngp = 5
    figs_flag = True

    n_cpus = 8

    out_dir.mkdir(exist_ok=True)

    data_df = pd.read_csv(in_data_file, sep=sep, index_col=0)

    data_df.index = pd.to_datetime(data_df.index, format=time_fmt)

    data_df = data_df.loc[f'{beg_year}':f'{end_year}'].iloc[:-1]

    data_df.dropna(axis=1, how='any', inplace=True)

    crds_df = pd.read_csv(in_crds_file, sep=sep, index_col=0)[['X', 'Y', 'Z']]

    crds_df = crds_df.loc[data_df.columns]

    crds_df.to_csv(Path(in_crds_file.stem + '_subset.csv'), sep=sep)
#     raise Exception

    if validation_cols:
        assert all([
            validation_col in crds_df.index
            for validation_col in validation_cols])

        crds_df = crds_df.loc[
            crds_df.index.difference(pd.Index(validation_cols))]

        data_df = data_df[crds_df.index]

    ft_df = pd.DataFrame(
        data=np.fft.rfft(data_df, axis=0), columns=data_df.columns)

    for part in ['real', 'imag']:
        fit_vg_cls = FitVariograms()

        part_df = pd.DataFrame(
            getattr(ft_df.values, part), columns=data_df.columns)

        part_df.to_csv(out_dir / f'{part}.csv', sep=sep)

        fit_vg_cls.set_data(part_df, crds_df, index_type='obj')

        fit_vg_cls.set_vg_fitting_parameters(
            mdr,
            perm_r_list,
            fil_nug_vg,
            ngp,
            fit_vgs,
            n_best)

        fit_vg_cls.set_misc_settings(n_cpus, min_valid_stns)

        fit_vg_cls.set_output_settings(out_dir / part, figs_flag)

        fit_vg_cls.verify()

        fit_vg_cls.fit_vgs()

        fit_vg_cls.save_fin_vgs_df()
        fit_vg_cls = None
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
