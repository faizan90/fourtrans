'''
@author: Faizan-Uni-Stuttgart

Nov 11, 2020

11:52:34 AM

'''
import os
import time
import timeit
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from spinterps import FitVariograms

DEBUG_FLAG = False


def main():

    main_dir = Path(
        r'P:\Synchronize\IWS\Testings\fourtrans_practice\multisite_phs_spec_corr\5min\v7')

    os.chdir(main_dir)

    in_data_file = Path(r'../neckar_1min_ppt_data_20km_buff_Y2009__RR5min_RTsum.pkl')
    in_crds_file = Path(r'../metadata_ppt_gkz3_crds.csv')  # has X, Y cols

    sep = ';'
    time_fmt = '%Y-%m-%d %H:%M:%S'

    beg_time = '2009-01-01 00:00:00'
    end_time = '2009-03-31 23:59:00'

    out_dir = main_dir

    min_valid_stns = 10

    mdr = 0.5
    perm_r_list = [1, 2]
    fit_vgs = ['Exp']
    fil_nug_vg = 'Sph'
    n_best = 1
    ngp = 20
    figs_flag = True

    vg_vars = ['orig', 'data', ]  # 'phs', 'mag', 'sin', 'cos',

    n_cpus = 8

    out_dir.mkdir(exist_ok=True)

    if in_data_file.suffix == '.csv':
        data_df = pd.read_csv(in_data_file, sep=sep, index_col=0)
        data_df.index = pd.to_datetime(data_df.index, format=time_fmt)

    elif in_data_file.suffix == '.pkl':
        data_df = pd.read_pickle(in_data_file)

    else:
        raise NotImplementedError(
            f'Unknown extension of in_data_file: {in_data_file.suffix}!')

    data_df = data_df.loc[f'{beg_time}':f'{end_time}']

    if data_df.shape[0] % 2:
        data_df = data_df.iloc[:-1, :]
        print('Dropped last record in data_df!')

    data_df.dropna(axis=1, how='any', inplace=True)

    crds_df = pd.read_csv(in_crds_file, sep=sep, index_col=0)[['X', 'Y']]

    crds_df = crds_df.loc[data_df.columns]

    probs_df = data_df.rank(axis=0) / (data_df.shape[0] + 1)

    norms_df = pd.DataFrame(
        data=norm.ppf(probs_df.values), columns=data_df.columns)

    ft_df = pd.DataFrame(
        data=np.fft.rfft(norms_df, axis=0),
        columns=data_df.columns)

    mag_df = pd.DataFrame(data=np.abs(ft_df), columns=data_df.columns)

    phs_df = pd.DataFrame(data=np.angle(ft_df), columns=data_df.columns)

    phs_le_idxs = phs_df < 0

    phs_df[phs_le_idxs] = (2 * np.pi) + phs_df[phs_le_idxs]

    for part in vg_vars:

        (out_dir / part).mkdir(exist_ok=True)

        index_type = 'obj'

        if part == 'mag':
            part_df = mag_df

        elif part == 'phs':
            part_df = phs_df

        elif part == 'cos':
            part_df = pd.DataFrame(
                data=np.cos(phs_df), columns=data_df.columns)

        elif part == 'sin':
            part_df = pd.DataFrame(
                data=np.sin(phs_df), columns=data_df.columns)

        elif part == 'data':
            part_df = data_df.copy()

            part_df.values[:] = np.sort(part_df.values, axis=0)

            index_type = 'date'

#             part_df = part_df.iloc[-2:]

        elif part == 'orig':
            part_df = data_df.copy()
            index_type = 'date'

        else:
            raise ValueError(f'Undefined: {part}!')

        part_df.to_csv(out_dir / f'{part}/{part}.csv', sep=sep)

#         continue

        fit_vg_cls = FitVariograms()

        fit_vg_cls.set_data(part_df, crds_df, index_type=index_type)

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
