'''
@author: Faizan-Uni-Stuttgart

Nov 4, 2020

12:59:59 PM

'''
import os
import time
import timeit
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

DEBUG_FLAG = False


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Testings\fourtrans_practice\shuff_phs_specs')
    os.chdir(main_dir)

    in_data_file = Path(r'precipitation_bw_1961_2015.csv')
    col = 'P5229'
    out_pref = 'ppt'

#     in_data_file = Path(r'neckar_norm_cop_infill_discharge_1961_2015_20190118.csv')
#     col = '3470'
#     out_pref = 'dis'

    sep = ';'

    beg_year = 1961
    end_year = 2015

    time_fmt = '%Y-%m-%d'

    data_df = pd.read_csv(in_data_file, sep=sep, index_col=0)[col]
    data_df.index = pd.to_datetime(data_df.index, format=time_fmt)

    window = 4  # leap year windows

#     vals_clss = {}
    for i, year in enumerate(range(beg_year, end_year, window)):

        sel_idxs = np.zeros(data_df.shape[0], dtype=bool)

        sel_idxs[(data_df.index.year >= year) & (data_df.index.year < year + window)] = 1

        vals = data_df.loc[sel_idxs].values[:-1]

        print(year, year + window - 1, vals.size)

        probs = rankdata(vals) / (vals.size + 1.0)

        ft = np.fft.rfft(probs)

        phs_spec = np.angle(ft)
        mag_spec = np.abs(ft)

        vals_cls = {}

        vals_cls['vals'] = vals
        vals_cls['probs'] = probs

        vals_cls['ft'] = ft
        vals_cls['phs_spec'] = phs_spec
        vals_cls['mag_spec'] = mag_spec

#         vals_clss[f'{year}_{year + 4}'] = vals_cls

        with open(f'{out_pref}_{col}_{i}.pkl', 'wb') as pkl_hdl:
            pickle.dump(vals_cls, pkl_hdl)

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
