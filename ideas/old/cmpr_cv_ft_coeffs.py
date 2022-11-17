'''
@author: Faizan-Uni-Stuttgart

8 Jun 2020

14:35:05

'''
import os
import time
import timeit
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
import matplotlib.pyplot as plt

plt.ioff()

DEBUG_FLAG = True


def get_mag_and_phs_spec(data):

    ft = np.fft.rfft(data, axis=0)

    mag_spec = np.abs(ft)
    phs_spec = np.angle(ft)

    return mag_spec, phs_spec


def get_cumm_ft_corr(mag_spec):

    mag_spec_sq = mag_spec ** 2

    cumm_corr = np.cumsum(mag_spec_sq)
    cumm_corr /= cumm_corr[-1]

    return cumm_corr


def get_data_and_probs_cumm_corr(data):

    if (data.size % 2):
        data = data[:-1]

    probs = rankdata(data) / (data.size + 1.0)

#     norms = norm.ppf(probs)

    data_mag_spec, data_phs_spec = get_mag_and_phs_spec(data)

    probs_mag_spec, probs_phs_spec = get_mag_and_phs_spec(probs)

    data_cumm_corr = get_cumm_ft_corr(data_mag_spec[1:-1])
    probs_cumm_corr = get_cumm_ft_corr(probs_mag_spec[1:-1])

    return data_cumm_corr, probs_cumm_corr


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Testings\fourtrans_practice\phsann')
    os.chdir(main_dir)

    in_file = Path(r'neckar_norm_cop_infill_discharge_1961_2015_20190118.csv')

    steps_ulim = 365

    in_ser = pd.read_csv(in_file, sep=';', index_col=0)['420']

    in_ser.index = pd.to_datetime(in_ser.index, format='%Y-%m-%d')

    idx_labs = np.unique(in_ser.index.year)

    props = []
    for idx_lab in idx_labs:
        data = in_ser.loc[
            f'{idx_lab}-01-01':f'{idx_lab}-12-31'].values[:steps_ulim]

        probs = rankdata(data) / (data.size + 1.0)

        mag_spec = get_mag_and_phs_spec(probs)[0]

#         prop = get_cumm_ft_corr(mag_spec[1:-1])
        prop = mag_spec[1:-1]
#         periods = (prop.size * 2) / np.arange(1, prop.size + 1)

        props.append(prop)

    props = np.array(props)

    prop_mean = props.mean(axis=0)
    prop_std = props.std(axis=0)
    prop_cv = prop_std / prop_mean

    periods = (props.shape[1] * 2) / np.arange(1, props.shape[1] + 1)

#     for i in range(props.shape[0]):
#         plt.semilogx(periods[::-1], props[i], color='k', alpha=0.5)

    plt.semilogx(periods, prop_cv, color='k', alpha=0.5)

    plt.xlabel('Period')
    plt.ylabel('Property')

    plt.xlim(plt.xlim()[::-1])

    plt.grid()

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
