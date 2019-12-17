'''
@author: Faizan-Uni-Stuttgart

'''
import os
import time
import timeit
from pathlib import Path

import numpy as np
import pandas as pd


def get_mag_phs_spec(arr):

    assert isinstance(arr, np.ndarray)
    assert arr.ndim == 1
    assert np.all(np.isfinite(arr))
    assert arr.size >= 3

    ft = np.fft.rfft(arr)

    mag_spec = np.abs(ft)
    phs_spec = np.angle(ft)

    return mag_spec[1:-1], phs_spec[1:-1]


def get_corr_max_corr(arr_1, arr_2):

    mag_spec_1, phs_spec_1 = get_mag_phs_spec(arr_1)
    mag_spec_2, phs_spec_2 = get_mag_phs_spec(arr_2)

    assert mag_spec_1.size == mag_spec_2.size
    assert phs_spec_1.size == phs_spec_2.size

    mag_specs_prod = mag_spec_1 * mag_spec_2

    denom_corrs = np.sqrt((mag_spec_1 ** 2).sum() * (mag_spec_2 ** 2).sum())

    max_corr = mag_specs_prod.sum() / denom_corrs

    corr = (
        (mag_specs_prod * np.cos(phs_spec_1 - phs_spec_2)).sum() /
        denom_corrs)

    assert -1 <= max_corr <= +1
    assert -1 <= corr <= +1

    return corr, max_corr


def main():

    main_dir = Path(os.getcwd())
    os.chdir(main_dir)

    df = pd.read_csv(
        r'P:\Synchronize\IWS\Colleagues_Students\Scheuring\miami_daily.csv',
        sep=';',
        index_col=0)

    df.replace(np.nan, 0.0, inplace=True)

    tst_dss = ['rt', 'trmm', 'persiann', 'ccs', 'cdr']
    ref_prec = df['prec'].values

    for tst_ds in tst_dss:
        tst_prec = df[tst_ds].values

        corr, max_corr = get_corr_max_corr(ref_prec, tst_prec)

        print('\n')
        print(tst_ds)
        print(corr, max_corr)
        print(np.corrcoef(ref_prec, tst_prec)[0, 1])

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

    try:
        main()

    except:
        import pdb
        pdb.post_mortem()

    STOP = timeit.default_timer()
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))

    if _save_log_:
        log_link.stop()
