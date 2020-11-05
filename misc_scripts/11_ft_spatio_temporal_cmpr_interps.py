'''
@author: Faizan-Uni-Stuttgart

Nov 5, 2020

2:36:34 PM

'''
import os
import time
import timeit
from pathlib import Path

import h5py
import pandas as pd
import matplotlib.pyplot as plt

from faizpy import get_ns

plt.ioff()

DEBUG_FLAG = False


def main():

    main_dir = Path(
        r'P:\Synchronize\IWS\Testings\fourtrans_practice\ft_spatio_temporal_interps\temperature_interpolation_validation')

    os.chdir(main_dir)

    # CSV file.
    ref_file = Path(r'../temperature_avg.csv')

    # HDF5 files.
    test_files = [
        Path(r'obs/ts_valid.h5'),
        Path(r'combined_real_imag/ts_valid.h5')]

    # Corresponds to test_files.
    test_file_labels = ['obs', 'ft']

    stn_labels = ['T3705', 'T1875', 'T1197']
#     stn_labels = ['P3733', 'P3315', 'P3713', 'P3454']

    # Corresponds to test_files.
    h5_vars = [
        'temperature_kriging_1989-01-01_to_1992-12-30_1km_obs/OK',
        'temperature_kriging_1989-01-01_to_1992-12-30_1km_ft/OK']

    # Corresponds to test_files.
    time_var = 'time/time_strs'

    beg_time = '1992-01-01'
    end_time = '1992-12-30'

    time_fmt = '%Y-%m-%d'
    time_fmt_h5 = '%Y%m%dT%H%M%S'

    sep = ';'

    fig_size = (15, 7)
    dpi = 600

    fig_xlabel = 'Time (days)'
    fig_ylabel = 'Temperature (C)'

    out_dir = Path('cmpr_figs')
    fig_name_pref = 'temp'

    out_dir.mkdir(exist_ok=True)

    ref_df = pd.read_csv(ref_file, index_col=0, sep=sep)[stn_labels]

    ref_df.index = pd.to_datetime(ref_df.index, format=time_fmt)

    ref_df = ref_df.loc[beg_time:end_time]

    test_dfs = {}
    for test_file, test_file_label, h5_var in zip(
        test_files, test_file_labels, h5_vars):

        print('Reading:', test_file)

        with h5py.File(test_file, mode='r', driver=None) as h5_hdl:
            h5_times = pd.to_datetime(
                h5_hdl[time_var][...], format=time_fmt_h5)

            data_ds = h5_hdl[h5_var]

            out_df = pd.DataFrame(
                index=h5_times, columns=stn_labels, dtype=float)

            for stn_label in stn_labels:
                out_df[stn_label] = data_ds[stn_label][:]

        test_dfs[test_file_label] = out_df.loc[beg_time:end_time]

    out_df = None

    for stn_label in stn_labels:
        print('Plotting', stn_label)

        plt.figure(figsize=fig_size)

        plt.plot(
            ref_df.index,
            ref_df[stn_label].values,
            alpha=0.7,
            label='ref',
            color='red',
            lw=2)

        for test_file_label in test_file_labels:

            ns = get_ns(
                ref_df[stn_label].values,
                test_dfs[test_file_label][stn_label].values)

            plt.plot(
                test_dfs[test_file_label].index,
                test_dfs[test_file_label][stn_label].values,
                alpha=0.7,
                label=f'{test_file_label} (NS={ns:0.3f})',
                lw=1)

        plt.legend()

        plt.xlabel(fig_xlabel)
        plt.ylabel(fig_ylabel)

        plt.grid()

        out_fig_path = (
            out_dir /
            Path(f'{fig_name_pref}_cmpr__{stn_label}.png'))

        plt.savefig(str(out_fig_path), bbox_inches='tight', dpi=dpi)

        plt.close()

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
