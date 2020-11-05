'''
@author: Faizan-Uni-Stuttgart

Nov 5, 2020

3:31:03 PM

'''
import os
import time
import timeit
from pathlib import Path

import numpy as np
import pandas as pd
import netCDF4 as nc
import matplotlib.pyplot as plt

from faizpy import get_ns

plt.ioff()

DEBUG_FLAG = False


def main():

    main_dir = Path(
        r'P:\Synchronize\IWS\Testings\fourtrans_practice\ft_spatio_temporal_interps')

    os.chdir(main_dir)

    in_data_file = Path(r'precipitation.csv')
    in_data_crds_file = Path(r'precipitation_coords.csv')

    sep = ';'

    test_files = [
        Path(r'precipitation_interpolation_validation\obs\precipitation_kriging_1989-01-01_to_1992-12-30_1km_obs.nc'),
        Path(r'precipitation_interpolation_validation\combined_real_imag\precipitation_kriging_1989-01-01_to_1992-12-30_1km_ft.nc')]

    # Corresponds to test_files.
    test_file_labels = ['obs', 'ft']

    # Corresponds to test_files.
    nc_vars = [
        'OK',
        'OK']

    # Corresponds to test_files.
    time_var = 'time'

    beg_time = '1989-01-01'
    end_time = '1989-12-30'

    time_fmt = '%Y-%m-%d'

    out_dir = Path(r'precipitation_interpolation_validation/cmpr_figs_variance')

    # Selected post subsetting.
#     validation_cols = []
#     validation_cols = ['T3705', 'T1875', 'T5664', 'T1197']
    validation_cols = ['P3733', 'P3315', 'P3713', 'P3454']

    fig_size = (15, 7)
    dpi = 600

    fig_xlabel = 'Time (days)'
    fig_ylabel = 'Precipitation variance'

    fig_name_pref = 'ppt'

    assert 'ref' not in test_file_labels, 'label ref not allowed!'

    out_dir.mkdir(exist_ok=True)

    data_df = pd.read_csv(in_data_file, sep=sep, index_col=0)

    data_df.index = pd.to_datetime(data_df.index, format=time_fmt)

    crds_df = pd.read_csv(in_data_crds_file, sep=sep, index_col=0)

    data_df = data_df[crds_df.index]

    if validation_cols:
        data_df.drop(labels=validation_cols, axis=1, inplace=True)

    data_df = data_df.loc[beg_time:end_time]

    data_df.dropna(axis=1, how='any', inplace=True)

    variance_df = pd.DataFrame(data={'ref': data_df.var(axis=1)})

    for test_file, test_file_label, nc_var in zip(
        test_files, test_file_labels, nc_vars):

        print('Reading:', test_file)

        with nc.Dataset(test_file, mode='r') as nc_hdl:
            time_steps = nc.num2date(
                nc_hdl[time_var][:].data,
                nc_hdl[time_var].units,
                nc_hdl[time_var].calendar)

            ser = pd.Series(
                index=time_steps,
                data=np.nanvar(nc_hdl[nc_var][:, :, :], axis=(1, 2)))

        variance_df[test_file_label] = ser.loc[beg_time:end_time]

    ser = None

    plt.figure(figsize=fig_size)
    for variance_label in variance_df:
        print('Plotting', variance_label)

        if variance_label == 'ref':
            plt.plot(
                variance_df.index,
                variance_df[variance_label].values,
                alpha=0.7,
                label=variance_label,
                color='red',
                lw=1)

        else:
            ns = get_ns(variance_df['ref'].values, variance_df[variance_label].values)

            plt.plot(
                variance_df.index,
                variance_df[variance_label].values,
                alpha=0.7,
                label=f'{variance_label} (NS={ns:0.3f})',
                lw=1)

    plt.legend()

    plt.xlabel(fig_xlabel)
    plt.ylabel(fig_ylabel)

    plt.grid()

    out_fig_path = (
        out_dir /
        Path(f'{fig_name_pref}_cmpr_variance.png'))

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
