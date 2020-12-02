'''
@author: Faizan-Uni-Stuttgart

Nov 12, 2020

12:23:45 PM

'''
import os
import time
import timeit
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt

plt.ioff()

DEBUG_FLAG = False


def main():

    main_dir = Path(
        r'P:\Synchronize\IWS\Testings\fourtrans_practice\multisite_phs_spec_corr\5min\v7')

    os.chdir(main_dir)

    in_data_file = Path(r'../neckar_1min_ppt_data_20km_buff_Y2009__RRD_RTsum.pkl')

    sep = ';'

    suff = '__RR1D_RTsum'

    test_files = [
        Path(f'orig/orig{suff}.h5'),
        Path(f'ifted/ifted{suff}.h5')]

    # Corresponds to test_files.
    test_file_labels = ['OK', 'FT']

    # Corresponds to test_files.
    h5_vars = [
        f'orig{suff}/OK',
        f'ifted{suff}/OK']

    beg_time = '2009-01-01 00:00:00'
    end_time = '2009-03-31 23:59:00'

    time_fmt = '%Y-%m-%d %H:%M:%S'

    out_dir = Path(f'cmpr_figs__stats{suff}')

    fig_size = (15, 7)
    dpi = 200

    fig_xlabel = 'Time (1Day)'
    fig_ylabel_vrc = 'Precipitation variance'
    fig_ylabel_mean = 'Precipitation mean'
    fig_ylabel_min = 'Precipitation min'
    fig_ylabel_max = 'Precipitation max'

    fig_name_pref = 'ppt'

    assert 'ref' not in test_file_labels, 'label ref not allowed!'

    out_dir.mkdir(exist_ok=True)

    if in_data_file.suffix == '.csv':
        data_df = pd.read_csv(in_data_file, sep=sep, index_col=0)
        data_df.index = pd.to_datetime(data_df.index, format=time_fmt)

    elif in_data_file.suffix == '.pkl':
        data_df = pd.read_pickle(in_data_file)

    else:
        raise NotImplementedError(
            f'Unknown extension of in_data_file: {in_data_file.suffix}!')

    data_df = data_df.loc[beg_time:end_time]

    data_df.dropna(axis=1, how='any', inplace=True)

    variance_df = pd.DataFrame(data={'ref': data_df.var(axis=1)})

    cats = []
    variance_dfs = {'ref': pd.Series(data_df.var(axis=1))}
    means_dfs = {'ref': pd.Series(data_df.mean(axis=1))}
    mins_dfs = {'ref': pd.Series(data_df.min(axis=1))}
    maxs_dfs = {'ref': pd.Series(data_df.max(axis=1))}
    for test_file, test_file_label, h5_var in zip(
        test_files, test_file_labels, h5_vars):

        print('Reading:', test_file)

        with h5py.File(test_file, 'r') as h5_hdl:
            time_steps = pd.to_datetime(
                h5_hdl['time/time_strs'][:], format='%Y%m%dT%H%M%S')

            variance_df = None
            for cat in h5_hdl[h5_var]:
                vrc_ser = pd.Series(
                    index=time_steps,
                    data=np.var(h5_hdl[f'{h5_var}/{cat}'][:, :], axis=(1)))

                mean_ser = pd.Series(
                    index=time_steps,
                    data=np.mean(h5_hdl[f'{h5_var}/{cat}'][:, :], axis=(1)))

                min_ser = pd.Series(
                    index=time_steps,
                    data=np.min(h5_hdl[f'{h5_var}/{cat}'][:, :], axis=(1)))

                max_ser = pd.Series(
                    index=time_steps,
                    data=np.max(h5_hdl[f'{h5_var}/{cat}'][:, :], axis=(1)))

                if variance_df is None:
                    variance_df = pd.DataFrame(
                        index=vrc_ser.loc[beg_time:end_time].index)

                    means_df = pd.DataFrame(
                        index=mean_ser.loc[beg_time:end_time].index)

                    mins_df = pd.DataFrame(
                        index=min_ser.loc[beg_time:end_time].index)

                    maxs_df = pd.DataFrame(
                        index=max_ser.loc[beg_time:end_time].index)

                variance_df[cat] = vrc_ser.loc[beg_time:end_time]

                means_df[cat] = mean_ser.loc[beg_time:end_time]
                mins_df[cat] = min_ser.loc[beg_time:end_time]
                maxs_df[cat] = max_ser.loc[beg_time:end_time]

                if cat not in cats:
                    cats.append(cat)

            variance_dfs[test_file_label] = variance_df
            means_dfs[test_file_label] = means_df
            mins_dfs[test_file_label] = mins_df
            maxs_dfs[test_file_label] = maxs_df

    for cat in cats:
        print('cat:', cat)

        # Variances.
        plt.figure(figsize=fig_size)
        for variance_label in variance_dfs:
            if variance_label == 'ref':
                plt.plot(
                    variance_dfs[variance_label].index,
                    variance_dfs[variance_label].values,
                    alpha=0.7,
                    label=variance_label,
                    color='red',
                    lw=1)

            else:
                plt.plot(
                    variance_dfs[variance_label].index,
                    variance_dfs[variance_label][cat].values,
                    alpha=0.7,
                    label=f'{variance_label}',
                    lw=1)

        plt.legend()

        plt.xlabel(fig_xlabel)
        plt.ylabel(fig_ylabel_vrc)

        plt.grid()

        out_fig_path = (
            out_dir /
            Path(f'{fig_name_pref}_cmpr_variance_{cat}.png'))

        plt.savefig(str(out_fig_path), bbox_inches='tight', dpi=dpi)

        plt.close()

        # Means.
        plt.figure(figsize=fig_size)
        for means_label in means_dfs:
            if means_label == 'ref':
                plt.plot(
                    means_dfs[means_label].index,
                    means_dfs[means_label].values,
                    alpha=0.7,
                    label=means_label,
                    color='red',
                    lw=1)

            else:
                plt.plot(
                    means_dfs[means_label].index,
                    means_dfs[means_label][cat].values,
                    alpha=0.7,
                    label=f'{means_label}',
                    lw=1)

        plt.legend()

        plt.xlabel(fig_xlabel)
        plt.ylabel(fig_ylabel_mean)

        plt.grid()

        out_fig_path = (
            out_dir /
            Path(f'{fig_name_pref}_cmpr_means_{cat}.png'))

        plt.savefig(str(out_fig_path), bbox_inches='tight', dpi=dpi)

        plt.close()

        # Minimums.
        plt.figure(figsize=fig_size)
        for mins_label in mins_dfs:
            if mins_label == 'ref':
                plt.plot(
                    mins_dfs[mins_label].index,
                    mins_dfs[mins_label].values,
                    alpha=0.7,
                    label=mins_label,
                    color='red',
                    lw=1)

            else:
                plt.plot(
                    mins_dfs[mins_label].index,
                    mins_dfs[mins_label][cat].values,
                    alpha=0.7,
                    label=f'{mins_label}',
                    lw=1)

        plt.legend()

        plt.xlabel(fig_xlabel)
        plt.ylabel(fig_ylabel_min)

        plt.grid()

        out_fig_path = (
            out_dir /
            Path(f'{fig_name_pref}_cmpr_mins_{cat}.png'))

        plt.savefig(str(out_fig_path), bbox_inches='tight', dpi=dpi)

        plt.close()

        # Maximums.
        plt.figure(figsize=fig_size)
        for maxs_label in maxs_dfs:
            if maxs_label == 'ref':
                plt.plot(
                    maxs_dfs[maxs_label].index,
                    maxs_dfs[maxs_label].values,
                    alpha=0.7,
                    label=maxs_label,
                    color='red',
                    lw=1)

            else:
                plt.plot(
                    maxs_dfs[maxs_label].index,
                    maxs_dfs[maxs_label][cat].values,
                    alpha=0.7,
                    label=f'{maxs_label}',
                    lw=1)

        plt.legend()

        plt.xlabel(fig_xlabel)
        plt.ylabel(fig_ylabel_max)

        plt.grid()

        out_fig_path = (
            out_dir /
            Path(f'{fig_name_pref}_cmpr_maxs_{cat}.png'))

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
