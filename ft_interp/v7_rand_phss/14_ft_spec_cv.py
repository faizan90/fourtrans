'''
@author: Faizan-Uni-Stuttgart

Nov 20, 2020

3:10:04 PM

'''
import os
import time
import timeit
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

plt.ioff()

DEBUG_FLAG = False


def main():

    main_dir = Path(
        r'P:\Synchronize\IWS\Testings\fourtrans_practice\multisite_phs_spec_corr\5min\v7')

    os.chdir(main_dir)

    suff = '__RR1D_RTsum'

    # First files is for observed stns.
    in_data_files = [
        Path(r'../neckar_1min_ppt_data_20km_buff_Y2009__RRD_RTsum.pkl'),
        Path(f'orig/ts_OK{suff}.csv'),
        Path(f'ifted/ts_OK{suff}.csv'), ]

    data_labels = ['REF', 'OK', 'FT']

    out_dirs = [data_file.parents[0] for data_file in in_data_files]

    sep = ';'

    fig_xlabel = 'Frequency'
    fig_ylabel = 'CV'

    beg_time = '2009-01-01 00:00:00'
    end_time = '2009-03-31 23:59:00'

    fig_size = (15, 7)

    dpi = 200

    for i in range(len(in_data_files)):
        print('Going through:', in_data_files[i])

        if in_data_files[i].suffix == '.csv':
            data_df = pd.read_csv(in_data_files[i], sep=sep, index_col=0)

        elif in_data_files[i].suffix == '.pkl':
            data_df = pd.read_pickle(in_data_files[i])

        else:
            raise NotImplementedError(
                f'Unknown extension of in_data_file: {in_data_files[i].suffix}!')

        data_df = data_df.loc[beg_time:end_time]

        if i == 0:
            data_df.dropna(axis=1, how='any', inplace=True)

        else:
            data_df.dropna(axis=0, how='any', inplace=True)

        assert np.all(np.isfinite(data_df.values))

        probs_df = data_df.rank(axis=0) / (data_df.shape[0] + 1)

        norms_df = pd.DataFrame(
            data=norm.ppf(probs_df.values), columns=data_df.columns)

        ft_df = pd.DataFrame(
            data=np.fft.rfft(norms_df, axis=0),
            columns=data_df.columns)

        mag_df = pd.DataFrame(data=np.abs(ft_df), columns=data_df.columns)

#         phs_df = pd.DataFrame(data=np.angle(ft_df), columns=data_df.columns)
#
#         phs_le_idxs = phs_df < 0
#
#         phs_df[phs_le_idxs] = (2 * np.pi) + phs_df[phs_le_idxs]

        mag_mean_df = mag_df.mean(axis=1)
        mag_std_df = mag_df.std(axis=1)

        mag_cv_df = mag_std_df / mag_mean_df

        plt.figure(figsize=fig_size)

        plt.bar(
            np.arange(mag_cv_df.size),
            mag_cv_df.values,
            width=1.0,
            alpha=0.8)

        plt.xlabel(fig_xlabel)
        plt.ylabel(fig_ylabel)

        plt.grid()
        plt.gca().set_axisbelow(True)

        plt.savefig(
            str(out_dirs[i] / f'mag_cv__{data_labels[i].lower()}{suff}.png'),
            bbox_inches='tight',
            dpi=dpi)

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
