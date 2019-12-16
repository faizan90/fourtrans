'''
@author: Faizan-Uni-Stuttgart

'''
import os
import timeit
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import YearLocator, DateFormatter
from pandas.plotting import register_matplotlib_converters

plt.ioff()
register_matplotlib_converters()


def main():

    main_dir = Path(
        r'P:\Synchronize\IWS\Testings\fourtrans_practice\similar'
        r'_dissimilar_series')

    os.chdir(main_dir)

    in_files = [
        'inn_daily_discharge.csv',
        'mulde_daily_discharge.csv',
        'neckar_daily_discharge_1961_2015.csv',
        'niedersachsen_daily_discharge.csv',
        'oesterreich_daily_discharge.csv']

    cmbd_df_path = r'combined_discharge_df.pkl'

    sep = ';'
    time_fmt = '%Y-%m-%d'
    time_freq = 'D'

    fig_size = (13, 7)
    plot_avail_flag = True

    dfs = []

    for in_file in in_files:
        df = pd.read_csv(in_file, sep=sep, index_col=0)

        df.index = pd.to_datetime(df.index, format=time_fmt)

        dfs.append(df)

        if plot_avail_flag:
            years = YearLocator(5)  # put a tick on x axis every N years
            yearsFmt = DateFormatter('%Y')  # how to show the year

            ppt_stn_cts_ser = df.count(axis=1)

            plt.figure(figsize=fig_size)

            plt.plot(ppt_stn_cts_ser.index, ppt_stn_cts_ser.values)

            plt.xlabel('Time (days)')
            plt.ylabel('Available stations')

            fig_name_suff = os.path.basename(in_file).rsplit('.', 1)[0]

            plt.title(
                f'Time series of active stations in {fig_name_suff}')

            plt.gca().xaxis.set_major_locator(years)
            plt.gca().xaxis.set_major_formatter(yearsFmt)

            plt.xlim(df.index[0], df.index[-1])

            plt.grid()

            plt.savefig(
                f'stns_availability_{fig_name_suff}.png',
                bbox_inches='tight')

            plt.close()

    df = None

    min_time = min([df.index.min() for df in dfs])
    max_time = max([df.index.max() for df in dfs])

    columns = pd.Index(set(np.concatenate([df.columns for df in dfs])))

    time_range = pd.date_range(min_time, max_time, freq=time_freq)

    out_df = pd.DataFrame(
        index=time_range,
        columns=columns,
        dtype=float)

    for df in dfs:
        out_df.update(df, overwrite=False, errors='ignore')

    dfs = None

    out_df.to_pickle(cmbd_df_path)

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
    START = timeit.default_timer()  # to get the runtime of the program

    main()

    STOP = timeit.default_timer()  # Ending time
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))

    if _save_log_:
        log_link.stop()
