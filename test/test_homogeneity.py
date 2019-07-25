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
from pandas.plotting import register_matplotlib_converters

plt.ioff()
register_matplotlib_converters()


def main():

    main_dir = Path(r'P:\Synchronize\IWS\fourtrans_practice\test_homogeneity')
    os.chdir(main_dir)

    in_file_path = (
        r'P:\Synchronize\IWS\Discharge_data_longer_series\neckar_norm_'
        r'cop_infill_discharge_1961_2015_20190118\02_combined_station_'
        r'outputs\infilled_var_df_infill_stns.csv')

    stns = [420, 427, 3421, 3465, 3470, 454]

    cycle_steps = 365

    beg_idx = '1961-01-01'
    end_idx = '2015-12-31'

    out_fig_suff = 'neckar_'

#     in_file_path = (
#         r'P:\Synchronize\IWS\2015_Water_Balance_Peru\04_hydrological_'
#         r'data_prep\santa_q_daily_infill_20181031_no_zeros\02_combined_'
#         r'station_outputs\infilled_var_df_infill_stns.csv')
#
#     stns = ['CONDORCERRO', 'LA BALSA']
#
#     cycle_steps = 365
#
#     beg_idx = '1953-01-01'
#     end_idx = '2017-12-31'
#
#     out_fig_suff = 'peru_'

    kept_idxs_type = 'before'

    time_fmt = '%Y-%m-%d'

    fig_size = (19, 8)

    norm_flag = True
    line_alpha = 0.6

    assert kept_idxs_type in ('before', 'after')

    stns = [str(stn) for stn in stns]

    in_df = pd.read_csv(
        in_file_path,
        sep=';',
        index_col=0).loc[beg_idx:end_idx, stns]

    if in_df.shape[0] % 2:
        in_df = in_df.iloc[:-1]

    in_df.index = pd.to_datetime(in_df.index, format=time_fmt)

    assert not in_df.isna().sum().sum(), 'Invalid values!'

    n_steps = in_df.shape[0]

    n_kept_freqs_idx = n_steps // cycle_steps

    if kept_idxs_type == 'before':
        nkfi_str = f'{n_kept_freqs_idx - 1}'

    elif kept_idxs_type == 'after':
        nkfi_str = f'{n_steps - n_kept_freqs_idx}'

    else:
        raise NotImplementedError

    plt.figure(figsize=fig_size)

    for stn in stns:

        ft = np.fft.rfft(in_df[stn].values)

        ft[0] = 0.0

        hi_mag_idx = np.argmax(np.abs(ft)[1:]) + 1

        if kept_idxs_type == 'before':
            ft[n_kept_freqs_idx:] = 0

        elif kept_idxs_type == 'after':
            ft[:n_kept_freqs_idx] = 0

        else:
            raise NotImplementedError

        trunc_ser = pd.Series(data=np.fft.irfft(ft), index=in_df.index)

        ini_val = trunc_ser.values[0]

        if norm_flag:
            trunc_ser = (
                (trunc_ser - trunc_ser.values.min()) /
                (trunc_ser.max() - trunc_ser.min()))

        plt.plot(
            trunc_ser,
            alpha=line_alpha,
            label=f'{stn} ({ini_val:4.3E}, {hi_mag_idx})')

    plt.grid()
    plt.legend(framealpha=0.5)

    plt.xlabel('Time')
    plt.ylabel('Normalized IFT value')

    if norm_flag:
        norm_str = ''

    else:
        norm_str = 'non-'

    plt.title(
        f'Inverse Fourier transformed series after chopping off '
        f'freqeuncies greater than {cycle_steps - 1} steps ({norm_str}normed)\n'
        f'No. steps: {n_steps}, No. of waves kept: {nkfi_str}, '
        f'Kept indices type: {kept_idxs_type}')

    plt.savefig(
        str(
            f'{out_fig_suff}ft_homog__ncyc_{cycle_steps}__'
            f'{norm_str}normed__kit_{kept_idxs_type}'),
        bbox_inches='tight')
    plt.close()

#     plt.show()
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
