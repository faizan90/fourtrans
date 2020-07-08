'''
@author: Faizan-Uni-Stuttgart

6 Jul 2020

09:25:35

'''
import os
import time
import timeit
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.ioff()

DEBUG_FLAG = True

cos_var_types = (
    'cosines_product',
    'cosines_minimum',
    'cosines_maximum',
    'cosines_mean')


def get_cos_phs_var(phss, var_cls):

    idxs = np.arange(phss.shape[1])

    combs = combinations(idxs, 2)

    temp = []
    for comb in combs:
        temp.append(np.cos(phss[:, comb[0]] - phss[:, comb[1]]))

    temp = np.array(temp)

    if var_cls.var_type == cos_var_types[0]:
        temp = temp.prod(axis=0)

    elif var_cls.var_type == cos_var_types[1]:
        temp = temp.min(axis=0)

    elif var_cls.var_type == cos_var_types[2]:
        temp = temp.max(axis=0)

    elif var_cls.var_type == cos_var_types[3]:
        temp = temp.mean(axis=0)

    else:
        raise NotImplementedError(f'Unknown var_type: {var_cls.var_type}')

    return temp.reshape(-1, 1)


def get_df(var_cls):

    in_df = pd.read_csv(
        var_cls.in_file, sep=var_cls.sep, index_col=0)[var_cls.stns]

    in_df.index = pd.to_datetime(in_df.index, format=var_cls.time_fmt)

    in_df = in_df.loc[var_cls.beg_time:var_cls.end_time, :]

    if var_cls.lag:
        lag_n_steps = in_df.shape[0] - (var_cls.lag * in_df.shape[1])

        assert lag_n_steps > 1

        lag_arr = np.empty((lag_n_steps, in_df.shape[1]), dtype=np.float64)

        for i in range(lag_arr.shape[1]):
            lag_arr[:, i] = in_df.values[i * var_cls.lag:lag_n_steps + i * var_cls.lag, i]

        lag_df = pd.DataFrame(index=in_df.index[:lag_n_steps], data=lag_arr)

        in_df = lag_df

    if in_df.shape[0] % 2:
        in_df = in_df.iloc[:-1, :]

    return in_df


def get_cos_vars(var_cls):

    in_fts = np.fft.rfft(var_cls.df.values, axis=0)

    in_periods = ((var_cls.df.shape[0]) / (np.arange(1, in_fts.shape[0])))

    in_mags = np.abs(in_fts)[1:]
    in_phss = np.angle(in_fts)[1:]

    cos_vars = get_cos_phs_var(in_phss, var_cls)

    if var_cls.var_type in cos_var_types:
        pass

    else:
        raise NotImplementedError(f'Unknown var_type: {var_cls.var_type}')

    if var_cls.mag_flag:
        cos_vars *= in_mags.prod(axis=1).reshape(-1, 1)

    cos_vars_cumsum = cos_vars.cumsum()

    if var_cls.norm_flag:
        if var_cls.mag_flag:
            norm_val = (((in_mags ** 2).sum(axis=0).prod()) ** 0.5)

        else:
            if var_cls.var_type in cos_var_types:
                norm_val = cos_vars_cumsum.size

            else:
                raise NotImplementedError(
                    f'Unknown var_type: {var_cls.var_type}')

        cos_vars_cumsum /= norm_val

    return in_periods, cos_vars_cumsum


class VarCls:
    pass


def main():

    main_dir = Path(
        r'P:\Synchronize\IWS\Testings\fourtrans_practice\simult_exts_detection')

    os.chdir(main_dir)

    sep = ';'

    time_fmt = '%Y-%m-%d'

    beg_time = '1971-01-01'
    end_time = '1975-12-31'

    norm_flag = True
    mag_flag = True
    lag = 1

    fig_size = (10, 7)
#     fig_name = 'mag_cos_prod_cumsum.png'

    s1_cls = VarCls()
    s1_cls.in_file = Path(
        r'neckar_norm_cop_infill_discharge_1961_2015_20190118.csv')

    s1_cls.var_type = 'cosines_minimum'
    s1_cls.norm_flag = norm_flag
    s1_cls.mag_flag = mag_flag
    s1_cls.lag = 0
    s1_cls.sep = sep
    s1_cls.time_fmt = time_fmt
    s1_cls.beg_time = beg_time
    s1_cls.end_time = end_time
    s1_cls.stns = ['420', '454']  # , '427']
    s1_cls.label = '_'.join(s1_cls.stns)
    s1_cls.df = get_df(s1_cls)
    s1_cls.periods, s1_cls.cos_vars_cumsum = get_cos_vars(s1_cls)

#     s2_cls = VarCls()
#     s2_cls.var_type = 'cosines_product'
#     s2_cls.norm_flag = norm_flag
#     s2_cls.mag_flag = mag_flag
#     s2_cls.lag = lag
#     s2_cls.in_file = Path(r'oesterreich_daily_discharge.csv')
#     s2_cls.sep = sep
#     s2_cls.time_fmt = time_fmt
#     s2_cls.beg_time = beg_time
#     s2_cls.end_time = end_time
#     s2_cls.stns = ['201525', '205765', '207852']
#     s2_cls.label = '_'.join(s2_cls.stns)
#     s2_cls.df = get_df(s2_cls)
#     s2_cls.periods, s2_cls.cos_vars_cumsum = get_cos_vars(s2_cls)

    s2_cls = VarCls()
    s2_cls.var_type = 'cosines_minimum'
    s2_cls.norm_flag = norm_flag
    s2_cls.mag_flag = mag_flag
    s2_cls.lag = 3
    s2_cls.in_file = Path(
        r'neckar_norm_cop_infill_discharge_1961_2015_20190118.csv')

    s2_cls.sep = sep
    s2_cls.time_fmt = time_fmt
    s2_cls.beg_time = beg_time
    s2_cls.end_time = end_time
    s2_cls.stns = ['420', '454']  # , '427']
    s2_cls.label = '_'.join(s2_cls.stns)
    s2_cls.df = get_df(s2_cls)
    s2_cls.periods, s2_cls.cos_vars_cumsum = get_cos_vars(s2_cls)

#     s3_cls = VarCls()
#     s3_cls.in_file = Path(r'oesterreich_daily_discharge.csv')
#     s3_cls.var_type = 'cosines_product'
#     s3_cls.norm_flag = norm_flag
#     s3_cls.mag_flag = mag_flag
#     s3_cls.lag = lag
#     s3_cls.sep = sep
#     s3_cls.time_fmt = time_fmt
#     s3_cls.beg_time = beg_time
#     s3_cls.end_time = end_time
#     s3_cls.stns = ['207787', '205914', '201574']
#     s3_cls.label = '_'.join(s3_cls.stns)
#     s3_cls.df = get_df(s3_cls)
#     s3_cls.periods, s3_cls.cos_vars_cumsum = get_cos_vars(s3_cls)

#     s3_cls = VarCls()
#     s3_cls.in_file = Path(
#         r'neckar_norm_cop_infill_discharge_1961_2015_20190118.csv')
#
#     s3_cls.var_type = 'cosines_maximum'
#     s3_cls.norm_flag = norm_flag
#     s3_cls.mag_flag = mag_flag
#     s3_cls.sep = sep
#     s3_cls.lag = lag
#     s3_cls.time_fmt = time_fmt
#     s3_cls.beg_time = beg_time
#     s3_cls.end_time = end_time
#     s3_cls.stns = ['420', '427', '454']
#     s3_cls.label = '_'.join(s3_cls.stns)
#     s3_cls.df = get_df(s3_cls)
#     s3_cls.periods, s3_cls.cos_vars_cumsum = get_cos_vars(s3_cls)
#
#     s4_cls = VarCls()
#     s4_cls.in_file = Path(
#         r'neckar_norm_cop_infill_discharge_1961_2015_20190118.csv')
#
#     s4_cls.var_type = 'cosines_mean'
#     s4_cls.norm_flag = norm_flag
#     s4_cls.mag_flag = mag_flag
#     s4_cls.lag = lag
#     s4_cls.sep = sep
#     s4_cls.time_fmt = time_fmt
#     s4_cls.beg_time = beg_time
#     s4_cls.end_time = end_time
#     s4_cls.stns = ['420', '427', '454']
#     s4_cls.label = '_'.join(s4_cls.stns)
#     s4_cls.df = get_df(s4_cls)
#     s4_cls.periods, s4_cls.cos_vars_cumsum = get_cos_vars(s4_cls)

    var_clss = [s1_cls, s2_cls]  # , s3_cls, s4_cls]

    plt.figure(figsize=fig_size)

    for var_cls in var_clss:
        plt.semilogx(
            var_cls.periods,
            var_cls.cos_vars_cumsum,
            alpha=0.5,
            label=f'{var_cls.label}__{var_cls.var_type}')

    plt.xlabel('Period')
    plt.ylabel(f'Variable')

    plt.xlim(plt.xlim()[::-1])

    plt.grid()

    plt.legend()

    plt.show()

#     plt.savefig(fig_name, bbox_inches='tight')

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
