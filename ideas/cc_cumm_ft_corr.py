'''
@author: Faizan-Uni-Stuttgart

Jun 3, 2022

10:28:38 AM

'''
import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt; plt.ioff()

DEBUG_FLAG = False


def main():

    main_dir = Path(os.getcwd())
    os.chdir(main_dir)

    #==========================================================================

    # ts = np.random.random(100000)
    ts = np.arange(100)

    ts = np.concatenate((ts, ts[::-1]))

    if (ts.size % 2):
        ts = ts[:-1]

    cumm_ft_corr, periods = get_cumm_ft_corr_auto(ts)

    ft_ts = np.fft.rfft(ts)
    ft_ts[5:] = 0

    mags = np.abs(ft_ts)

    ift_ts = np.fft.irfft(ft_ts)

    if False:
        plt.semilogx(
            periods,
            cumm_ft_corr,
            alpha=0.75,
            color='r',
            label='REF',
            lw=4,
            zorder=1)

        plt.scatter(
            periods,
            cumm_ft_corr,
            alpha=0.75,
            color='r',
            zorder=2)

        plt.xlim(plt.xlim()[::-1])

    else:
        plt.plot(
            ts,
            alpha=0.75,
            color='r',
            label='REF',
            lw=4,
            zorder=2)

        plt.plot(
            ift_ts,
            alpha=0.75,
            color='b',
            label='TNC',
            lw=2,
            zorder=3)

        for i in range(100):

            rand_ft_ts = np.empty_like(ft_ts)

            rand_phss = -np.pi + (
                2 * np.pi * np.random.random(size=ft_ts.shape[0]))

            rand_phss[0] = 0

            rand_ft_ts.real = mags * np.cos(rand_phss)
            rand_ft_ts.imag = mags * np.sin(rand_phss)

            rand_ft_ts[:2] = ft_ts[:2]

            rand_ift_ts = np.fft.irfft(rand_ft_ts)

            if i == 0:
                label = 'SIM'

            else:
                label = None

            plt.plot(
                rand_ift_ts,
                alpha=0.2,
                color='k',
                label=label,
                lw=1,
                zorder=1)

    plt.legend()

    plt.grid()
    plt.gca().set_axisbelow(True)

    plt.xlabel('Period')
    plt.ylabel('Cummulative power')

    plt.show()

    # plt.savefig(f'{in_file.stem}_{out_fig_name_pwr}', bbox_inches='tight')
    plt.close()
    return


def get_cumm_ft_corr_auto(data):

    '''
    Get the cummulative correlation contribution by each coefficient of the
    Foruier transform of the input data.

    Parameters
    ----------
    data : np.ndarray
        The input data as a 1D numpy array. Should have an even number of values.
        All values should be finite real values.

    Returns
    -------
    Cummulative correlation contribution specturm and the corresponding period
    of each value in this spectrum as a tuple.
    '''

    assert isinstance(data, np.ndarray), type(data)

    assert data.ndim == 1, data.ndim

    assert data.size > 1, 'Too few values in the input!'

    assert (data.size % 2) == 0, 'Not even number of time steps!'

    assert np.all(np.isfinite(data)), 'Invalid values in the input!'

    # The first value is the sum of the values. Drop that.
    ft = np.fft.rfft(data)[1:]

    pwrs = np.abs(ft) ** 2

    assert np.all(np.isfinite(pwrs)), 'Invalid values in the power spectrum!'

    numr = pwrs.cumsum()

    cumm_ft_corr = numr / numr[-1]

    assert np.all(np.isfinite(cumm_ft_corr)), (
        'Invalid values in the cummulative correlation contribution!')

    periods = (pwrs.size * 2) / (np.arange(1, pwrs.size + 1))

    assert periods.size == pwrs.size, (periods.size, pwrs.size)

    return cumm_ft_corr, periods


if __name__ == '__main__':
    print('#### Started on %s ####\n' % time.asctime())
    START = timeit.default_timer()

    #==========================================================================
    # When in post_mortem:
    # 1. "where" to show the stack,
    # 2. "up" move the stack up to an older frame,
    # 3. "down" move the stack down to a newer frame, and
    # 4. "interact" start an interactive interpreter.
    #==========================================================================

    if DEBUG_FLAG:
        try:
            main()

        except:
            pre_stack = tb.format_stack()[:-1]

            err_tb = list(tb.TracebackException(*sys.exc_info()).format())

            lines = [err_tb[0]] + pre_stack + err_tb[2:]

            for line in lines:
                print(line, file=sys.stderr, end='')

            import pdb
            pdb.post_mortem()
    else:
        main()

    STOP = timeit.default_timer()
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))
