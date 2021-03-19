'''
@author: Faizan-Uni-Stuttgart

Mar 16, 2021

6:24:48 PM

'''
import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path
from math import cos, acos, pi

import numpy as np
import pandas as pd
from sympy import Symbol
from sympy.solvers import solve
import matplotlib.pyplot as plt

from scipy.stats import rankdata, norm

from phsann.misc import roll_real_2arrs

asymms_exp = 3.0

plt.ioff()

DEBUG_FLAG = True


def get_cube_root(x):

    # From https://stackoverflow.com/questions/28014241/how-to-find-cube-root-using-python

    if 0 <= x:
        return x ** (1. / 3.)

    return -(-x) ** (1. / 3.)


def get_real_root_cubic(a3, a2, a1, a0):

    # From https://mathworld.wolfram.com/CubicFormula.html
    # a3 is always 1.

    assert a3 == 1

    Q = ((3 * a1) - (a2 ** 2)) / 9
    R = ((9 * a2 * a1) - (27 * a0) - (2 * (a2 ** 3))) / 54

    D = (Q ** 3) + (R ** 2)

    S = get_cube_root(R + (D ** 0.5))

    T = get_cube_root(R - (D ** 0.5))

    z1 = ((-1 / 3) * a2) + (S + T)

#     t1 = ((-1 / 3) * a2) - (0.5 * (S + T))
#
#     t2 = (0.5j * (3 ** 0.5)) * (S - T)
#
#     z2 = t1 + t2
#     z3 = t1 - t2

#     return (z1, z2, z3)

    return [z1]

# def get_real_root_cubic(a, b, c, d):

    # From https://math.vanderbilt.edu/schectex/courses/cubic/

#     p = -b / (3 * a)
#
#     q = p ** 3 + ((b * c - (3 * a * d)) / (6 * (a ** 2)))
#
#     r = c / (3 * a)
#
#     t1 = (r - (p ** 2)) ** 3
#
#     t2 = ((q ** 2) + t1) ** 0.5
#
#     t3 = (q + t2) ** (1 / 3)
#
#     t4 = (q - t2) ** (1 / 3)
#
#     x = t3 + t4 + p

    # From https://en.wikipedia.org/wiki/Cubic_equation

#     d0 = (b ** 2) - (3 * a * c)
#     d1 = (2 * (b ** 3)) - (9 * a * b * c) + (27 * (a ** 2) * d)
#
#     t1 = ((d1 ** 2) - (4 * (d0 ** 3))) ** 0.5
#
#     c1 = (0.5 * (d1 + t1)) ** (1 / 3)
#
#     t3 = 0.5 * (-1 + ((-3) ** 0.5))
#     t4 = 0.5 * (-1 - ((-3) ** 0.5))
#
# #     if c1 != 0:
#     x1 = -(1 / (3 * a)) * (b + c1 + (d0 / c1))
#
# #     else:
#     c2 = (0.5 * (d1 - t1)) ** (1 / 3)
#     x4 = -(1 / (3 * a)) * (b + c2 + (d0 / c2))
#
#     x2 = -(1 / (3 * a)) * (b + (c1 * t3) + (d0 / (c1 * t3)))
#     x3 = -(1 / (3 * a)) * (b + (c1 * t4) + (d0 / (c1 * t4)))

#     if np.any(~np.isfinite([x1, x2, x3])):
#         raise Exception

    # Also from https://en.wikipedia.org/wiki/Cubic_equation

#     p = ((3 * a * c) - (b ** 2)) / (3 * (a ** 2))
#
#     q = (
#         (2 * (b ** 3)) - (9 * a * b * c) + (27 * (a ** 2) * d) /
#         (27 * (a ** 3)))
#
#     t1 = 2 * ((-p / 3) ** 0.5) * cos(
#         (1 / 3) * acos(((3 * q) / (2 * p)) * ((-3 / p) ** 0.5)) - ((2 * pi * 1.0) / 3))
#
#     t2 = 2 * ((-p / 3) ** 0.5) * cos(
#         (1 / 3) * acos(((3 * q) / (2 * p)) * ((-3 / p) ** 0.5)) - ((2 * pi * 2.0) / 3))
#
#     t3 = 2 * ((-p / 3) ** 0.5) * cos(
#         (1 / 3) * acos(((3 * q) / (2 * p)) * ((-3 / p) ** 0.5)) - ((2 * pi * 3.0) / 3))
#
#     x1 = t1 - (b / (3 * a))
#     x2 = t2 - (b / (3 * a))
#     x3 = t3 - (b / (3 * a))

    # Also from https://en.wikipedia.org/wiki/Cubic_equation
    # Cardano.

#     p = ((3 * a * c) - (b ** 2)) / (3 * (a ** 2))
#
#     q = (
#         (2 * (b ** 3)) - (9 * a * b * c) + (27 * (a ** 2) * d) /
#         (27 * (a ** 3)))
#
#     t1 = (((q ** 2) * 0.25) + ((1 / 27) * (p ** 3))) ** 0.5
#
#     t2 = ((-q * 0.5) + t1) ** (1 / 3)
#
#     t3 = ((-q * 0.5) - t1) ** (1 / 3)
#
#     x = (t2 + t3) - (b / (3 * a))
#

#     return x
#     x1 = x2 = x3 = 0
#     return (x1, x2, x3, x4)
#     return


def get_mag_and_phs_spec(data):

    ft = np.fft.rfft(data)

    mag_spec = np.abs(ft)
    phs_spec = np.angle(ft)

    return mag_spec, phs_spec


def main():

    main_dir = Path(os.getcwd())
    os.chdir(main_dir)

    data_file = Path(
        r'P:\Synchronize\IWS\Testings\fourtrans_practice\phsann\neckar_norm_cop_infill_discharge_1961_2015_20190118.csv')

    beg_time = '2009-01-01'
    end_time = '2019-12-31'

    data_ser = pd.read_csv(
        data_file, sep=';', index_col=0).loc[beg_time:end_time, '420']

    data = data_ser.values

    if (data.size % 2):
        data = data[:-1]

    data_probs_orig = rankdata(data) / (data.size + 1.0)
#     data_norms = norm.ppf(data_probs_orig)

    data_probs, data_probs_rolled = roll_real_2arrs(
            data_probs_orig, data_probs_orig, 1)

    data_asymm = (data_probs - data_probs_rolled) ** asymms_exp

    asymm_mag_spec, asymm_phs_spec = get_mag_and_phs_spec(data_asymm)

    rand_phs_spec = np.empty_like(asymm_phs_spec)

    rand_phs_spec[1:-1] = -np.pi + (
        2 * np.pi * np.random.random(asymm_phs_spec.size - 2))

    rand_phs_spec[+0] = asymm_phs_spec[+0]
    rand_phs_spec[-1] = asymm_phs_spec[-1]

    rand_asymm_ft = np.empty_like(asymm_phs_spec, dtype=complex)
    rand_asymm_ft.real = np.cos(rand_phs_spec) * asymm_mag_spec
    rand_asymm_ft.imag = np.sin(rand_phs_spec) * asymm_mag_spec

    rand_asymm = np.fft.irfft(rand_asymm_ft)

    x_s = 0.002

    y = Symbol('y')

    rand_ps = [x_s]

    for a in rand_asymm:

        x = rand_ps[-1]

#         ys = [0]  # solve((x ** 3) - (y ** 3) - (3 * x * y * (x - y)) - a, y)

        ys = get_real_root_cubic(1, -3 * x, 3 * (x ** 2), a - (x ** 3))

#         print(ys)
#         print(zs)

#         rand_ps.append(norm.cdf(ys[0], loc=asymm_mean, scale=asymm_std))
        rand_ps.append(ys[0])

#         print('\n')

    rand_ps = np.array(rand_ps, dtype=float)
    rand_ps = rankdata(rand_ps) / (rand_ps.size + 1)

    plt.figure(figsize=(10, 10))

    plt.hist(rand_ps, bins=10)

    plt.savefig(r'P:\Downloads\rand_ps.png', bbox_inches='tight')

    plt.close()

    data_corrs = []
    rand_corrs = []

    lags = np.arange(1, 30, dtype=np.int64)
    for i in lags:
        data_probs, data_probs_rolled = roll_real_2arrs(
            data_probs_orig, data_probs_orig, i)

        data_corrs.append(
            np.corrcoef(data_probs, data_probs_rolled)[0, 1])

        rand_probs, rand_probs_rolled = roll_real_2arrs(
            rand_ps, rand_ps, i)

        rand_corrs.append(
            np.corrcoef(rand_probs, rand_probs_rolled)[0, 1])

    data_corrs = np.array(data_corrs, dtype=float)
    rand_corrs = np.array(rand_corrs, dtype=float)

    plt.figure(figsize=(10, 10))

    plt.plot(lags, data_corrs, c='r', alpha=0.75)
    plt.plot(lags, rand_corrs, c='b', alpha=0.75)

    plt.savefig(r'P:\Downloads\rand_ps_corrs.png', bbox_inches='tight')

    plt.close()

    return


if __name__ == '__main__':
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
