'''
@author: Faizan-Uni-Stuttgart

Jan 8, 2020

2:21:48 PM

'''
import os
import time
import timeit
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from fourtrans import get_asymms_sample

plt.ioff()

DEBUG_FLAG = True


def get_alphaed_x(probs, alpha):
    probs = probs.copy()

    a_idxs = probs < alpha

    alphaed = probs[a_idxs]

    alphaed = alpha - alphaed

    probs[a_idxs] = alphaed

    return probs


def main():

    main_dir = Path(
        r'P:\Synchronize\IWS\Testings\fourtrans_practice\asymmetries_max')

    os.chdir(main_dir)

    # works well when n_vals is ca. more than a 1000
    n_vals = 10000
    n_slices = 100

    asymm_type = 2

    suff = '1001'

    alphas = np.linspace(0.0, 1.0, n_slices)

    scorrs = []
    asymms_1 = []
    asymms_2 = []

    for i in range(n_slices):
        x_probs = np.arange(1.0, n_vals) / (n_vals + 1.0)
        if asymm_type == 1:
            y_probs = x_probs  # asymm_1

        elif asymm_type == 2:
            y_probs = x_probs[::-1]  # asymm_2

        else:
            raise ValueError(f'Incorrect asymm_type: {asymm_type}!')

        x_probs = get_alphaed_x(x_probs, alphas[i])

        scorr = np.corrcoef(x_probs, y_probs)[0, 1]

        asymm_1, asymm_2 = get_asymms_sample(x_probs, y_probs)

#         plt.scatter(x_probs, y_probs, alpha=0.7)
#
#         plt.xlabel('x_probs')
#         plt.ylabel('y_probs')
#
#         plt.title(
#             f'scorr: {scorr:+0.3f}, '
#             f'asymm_1: {asymm_1:+0.6f}, '
#             f'asymm_2: {asymm_2:+0.6f}')
#
#         plt.grid()
#
#         plt.show()
#         plt.close()

        scorrs.append(scorr)
        asymms_1.append(asymm_1)
        asymms_2.append(asymm_2)

    scorrs = np.array(scorrs)
    asymms_1 = np.array(asymms_1)
    asymms_2 = np.array(asymms_2)

#     plt.figure(figsize=(6, 6))
#
#     x_probs = np.arange(1.0, n_vals) / (n_vals + 1.0)
#     y_probs = x_probs  # [::-1]
#
#     scorr = np.corrcoef(x_probs, y_probs)[0, 1]
#
#     asymm_1, asymm_2 = get_asymms_sample(x_probs, y_probs)
#
#     plt.scatter(x_probs, y_probs, alpha=0.7)
#
#     plt.xlabel('x_probs')
#     plt.ylabel('y_probs')
#
#     plt.title(
#         f'scorr: {scorr:+0.3f}, '
#         f'asymm_1: {asymm_1:+0.6f}, '
#         f'asymm_2: {asymm_2:+0.6f}')
#
#     plt.grid()
#
#     plt.show(block=False)

    if asymm_type == 1:
        alpha_scorr_ftn = (-2 * (alphas) ** 3) + 1
        scorr_asymm_ftn = (0.5 * (1 - scorrs)) * (1 - ((0.5 * (1 - scorrs)) ** (1.0 / 3.0)))

    elif asymm_type == 2:
        alpha_scorr_ftn = (+2 * (alphas) ** 3) - 1
        scorr_asymm_ftn = (0.5 * (scorrs + 1)) * (1 - ((0.5 * (scorrs + 1)) ** (1.0 / 3.0)))

    else:
        raise ValueError(f'Incorrect asymm_type: {asymm_type}!')

    axes = plt.subplots(2, 3, figsize=(15, 10))[1]

    # asymm_1
    axes[0, 0].scatter(alphas, scorrs, alpha=0.5)
    axes[0, 0].plot(alphas, alpha_scorr_ftn, alpha=0.9, color='r')
    axes[0, 0].set_xlabel('Alpha')
    axes[0, 0].set_ylabel('Spearman correlation')
    axes[0, 0].grid()

    axes[0, 1].scatter(alphas, asymms_1, alpha=0.5)
    axes[0, 1].set_xlabel('Alpha')
    axes[0, 1].set_ylabel('Asymm_1')
    axes[0, 1].grid()

    axes[0, 2].scatter(scorrs, asymms_1, alpha=0.5)
    if asymm_type == 1:
        axes[0, 2].plot(scorrs, scorr_asymm_ftn, alpha=0.9, color='r')
    axes[0, 2].set_xlabel('Spearman correlation')
    axes[0, 2].set_ylabel('Asymm_1')
    axes[0, 2].grid()

    # asymm_2
    axes[1, 0].scatter(alphas, scorrs, alpha=0.5)
    axes[1, 0].plot(alphas, alpha_scorr_ftn, alpha=0.9, color='r')
    axes[1, 0].set_xlabel('Alpha')
    axes[1, 0].set_ylabel('Spearman correlation')
    axes[1, 0].grid()

    axes[1, 1].scatter(alphas, asymms_2, alpha=0.5)
    axes[1, 1].set_xlabel('Alpha')
    axes[1, 1].set_ylabel('Asymm_2')
    axes[1, 1].grid()

    axes[1, 2].scatter(scorrs, asymms_2, alpha=0.5)
    if asymm_type == 2:
        axes[1, 2].plot(scorrs, scorr_asymm_ftn, alpha=0.9, color='r')
    axes[1, 2].set_xlabel('Spearman correlation')
    axes[1, 2].set_ylabel('Asymm_2')
    axes[1, 2].grid()

    plt.savefig(f'max_asymm_{suff}_type_{asymm_type}.png')

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
