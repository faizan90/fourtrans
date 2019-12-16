'''
@author: Faizan-Uni-Stuttgart

'''
import os
import timeit
import time
from pathlib import Path
from timeit import default_timer

import numpy as np
import matplotlib.pyplot as plt

from fourtrans.cyth.chaar import haar_trans_1d

plt.ioff()


def read_tser(filn):
    ab = np.loadtxt(filn, skiprows=1, delimiter=",")
    return ab


def discreteHaarWaveletTransform(x):
    N = len(x)
    output = np.copy(x)
    div = np.sqrt(2.0)

    xx = x.copy()

    while N > 1:
        N = N // 2
        for i in range(0, N):
            summ = (xx[i * 2] + xx[i * 2 + 1]) / div
            difference = (xx[i * 2] - xx[i * 2 + 1]) / div
            output[i] = summ
            output[N + i] = difference
        xx[:N] = output[:N]
#        print(N,output[:],div)
    return output


def main():

    main_dir = Path(r'P:\Synchronize\IWS\fourtrans_practice')
    os.chdir(main_dir)

    filn = r"USUKv2.csv"
    usuk = read_tser(filn)
    filn = r"CHUSv2.csv"
    chus = read_tser(filn)
    for i in range(usuk.shape[0]):
        if usuk[i, 2] != chus[i, 2]:
            print(usuk[i, :])
            print(chus[i, :])
    uk1 = usuk[:, 3]
    ch1 = chus[:, 3]
    all = []

    uk = np.copy(uk1)
    ch = np.copy(ch1)
    na1 = uk.shape[0]
    uk = uk[na1 - 8192:]
    ch = ch[na1 - 8192:]
    rr = np.corrcoef(uk, ch)
    beg_time = default_timer()

    retuk = discreteHaarWaveletTransform(uk)
    retch = discreteHaarWaveletTransform(ch)

    pyth_time = default_timer() - beg_time

    print(f'python took: {pyth_time:0.8f}')

    for i1 in range(1, 13, 1):
        n1 = 2 ** i1
        n2 = 2 * n1
        print(n1, n2)

        hstda = np.sum(retuk[n1:n2] ** 2)
        hstdb = np.sum(retch[n1:n2] ** 2)
        #                print(astd,hstda,bstd,hstdb)
        hcov = np.sum(retuk[n1:n2] * retch[n1:n2])
        #                ch = hcov/ np.sqrt(hstda*hstdb)
        chh = np.corrcoef(retuk[n1:n2], retch[n1:n2])[0, 1]
        rch = hcov / np.sqrt(hstda * hstdb)

        print(n1, n2, rr[0, 1], chh, rch)
        all = np.append(all, chh)
    plt.plot(all, alpha=0.7, lw=2, label='python')

    uk = np.copy(uk1)
    ch = np.copy(ch1)
    na1 = uk.shape[0]
    uk = uk[na1 - 8192:]
    ch = ch[na1 - 8192:]
    rr = np.corrcoef(uk, ch)

    beg_time = default_timer()

    retuk = haar_trans_1d(uk)
    retch = haar_trans_1d(ch)

    cyth_time = default_timer() - beg_time

    print(f'cython took: {cyth_time:0.8f}')

    print(f'Improvement factor: {(pyth_time / cyth_time) - 1: 0.3f}')

    all = []
    for i1 in range(1, 13, 1):
        n1 = 2 ** i1
        n2 = 2 * n1
        print(n1, n2)

        hstda = np.sum(retuk[n1:n2] ** 2)
        hstdb = np.sum(retch[n1:n2] ** 2)
        #                print(astd,hstda,bstd,hstdb)
        hcov = np.sum(retuk[n1:n2] * retch[n1:n2])
        #                ch = hcov/ np.sqrt(hstda*hstdb)
        chh = np.corrcoef(retuk[n1:n2], retch[n1:n2])[0, 1]
        rch = hcov / np.sqrt(hstda * hstdb)

        print(n1, n2, rr[0, 1], chh, rch)
        all = np.append(all, chh)

    plt.plot(all, alpha=0.7, lw=1, label='cython')

    plt.legend()

    plt.show()

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
