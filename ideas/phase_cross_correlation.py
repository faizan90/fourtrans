'''
@author: Faizan-Uni-Stuttgart

'''
import os
import time
import timeit
from pathlib import Path

import matplotlib as mpl
mpl.rc('font', size=14)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid

from spinterps import Variogram as VG

plt.ioff()


def get_corr_mat(phas):

    n_phas, n_stns = phas.shape

    cov_mat = np.empty((n_stns, n_phas, n_phas), dtype=float)

    for i in range(n_stns):
        for j in range(n_phas):
            for k in range(n_phas):
                cov_mat[i, j, k] = np.cos(phas[j, i] - phas[k, i])

    return cov_mat


def plot_corr_mat(cov_mat, titles, out_fig_name):

    n_stns = cov_mat.shape[0]

    fig = plt.figure(figsize=(15, 7))

    axs = AxesGrid(
        fig,
        111,
        nrows_ncols=(1, 2),
        axes_pad=0.5,
        cbar_mode='single',
        cbar_location='right',
        cbar_pad=0.1)

    for i in range(n_stns):
        cb_input = axs[i].imshow(cov_mat[i, :, :], vmin=-1, vmax=+1)

        axs[i].set_title(titles[i])

        axs[i].set_xlabel('Phase index')

        if not i:
            axs[i].set_ylabel('Phase index')

    # when cbar_mode is 'single', for ax in grid, ax.cax = grid.cbar_axes[0]

    cbar = axs[-1].cax.colorbar(cb_input)  # use this or the next one
#     cbar = axs.cbar_axes[0].colorbar(cb_input)  # use this or the one before
    cbar.set_label_text('correlation')

#     cbar = fig.colorbar(cb_input, ax=axs.ravel().tolist(), shrink=0.6)
#     cbar.set_label('correlation')

#     cbar = fig.colorbar(cb_input, ax=axs.ravel().tolist(), shrink=0.6)
#     cbar.set_ticks(np.arange(0, 1.0, 0.5))
#     cbar.set_ticklabels(['low', 'medium', 'high'])

    plt.suptitle(
        'Phase spectrum cross correlation comparision\n'
        '(cosine of phase difference)')

    plt.savefig(str(out_fig_name), bbox_inches='tight')
    plt.close()
    return


def get_evg(vals):

    n_vals = vals.shape[0]

#     crds_lin = np.arange(n_vals)
#     x_crds, y_crds = np.meshgrid(crds_lin, crds_lin)

    x_crds = np.arange(n_vals)
    y_crds = np.zeros(n_vals)

    vg = VG(
        x=x_crds.ravel(),
        y=y_crds.ravel(),
        z=vals.ravel(),
        mdr=1.0,
        nk=10,
        typ='var',
        perm_r_list=[1, 2],
        fil_nug_vg='Nug',
        ld=None,
        uh=None,
        h_itrs=100,
        opt_meth='L-BFGS-B',
        opt_iters=1000,
        fit_vgs=['Sph', 'Exp', 'Gau'],
        n_best=3,
        evg_name='classic',
        use_wts=False,
        ngp=5,
        fit_thresh=0.01)

    vg.fit()

    return vg


def plot_vg(vg, out_fig_name):

    evg = vg.vg_vg_arr
    h_arr = vg.vg_h_arr
    vg_fit = vg.vg_fit
    vg_names = vg.best_vg_names

    fit_vg_list = vg.vg_str_list
#     fit_vgs_no = len(fit_vg_list) - 1

    plt.figure(figsize=(15, 7))

    plt.plot(h_arr, evg, 'bo', alpha=0.3)

    for m in range(len(vg_names)):
        plt.plot(
            vg_fit[m][:, 0],
            vg_fit[m][:, 1],
            c=pd.np.random.rand(3,),
            linewidth=4,
            zorder=m,
            label=fit_vg_list[m],
            alpha=0.6)

    plt.grid()

    plt.xlabel('Distance')
    plt.ylabel('Variogram')

    plt.legend(loc=4, framealpha=0.7)

    plt.savefig(str(out_fig_name), bbox_inches='tight')

#     plt.show()
    plt.close()
    return


def main():

    main_dir = Path(
        r'P:\Synchronize\IWS\Testings\fourtrans_practice\cross_corrs')

    os.chdir(main_dir)

    in_file = (
        '../similar_dissimilar_series/neckar_daily_discharge_1961_2015.csv')

    beg_time = '1961-01-01'
    end_time = '1961-12-30'
    stns = ['454', '3470']

    out_corr_fig_name = 'randomized_phase_cross_corrs.png'
    corr_fig_titles = ['Reference', 'Randomized']

    df = pd.read_csv(in_file, index_col=0, sep=';')
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d')

    df = df.loc[beg_time:end_time, stns]

    if df.shape[0] % 2:
        df = df.iloc[:-1, :]

    assert df.shape[0] and df.shape[1] > 1

    assert np.all(np.isfinite(df.values))

    ft = np.fft.rfft(df.values, axis=0)

    phas = np.angle(ft)[1:-1]
#     mags = np.abs(ft)[1:-1]

#     phas[:, 1] = phas[:, 0] + (
#         -np.pi + ((2 * np.pi) * np.random.random(phas.shape[0])))
#
#     phas[:, 1] = phas[:, 0]
#
    phas[:, 1] = (-np.pi + ((2 * np.pi) * np.random.random(phas.shape[0])))
#
#     phas[:, 1] = phas[:, 0] + 20  # (-np.pi + ((2 * np.pi) * np.random.random(phas.shape[0])))
#
#     phas[:, 1] = np.roll(phas[:, 0], 20)

    corr_mat = get_corr_mat(phas)

#     sel_corr = corr_mat[0, 1, :]
#
#     trgt_sum = (sel_corr).sum()
#
#     for i, pha in enumerate(np.linspace(-np.pi, np.pi, 100)):
#         print(i, pha, (np.cos(pha - phas[:, 0])).sum() - trgt_sum)

    plot_corr_mat(corr_mat, corr_fig_titles, out_corr_fig_name)

#     vg = get_evg(phas[:, 0])
#     plot_vg(vg)

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

    main()

    STOP = timeit.default_timer()
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))

    if _save_log_:
        log_link.stop()
