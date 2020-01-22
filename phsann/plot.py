'''
@author: Faizan-Uni

Jan 16, 2020

1:32:31 PM
'''
from math import ceil
from pathlib import Path

import h5py
import numpy as np
import matplotlib.cm as mpl_cm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from ..simultexts.misc import print_sl, print_el

plt.ioff()


class PlotSettings:

    '''For internal use

    Description of settings at:
    https://matplotlib.org/tutorials/introductory/customizing.html
    '''

    def __init__(self, figsize, dpi, fontsize):

        assert isinstance(figsize, tuple), 'figsize not a tuple!'

        assert len(figsize) == 2, 'Only two items allowed in figsize!'

        assert all([isinstance(size, (int, float)) for size in figsize]), (
            'Items in figsize can either be integers or floats!')

        assert all([size > 0 for size in figsize]), (
            'items in figsize cannot be <= 0!')

        assert isinstance(dpi, int), 'dpi not an integer!'

        assert dpi > 0, 'dpi should be > 0!'

        assert isinstance(fontsize, int), 'fontsize not an integer!'

        assert fontsize > 0

        self.prms_dict = {
            'figure.figsize': figsize,
            'figure.dpi': dpi,
            'font.size': fontsize
            }

        return


class PlotLineSettings(PlotSettings):

    '''For internal use'''

    def __init__(
            self, figsize, dpi, fontsize, alpha_1, alpha_2, lw, lc_1, lc_2):

        PlotSettings.__init__(self, figsize, dpi, fontsize)

        assert isinstance(alpha_1, (int, float)), (
            'alpha_1 can only be an integer or a float!')

        assert 0 <= alpha_1 <= 1, 'alpha can only be 0 <= alpha_1 <= 1!'

        assert isinstance(alpha_2, (int, float)), (
            'alpha_2 can only be an integer or a float!')

        assert 0 <= alpha_2 <= 1, 'alpha can only be 0 <= alpha_2 <= 1!'

        assert isinstance(lw, (int, float)), 'lw not an integer or a float!'

        assert lw > 0, 'lw must be > 0!'

        assert isinstance(lc_1, (str, tuple, hex)), 'Invalid lc_1!'
        assert isinstance(lc_2, (str, tuple, hex)), 'Invalid lc_2!'

        self.prms_dict.update({
            'lines.linewidth': lw,
            })

        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2

        self.lc_1 = lc_1
        self.lc_2 = lc_2
        return


class PlotImageSettings(PlotSettings):

    '''For internal use'''

    def __init__(self, figsize, dpi, fontsize, alpha_1, alpha_2, cmap):

        PlotSettings.__init__(self, figsize, dpi, fontsize)

        assert isinstance(cmap, str), 'cmap not a string!'

        assert cmap in mpl_cm.cmap_d, 'Unknown cmap!'

        assert isinstance(alpha_1, (int, float)), (
            'alpha_1 can only be an integer or a float!')

        assert 0 <= alpha_1 <= 1, 'alpha_text can only be 0 <= alpha_1 <= 1!'

        assert isinstance(alpha_2, (int, float)), (
            'alpha_2 can only be an integer or a float!')

        assert 0 <= alpha_2 <= 1, 'alpha_text can only be 0 <= alpha_2 <= 1!'

        self.prms_dict.update({
            'image.cmap': cmap,
            })

        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        return


class PlotScatterSettings(PlotSettings):

    '''For internal use'''

    def __init__(self, figsize, dpi, fontsize, alpha_1, alpha_2, color):

        PlotSettings.__init__(self, figsize, dpi, fontsize)

        assert isinstance(color, (str, tuple, hex)), 'Invalid color!'

        assert isinstance(alpha_1, (int, float)), (
            'alpha_1 can only be an integer or a float!')

        assert 0 <= alpha_1 <= 1, 'alpha_text can only be 0 <= alpha_1 <= 1!'

        assert isinstance(alpha_2, (int, float)), (
            'alpha_2 can only be an integer or a float!')

        assert 0 <= alpha_2 <= 1, 'alpha_text can only be 0 <= alpha_2 <= 1!'

        self.prms_dict.update({
            })

        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.color = color
        return


def get_mpl_prms(keys):

    prms_dict = {key: plt.rcParams[key] for key in keys}

    return prms_dict


def set_mpl_prms(prms_dict):

    plt.rcParams.update(prms_dict)

    return


class PhaseAnnealingPlot:

    def __init__(self, verbose):

        assert isinstance(verbose, bool), 'verbose not a Boolean!'

        self._vb = verbose

        self._plt_in_h5_file = None
        self._plt_outputs_dir = None

        self._plt_input_set_flag = False
        self._plt_output_set_flag = False

        self._init_plt_settings()

        self._plt_verify_flag = False
        return

    def _init_plt_settings(self):

        '''One place to change plotting parameters for all plots'''

        fontsize = 20
        dpi = 300

        default_line_sett = PlotLineSettings(
            (20, 7), dpi, fontsize, 0.2, 0.5, 2.0, 'k', 'r')

        self._plt_sett_tols = default_line_sett
        self._plt_sett_objs = default_line_sett
        self._plt_sett_acpt_rates = default_line_sett
        self._plt_sett_phss = default_line_sett
        self._plt_sett_temps = default_line_sett
        self._plt_sett_phs_red_rates = default_line_sett
        self._plt_sett_idxs = default_line_sett

        self._plt_sett_1D_vars = PlotLineSettings(
            (18, 6), dpi, fontsize, 0.2, 0.7, 2.0, 'k', 'r')

        self._plt_sett_ecops_denss = PlotImageSettings(
            (15, 15), dpi, fontsize, 0.9, 0.7, 'Blues')

        self._plt_sett_ecops_sctr = PlotScatterSettings(
            (15, 15), dpi, fontsize, 0.4, 0.7, 'C0')

        return

    def set_input(self, in_h5_file):

        if self._vb:
            print_sl()

            print(
                'Setting input HDF5 file for plotting phase annealing '
                'results...\n')

        assert isinstance(in_h5_file, (str, Path))

        in_h5_file = Path(in_h5_file)

        assert in_h5_file.exists(), 'in_h5_file does not exist!'

        self._plt_in_h5_file = in_h5_file

        if self._vb:
            print('Set the following input HDF5 file:', self._plt_in_h5_file)

            print_el()

        self._plt_input_set_flag = True
        return

    def set_output(self, outputs_dir):

        if self._vb:
            print_sl()

            print(
                'Setting outputs directory for plotting phase annealing '
                'results...\n')

        assert isinstance(outputs_dir, (str, Path))

        outputs_dir = Path(outputs_dir)

        if not outputs_dir.exists():
            outputs_dir.mkdir(exist_ok=True)

        assert outputs_dir.exists(), 'Could not create outputs_dir!'

        self._plt_outputs_dir = outputs_dir

        if self._vb:
            print(
                'Set the following outputs directory:', self._plt_outputs_dir)

            print_el()

        self._plt_output_set_flag = True
        return

    def plot_opt_state_vars(self):

        if self._vb:
            print_sl()

            print('Plotting optimization state variables...')

        assert self._plt_verify_flag, 'Plot in an unverified state!'

        opt_state_dir = self._plt_outputs_dir / 'optimization_state_variables'

        opt_state_dir.mkdir(exist_ok=True)

        h5_hdl = h5py.File(self._plt_in_h5_file, mode='r', driver=None)

        self._plot_tols(h5_hdl, opt_state_dir)

        self._plot_obj_vals(h5_hdl, opt_state_dir)

        self._plot_acpt_rates(h5_hdl, opt_state_dir)

        self._plot_phss(h5_hdl, opt_state_dir)

        self._plot_temps(h5_hdl, opt_state_dir)

        self._plot_phs_red_rates(h5_hdl, opt_state_dir)

        self._plot_idxs(h5_hdl, opt_state_dir)

        h5_hdl.close()

        if self._vb:
            print('Done plotting optimization state variables.')

            print_el()
        return

    def plot_comparison(self):

        if self._vb:
            print_sl()

            print('Plotting comparision...')

        assert self._plt_verify_flag, 'Plot in an unverified state!'

        cmpr_dir = self._plt_outputs_dir / 'comparison'

        cmpr_dir.mkdir(exist_ok=True)

        h5_hdl = h5py.File(self._plt_in_h5_file, mode='r', driver=None)

        self._plot_cmpr_1D_vars(h5_hdl, cmpr_dir)

        self._plot_cmpr_ecop_denss(h5_hdl, cmpr_dir)

        self._plot_cmpr_ecop_scatter(h5_hdl, cmpr_dir)

        h5_hdl.close()

        if self._vb:
            print('Done plotting comparision.')

            print_el()
        return

    def verify(self):

        assert self._plt_input_set_flag, 'Call set_input first!'
        assert self._plt_output_set_flag, 'Call set_output first!'

        self._plt_verify_flag = True
        return

    def _plot_tols(self, h5_hdl, out_dir):

        plt_sett = self._plt_sett_tols

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        sim_grp_main = h5_hdl['data_sim_rltzns']

        beg_iters = h5_hdl['settings'].attrs['_sett_ann_obj_tol_iters']

        plt.figure()

        for rltzn_lab in sim_grp_main:
            tol_iters = np.arange(
                sim_grp_main[f'{rltzn_lab}/tols'].shape[0]) + beg_iters

            plt.plot(
                tol_iters,
                sim_grp_main[f'{rltzn_lab}/tols'],
                alpha=plt_sett.alpha_1,
                color=plt_sett.lc_1)

        plt.ylim(0, plt.ylim()[1])

        plt.xlabel('Iteration')

        plt.ylabel(
            f'Mean absolute objective function\ndifference of previous '
            f'{beg_iters} iterations')

        plt.grid()

        plt.savefig(
            str(out_dir / f'opt_state__tols.png'), bbox_inches='tight')

        plt.close()

        set_mpl_prms(old_mpl_prms)
        return

    def _plot_obj_vals(self, h5_hdl, out_dir):

        plt_sett = self._plt_sett_objs

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        sim_grp_main = h5_hdl['data_sim_rltzns']

        # obj_vals_all
        plt.figure()

        for rltzn_lab in sim_grp_main:
            plt.plot(
                sim_grp_main[f'{rltzn_lab}/obj_vals_all'],
                alpha=plt_sett.alpha_1,
                color=plt_sett.lc_1)

        plt.ylim(0, plt.ylim()[1])

        plt.xlabel('Iteration')

        plt.ylabel(f'Raw objective function value')

        plt.grid()

        plt.savefig(
            str(out_dir / f'opt_state__obj_vals_all.png'), bbox_inches='tight')

        plt.close()

        # obj_vals_min
        plt.figure()

        for rltzn_lab in sim_grp_main:
            plt.plot(
                sim_grp_main[f'{rltzn_lab}/obj_vals_min'],
                alpha=plt_sett.alpha_1,
                color=plt_sett.lc_1)

        plt.ylim(0, plt.ylim()[1])

        plt.xlabel('Iteration')

        plt.ylabel(f'Running minimum objective function value')

        plt.grid()

        plt.savefig(
            str(out_dir / f'opt_state__obj_vals_min.png'), bbox_inches='tight')

        plt.close()

        set_mpl_prms(old_mpl_prms)
        return

    def _plot_acpt_rates(self, h5_hdl, out_dir):

        plt_sett = self._plt_sett_acpt_rates

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        sim_grp_main = h5_hdl['data_sim_rltzns']

        # acpt_rates_all
        plt.figure()

        for rltzn_lab in sim_grp_main:
            plt.plot(
                sim_grp_main[f'{rltzn_lab}/acpt_rates_all'],
                alpha=plt_sett.alpha_1,
                color=plt_sett.lc_1)

        plt.ylim(0, 1)

        plt.xlabel('Iteration')

        plt.ylabel(f'Running mean acceptance rate')

        plt.grid()

        plt.savefig(
            str(out_dir / f'opt_state__acpt_rates.png'), bbox_inches='tight')

        plt.close()

        # acpt_rates_dfrntl
        acpt_rate_iters = (
            h5_hdl['settings'].attrs['_sett_ann_acpt_rate_iters'])

        plt.figure()

        for rltzn_lab in sim_grp_main:
            acpt_rate_dfrntl = sim_grp_main[f'{rltzn_lab}/acpt_rates_dfrntl']

            plt.plot(
                acpt_rate_dfrntl[:, 0],
                acpt_rate_dfrntl[:, 1],
                alpha=plt_sett.alpha_1,
                color=plt_sett.lc_1)

        plt.ylim(0, 1)

        plt.xlabel('Iteration')

        plt.ylabel(
            f'Mean acceptance rate for the\npast {acpt_rate_iters} '
            f'iterations')

        plt.grid()

        plt.savefig(
            str(out_dir / f'opt_state__acpt_rates_dfrntl.png'),
            bbox_inches='tight')

        plt.close()

        set_mpl_prms(old_mpl_prms)
        return

    def _plot_phss(self, h5_hdl, out_dir):

        plt_sett = self._plt_sett_phss

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        sim_grp_main = h5_hdl['data_sim_rltzns']

        plt.figure()

        for rltzn_lab in sim_grp_main:
            plt.plot(
                sim_grp_main[f'{rltzn_lab}/phss_all'],
                alpha=plt_sett.alpha_1,
                color=plt_sett.lc_1)

        plt.xlabel('Iteration')

        plt.ylabel(f'Phase')

        plt.grid()

        plt.savefig(
            str(out_dir / f'opt_state__phss_all.png'), bbox_inches='tight')

        plt.close()

        set_mpl_prms(old_mpl_prms)
        return

    def _plot_idxs(self, h5_hdl, out_dir):

        plt_sett = self._plt_sett_idxs

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        sim_grp_main = h5_hdl['data_sim_rltzns']

        # idxs_all
        plt.figure()

        for rltzn_lab in sim_grp_main:
            plt.plot(
                sim_grp_main[f'{rltzn_lab}/idxs_all'],
                alpha=plt_sett.alpha_1,
                color=plt_sett.lc_1)

        plt.xlabel('Iteration')

        plt.ylabel(f'Index')

        plt.grid()

        plt.savefig(
            str(out_dir / f'opt_state__idxs_all.png'), bbox_inches='tight')

        plt.close()

        # idxs_acpt
        plt.figure()

        for rltzn_lab in sim_grp_main:

            idxs_acpt = sim_grp_main[f'{rltzn_lab}/idxs_acpt']

            plt.plot(
                idxs_acpt[:, 0],
                idxs_acpt[:, 1],
                alpha=plt_sett.alpha_1,
                color=plt_sett.lc_1)

        plt.xlabel('Iteration')

        plt.ylabel(f'Index')

        plt.grid()

        plt.savefig(
            str(out_dir / f'opt_state__idxs_acpt.png'), bbox_inches='tight')

        plt.close()

        set_mpl_prms(old_mpl_prms)
        return

    def _plot_temps(self, h5_hdl, out_dir):

        plt_sett = self._plt_sett_temps

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        sim_grp_main = h5_hdl['data_sim_rltzns']

        plt.figure()

        for rltzn_lab in sim_grp_main:
            temps_all = sim_grp_main[f'{rltzn_lab}/temps']

            plt.plot(
                temps_all[:, 0],
                temps_all[:, 1],
                alpha=plt_sett.alpha_1,
                color=plt_sett.lc_1)

        plt.xlabel('Iteration')

        plt.ylabel(f'Annealing temperature')

        plt.grid()

        plt.savefig(
            str(out_dir / f'opt_state__temps.png'), bbox_inches='tight')

        plt.close()

        set_mpl_prms(old_mpl_prms)
        return

    def _plot_phs_red_rates(self, h5_hdl, out_dir):

        plt_sett = self._plt_sett_phs_red_rates

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        sim_grp_main = h5_hdl['data_sim_rltzns']

        plt.figure()

        for rltzn_lab in sim_grp_main:
            temps_all = sim_grp_main[f'{rltzn_lab}/phs_red_rates']

            plt.plot(
                temps_all[:, 0],
                temps_all[:, 1],
                alpha=plt_sett.alpha_1)

        plt.ylim(0, 1)

        plt.xlabel('Iteration')

        plt.ylabel(f'Phase increment reduction rate')

        plt.grid()

        plt.savefig(
            str(out_dir / f'opt_state__phs_red_rates.png'),
            bbox_inches='tight')

        plt.close()

        set_mpl_prms(old_mpl_prms)
        return

    def _plot_cmpr_1D_vars(self, h5_hdl, out_dir):

        plt_sett = self._plt_sett_1D_vars

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        axes = plt.subplots(1, 3)[1]

        lag_steps = h5_hdl['settings/_sett_obj_lag_steps']

        sim_grp_main = h5_hdl['data_sim_rltzns']

        leg_flag = True
        for rltzn_lab in sim_grp_main:
            if leg_flag:
                label = 'sim'

            else:
                label = None

            axes[0].plot(
                lag_steps,
                sim_grp_main[f'{rltzn_lab}/scorrs'],
                alpha=plt_sett.alpha_1,
                color=plt_sett.lc_1,
                label=label)

            axes[1].plot(
                lag_steps,
                sim_grp_main[f'{rltzn_lab}/asymms_1'],
                alpha=plt_sett.alpha_1,
                color=plt_sett.lc_1,
                label=label)

            axes[2].plot(
                lag_steps,
                sim_grp_main[f'{rltzn_lab}/asymms_2'],
                alpha=plt_sett.alpha_1,
                color=plt_sett.lc_1,
                label=label)

            leg_flag = False

        axes[0].plot(
            lag_steps,
            h5_hdl['data_ref_rltzn/_ref_scorrs'],
            alpha=plt_sett.alpha_2,
            color=plt_sett.lc_2,
            label='ref')

        axes[1].plot(
            lag_steps,
            h5_hdl['data_ref_rltzn/_ref_asymms_1'],
            alpha=plt_sett.alpha_2,
            color=plt_sett.lc_2,
            label='ref')

        axes[2].plot(
            lag_steps,
            h5_hdl['data_ref_rltzn/_ref_asymms_2'],
            alpha=plt_sett.alpha_2,
            color=plt_sett.lc_2,
            label='ref')

        axes[0].grid()
        axes[1].grid()
        axes[2].grid()

        axes[0].legend(framealpha=0.7)
        axes[1].legend(framealpha=0.7)
        axes[2].legend(framealpha=0.7)

        axes[0].set_xlabel('Lag steps')
        axes[0].set_ylabel('Spearman correlation')

        axes[1].set_xlabel('Lag steps')
        axes[1].set_ylabel('Asymmetry (Type - 1)')

        axes[2].set_xlabel('Lag steps')
        axes[2].set_ylabel('Asymmetry (Type - 2)')

        plt.tight_layout()

        plt.savefig(
            str(out_dir / f'cmpr__scorrs_asymms.png'), bbox_inches='tight')

        plt.close()

        set_mpl_prms(old_mpl_prms)
        return

    def _plot_cmpr_ecop_denss_base(
            self,
            lag_steps,
            fig_suff,
            vmin,
            vmax,
            ecop_denss,
            cmap_mappable_beta,
            out_dir,
            plt_sett):

        rows = int(ceil(lag_steps.size ** 0.5))
        cols = ceil(lag_steps.size / rows)

        fig, axes = plt.subplots(rows, cols)

        dx = 1.0 / (ecop_denss.shape[2] + 1.0)
        dy = 1.0 / (ecop_denss.shape[1] + 1.0)

        y, x = np.mgrid[slice(dy, 1.0, dy), slice(dx, 1.0, dx)]

        ax_ctr = 0
        row = 0
        col = 0
        for i in range(rows * cols):

            if i >= (lag_steps.size):
                axes[row, col].set_axis_off()

            else:
                axes[row, col].pcolormesh(
                    x,
                    y,
                    ecop_denss[i],
                    vmin=vmin,
                    vmax=vmax,
                    alpha=plt_sett.alpha_1)

                axes[row, col].set_aspect('equal')

                axes[row, col].text(
                    0.1,
                    0.85,
                    f'{lag_steps[i]} step(s) lag',
                    alpha=plt_sett.alpha_2)

                if col:
                    axes[row, col].set_yticklabels([])

                else:
                    axes[row, col].set_ylabel('Probability')

                if row < (rows - 1):
                    axes[row, col].set_xticklabels([])

                else:
                    axes[row, col].set_xlabel('Probability')

            col += 1
            if not (col % cols):
                row += 1
                col = 0

            ax_ctr += 1

        cbaxes = fig.add_axes([0.2, 0.0, 0.65, 0.05])

        plt.colorbar(
            mappable=cmap_mappable_beta,
            cax=cbaxes,
            orientation='horizontal',
            label='Empirical copula density',
            extend='max',
            alpha=plt_sett.alpha_1)

        plt.savefig(
            str(out_dir / f'cmpr__ecop_denss_{fig_suff}.png'),
            bbox_inches='tight')

        plt.close()
        return

    def _plot_cmpr_ecop_denss(self, h5_hdl, out_dir):

        plt_sett = self._plt_sett_ecops_denss

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        lag_steps = h5_hdl['settings/_sett_obj_lag_steps']

        ecop_denss = h5_hdl['data_ref_rltzn/_ref_ecop_dens_arrs']

        vmin = 0.0
        vmax = np.max(ecop_denss) * 0.85

        fig_suff = 'ref'

        cmap_beta = plt.get_cmap(plt.rcParams['image.cmap'])

        cmap_mappable_beta = plt.cm.ScalarMappable(
            norm=Normalize(vmin / 100, vmax / 100, clip=True),
            cmap=cmap_beta)

        cmap_mappable_beta.set_array([])

        self._plot_cmpr_ecop_denss_base(
            lag_steps,
            fig_suff,
            vmin,
            vmax,
            ecop_denss,
            cmap_mappable_beta,
            out_dir,
            plt_sett)

        sim_grp_main = h5_hdl['data_sim_rltzns']

        for rltzn_lab in sim_grp_main:
            fig_suff = f'sim_{rltzn_lab}'
            ecop_denss = sim_grp_main[f'{rltzn_lab}/ecop_dens']

            self._plot_cmpr_ecop_denss_base(
                lag_steps,
                fig_suff,
                vmin,
                vmax,
                ecop_denss,
                cmap_mappable_beta,
                out_dir,
                plt_sett)

        set_mpl_prms(old_mpl_prms)
        return

    def _plot_cmpr_ecop_scatter_base(
            self,
            lag_steps,
            fig_suff,
            probs,
            out_dir,
            plt_sett):

        rows = int(ceil(lag_steps.size ** 0.5))
        cols = ceil(lag_steps.size / rows)

        axes = plt.subplots(rows, cols)[1]

        row = 0
        col = 0
        for i in range(rows * cols):

            if i >= (lag_steps.size):
                axes[row, col].set_axis_off()

            else:
                rolled_probs = np.roll(probs, lag_steps[i])

                axes[row, col].scatter(
                    probs,
                    rolled_probs,
                    color=plt_sett.color,
                    alpha=plt_sett.alpha_1)

                axes[row, col].grid()

                axes[row, col].set_aspect('equal')

                axes[row, col].text(
                    0.05,
                    0.9,
                    f'{lag_steps[i]} step(s) lag',
                    alpha=plt_sett.alpha_2)

                if col:
                    axes[row, col].set_yticklabels([])

                else:
                    axes[row, col].set_ylabel('Probability')

                if row < (rows - 1):
                    axes[row, col].set_xticklabels([])

                else:
                    axes[row, col].set_xlabel('Probability')

            col += 1
            if not (col % cols):
                row += 1
                col = 0

        plt.savefig(
            str(out_dir / f'cmpr__ecops_scatter_{fig_suff}.png'),
            bbox_inches='tight')

        plt.close()
        return

    def _plot_cmpr_ecop_scatter(self, h5_hdl, out_dir):

        plt_sett = self._plt_sett_ecops_sctr

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        lag_steps = h5_hdl['settings/_sett_obj_lag_steps']

        rnks = h5_hdl['data_ref_rltzn/_ref_rnk']

        probs = rnks / (rnks.size + 1)

        fig_suff = 'ref'

        self._plot_cmpr_ecop_scatter_base(
            lag_steps, fig_suff, probs, out_dir, plt_sett)

        sim_grp_main = h5_hdl['data_sim_rltzns']

        for rltzn_lab in sim_grp_main:
            rnks = sim_grp_main[f'{rltzn_lab}/rnk']

            probs = rnks / (rnks.size + 1)

            fig_suff = f'sim_{rltzn_lab}'

            self._plot_cmpr_ecop_scatter_base(
                lag_steps, fig_suff, probs, out_dir, plt_sett)

        set_mpl_prms(old_mpl_prms)
        return
