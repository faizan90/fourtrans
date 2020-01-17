'''
@author: Faizan-Uni

Jan 16, 2020

1:32:31 PM
'''
from math import ceil
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from ..simultexts.misc import print_sl, print_el

plt.ioff()


class PhaseAnnealingPlot:

    def __init__(self, verbose):

        assert isinstance(verbose, bool), 'verbose not a Boolean!'

        self._vb = verbose

        self._plt_in_h5_file = None
        self._plt_outputs_dir = None

        self._plt_input_set_flag = False
        self._plt_output_set_flag = False

        self._plt_verify_flag = False
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

        self._plot_tols_all(h5_hdl, opt_state_dir)

        self._plot_obj_vals_all(h5_hdl, opt_state_dir)

        self._plot_obj_vals_min(h5_hdl, opt_state_dir)

        self._plot_acpt_rates(h5_hdl, opt_state_dir)

        self._plot_phss_all(h5_hdl, opt_state_dir)

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

        self._plot_cmpr_scorr_asymms(h5_hdl, cmpr_dir)

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

    def _plot_tols_all(self, h5_hdl, out_dir):

        sim_grp_main = h5_hdl['data_sim_rltzns']

        beg_iters = h5_hdl['settings'].attrs['_sett_ann_obj_tol_iters']

        plt.figure(figsize=(20, 7))

        for rltzn_lab in sim_grp_main:
            tol_iters = np.arange(
                sim_grp_main[f'{rltzn_lab}/tols_all'].shape[0]) + beg_iters

            plt.plot(
                tol_iters,
                sim_grp_main[f'{rltzn_lab}/tols_all'],
                alpha=0.1,
                color='k')

        plt.ylim(0, plt.ylim()[1])

        plt.xlabel('Iteration')

        plt.ylabel(
            f'Mean absolute difference\nof previous {beg_iters} iterations')

        plt.grid()

        plt.savefig(
            str(out_dir / f'opt_state__tols_all.png'), bbox_inches='tight')

        plt.close()
        return

    def _plot_obj_vals_all(self, h5_hdl, out_dir):

        sim_grp_main = h5_hdl['data_sim_rltzns']

        plt.figure(figsize=(20, 7))

        for rltzn_lab in sim_grp_main:
            plt.plot(
                sim_grp_main[f'{rltzn_lab}/obj_vals_all'],
                alpha=0.1,
                color='k')

        plt.ylim(0, plt.ylim()[1])

        plt.xlabel('Iteration')

        plt.ylabel(f'Raw objective function value')

        plt.grid()

        plt.savefig(
            str(out_dir / f'opt_state__obj_vals_all.png'), bbox_inches='tight')

        plt.close()
        return

    def _plot_obj_vals_min(self, h5_hdl, out_dir):

        sim_grp_main = h5_hdl['data_sim_rltzns']

        plt.figure(figsize=(20, 7))

        for rltzn_lab in sim_grp_main:
            plt.plot(
                sim_grp_main[f'{rltzn_lab}/obj_vals_min'],
                alpha=0.1,
                color='k')

        plt.ylim(0, plt.ylim()[1])

        plt.xlabel('Iteration')

        plt.ylabel(f'Running minimum objective function value')

        plt.grid()

        plt.savefig(
            str(out_dir / f'opt_state__obj_vals_min.png'), bbox_inches='tight')

        plt.close()
        return

    def _plot_acpt_rates(self, h5_hdl, out_dir):

        sim_grp_main = h5_hdl['data_sim_rltzns']

        plt.figure(figsize=(20, 7))

        for rltzn_lab in sim_grp_main:
            plt.plot(
                sim_grp_main[f'{rltzn_lab}/acpt_rates'],
                alpha=0.1,
                color='k')

        plt.ylim(0, plt.ylim()[1])

        plt.xlabel('Iteration')

        plt.ylabel(f'Running mean acceptance rate')

        plt.grid()

        plt.savefig(
            str(out_dir / f'opt_state__acpt_rates.png'), bbox_inches='tight')

        plt.close()
        return

    def _plot_phss_all(self, h5_hdl, out_dir):

        sim_grp_main = h5_hdl['data_sim_rltzns']

        plt.figure(figsize=(20, 7))

        for rltzn_lab in sim_grp_main:
            plt.plot(
                sim_grp_main[f'{rltzn_lab}/phss_all'],
                alpha=0.1,
                color='k')

        plt.xlabel('Iteration')

        plt.ylabel(f'Phase')

        plt.grid()

        plt.savefig(
            str(out_dir / f'opt_state__phss_all.png'), bbox_inches='tight')

        plt.close()
        return

    def _plot_cmpr_scorr_asymms(self, h5_hdl, out_dir):

        axes = plt.subplots(1, 3, figsize=(18, 6))[1]

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
                alpha=0.3,
                color='k',
                label=label)

            axes[1].plot(
                lag_steps,
                sim_grp_main[f'{rltzn_lab}/asymms_1'],
                alpha=0.3,
                color='k',
                label=label)

            axes[2].plot(
                lag_steps,
                sim_grp_main[f'{rltzn_lab}/asymms_2'],
                alpha=0.3,
                color='k',
                label=label)

            leg_flag = False

        axes[0].plot(
            lag_steps,
            h5_hdl['data_ref_rltzn/_ref_scorrs'],
            alpha=0.7,
            color='r',
            label='ref')

        axes[1].plot(
            lag_steps,
            h5_hdl['data_ref_rltzn/_ref_asymms_1'],
            alpha=0.7,
            color='r',
            label='ref')

        axes[2].plot(
            lag_steps,
            h5_hdl['data_ref_rltzn/_ref_asymms_2'],
            alpha=0.7,
            color='r',
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
        return

    def _plot_cmpr_ecop_denss_base(
            self,
            lag_steps,
            fig_suff,
            vmin,
            vmax,
            ecop_denss,
            cmap_mappable_beta,
            out_dir):

        rows = int(ceil(lag_steps.size ** 0.5))
        cols = ceil(lag_steps.size / rows)

        fig, axes = plt.subplots(rows, cols, figsize=(15, 15))

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
                    cmap='Blues',
                    alpha=0.9,
                    vmin=vmin,
                    vmax=vmax)

                axes[row, col].set_aspect('equal')

                axes[row, col].text(
                    0.1, 0.85, f'{lag_steps[i]} step(s) lag', alpha=0.7)

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
            extend='max')

        plt.savefig(
            str(out_dir / f'cmpr__ecop_denss_{fig_suff}.png'),
            bbox_inches='tight')

        plt.close()
        return

    def _plot_cmpr_ecop_denss(self, h5_hdl, out_dir):

        lag_steps = h5_hdl['settings/_sett_obj_lag_steps']

        ecop_denss = h5_hdl['data_ref_rltzn/_ref_ecop_dens_arrs']

        vmin = 0.0
        vmax = np.max(ecop_denss) * 0.85

        fig_suff = 'ref'

        cmap_beta = plt.get_cmap('Blues')

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
            out_dir)

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
                out_dir)

        return

    def _plot_cmpr_ecop_scatter_base(
            self,
            lag_steps,
            fig_suff,
            probs,
            out_dir):

        rows = int(ceil(lag_steps.size ** 0.5))
        cols = ceil(lag_steps.size / rows)

        axes = plt.subplots(rows, cols, figsize=(15, 15))[1]

        row = 0
        col = 0
        for i in range(rows * cols):

            if i >= (lag_steps.size):
                axes[row, col].set_axis_off()

            else:
                rolled_probs = np.roll(probs, lag_steps[i])

                axes[row, col].scatter(probs, rolled_probs, alpha=0.4)

                axes[row, col].grid()

                axes[row, col].set_aspect('equal')

                axes[row, col].text(
                    0.05, 0.9, f'{lag_steps[i]} step(s) lag', alpha=0.7)

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

        lag_steps = h5_hdl['settings/_sett_obj_lag_steps']

        rnks = h5_hdl['data_ref_rltzn/_ref_rnk']
        probs = rnks / (rnks.size + 1)
        fig_suff = 'ref'

        self._plot_cmpr_ecop_scatter_base(lag_steps, fig_suff, probs, out_dir)

        sim_grp_main = h5_hdl['data_sim_rltzns']

        for rltzn_lab in sim_grp_main:
            rnks = sim_grp_main[f'{rltzn_lab}/rnk']
            probs = rnks / (rnks.size + 1)
            fig_suff = f'sim_{rltzn_lab}'

            self._plot_cmpr_ecop_scatter_base(
                lag_steps, fig_suff, probs, out_dir)

        return
