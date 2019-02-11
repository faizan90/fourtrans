'''
Created on Feb 7, 2019

@author: Faizan-Uni
'''

import psutil
from timeit import default_timer
from pathlib import Path
from collections import namedtuple

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathos.multiprocessing import ProcessPool

from .misc import print_sl, print_el

plt.ioff()


class SimultaneousExtremesPlot:

    def __init__(self, verbose=True):

        self._vb = verbose

        self._h5_hdl = None

        self._n_cpus = 1

        self._mp_pool = None

        self._set_out_dir_flag = False
        self._h5_path_set_flag = False
        self._set_misc_sett_flag = False
        self._set_plot_verify_flag = False
        return

    def set_outputs_directory(self, out_dir):

        assert isinstance(out_dir, (Path, str)), (
            'out_dir not a string or a Path-like object!')

        out_dir = Path(out_dir).absolute()

        assert out_dir.parents[0].exists(), (
            'Parent directory of the out_dir does not exist!')

        self._out_dir = out_dir

        if self._vb:
            print_sl()

            print('INFO: Set the plotting outputs directory as following:')
            print('\t', f'{str(self._out_dir)}')

            print_el()

        self._set_out_dir_flag = True
        return

    def set_hdf5_path(self, hdf5_path):

        assert isinstance(hdf5_path, (str, Path)), (
            'hdf5_path not a string or Path object!')

        hdf5_path = Path(hdf5_path).absolute()

        assert hdf5_path.exists(), 'Given hdf5_path does not exists!'
        assert hdf5_path.is_file(), 'Given hdf5_path not a file!'

        self._h5_path = hdf5_path

        self._h5_path_set_flag = True
        return

    def set_misc_settings(self, n_cpus):

        if isinstance(n_cpus, int):
            assert n_cpus > 0, 'n_cpus has to be one or more!'

        elif isinstance(n_cpus, str):
            assert n_cpus == 'auto'

            n_cpus = max(1, psutil.cpu_count() - 1)

        else:
            raise AssertionError('n_cpus can be an integer or \'auto\' only!')

        self._n_cpus = n_cpus

        if self._vb:
            print_sl()

            print(f'INFO: Set the number of running processes to: '
                f'{self._n_cpus}')

            print_el()

        self._set_misc_sett_flag = True
        return

    def verify(self):

        assert self._set_out_dir_flag, 'Outputs directory not set!'
        assert self._h5_path_set_flag, 'Path to HDF5 not set!'

        self._set_plot_verify_flag = True
        return

    def _prepare(self):

        if not self._out_dir.exists():
            self._out_dir.mkdir(exist_ok=True)

        if (self._n_cpus > 1) and (self._mp_pool is None):
            self._mp_pool = ProcessPool(self._n_cpus)

        if self._h5_hdl is None:
            self._h5_hdl = h5py.File(self._h5_path, 'r', driver='core')

        self._rps = self._h5_hdl['return_periods'][...]
        self._tws = self._h5_hdl['time_windows'][...]

        return

    def plot_simultext_probs(self):

        plot_beg_time = default_timer()

        assert self._set_plot_verify_flag, 'Unverified plotting state!'

        self._prepare()

        if self._vb:
            print_sl()

            print(f'Plotting simultaneous extremes\' frequency...')

            print_el()

        sims_grp = self._h5_hdl['simultexts_sims']

        PSE = PlotSimultaneousExtremesMP(self)

        PSE_gen = (stn_comb for stn_comb in sims_grp)

        if self._mp_pool is not None:
            list(self._mp_pool.uimap(PSE.plot, PSE_gen))

        else:
            list(map(PSE.plot, PSE_gen))

        self._h5_hdl.close()
        self._h5_hdl = None

        if self._vb:
            print_sl()
            print(
                f'Done plotting simultaneous extremes\' frequency\n'
                f'Total plotting time was: '
                f'{default_timer() - plot_beg_time:0.3f} seconds')
            print_el()
        return

    __verify = verify


class PlotSimultaneousExtremesMP:

    def __init__(self, SEP_cls):

        take_sep_cls_var_labs = [
            '_vb',
            '_rps',
            '_tws',
            '_n_cpus',
            '_out_dir',
            '_h5_path',
            ]

        for _var in take_sep_cls_var_labs:
            setattr(self, _var, getattr(SEP_cls, _var))

        if self._n_cpus > 1:
            self._vb = False
        return

    def plot(self, stn_comb):

        self._h5_hdl = h5py.File(self._h5_path, 'r', driver=None)

        stn_comb_grp = self._h5_hdl['simultexts_sims'][stn_comb]

        ref_evts_arr = stn_comb_grp[f'ref_evts'][...]
        n_steps = stn_comb_grp[f'n_steps'][...]

        sim_figs_dir = self._out_dir / 'simultexts_figs'
        sim_figs_dir.mkdir(exist_ok=True)

        stn_idxs_swth = [1, 0]

        TableTup = namedtuple('TableTup', ['i', 'j', 'tbl', 'lab'])

        n_fig_rows = 2
        n_fig_cols = 3
        fig_size = (15, 6)

        row_lab_strs = [
            f'{self._rps[i]} ({ref_evts_arr[i]})'
            for i in range(self._rps.shape[0])]

        col_hdr_clrs = [[0.75] * 4] * self._tws.shape[0]
        row_hdr_clrs = [[0.75] * 4] * self._rps.shape[0]

        max_tab_rows = self._rps.shape[0]

        stn_labs = eval(stn_comb)

        assert len(stn_labs) == 2, 'Only configured for pairs!'

        for stn_idx, stn in enumerate(stn_labs):

            neb_stn = stn_labs[stn_idxs_swth[stn_idx]]

            neb_evts_arr = stn_comb_grp[f'neb_evts_{stn}'][...]

            n_sims = neb_evts_arr.shape[0] - 1

            obs_vals = neb_evts_arr[0]

            sim_avgs = np.round(neb_evts_arr[1:].mean(axis=0)).astype(int)
            sim_maxs = np.round(neb_evts_arr[1:].max(axis=0)).astype(int)
            sim_mins = np.round(neb_evts_arr[1:].min(axis=0)).astype(int)
            sim_stds = np.round(neb_evts_arr[1:].std(axis=0), 2)

            mean_probs = np.round(
                sim_avgs / ref_evts_arr.reshape(-1, 1), 3)

            ax_arr = plt.subplots(
                nrows=n_fig_rows,
                ncols=n_fig_cols,
                sharex=True,
                sharey=True,
                figsize=fig_size)[1]

            tbls = [
                TableTup(0, 0, obs_vals, 'Observed frequency'),
                TableTup(0, 1, sim_avgs, 'Mean simulated frequency'),
                TableTup(0, 2, mean_probs, 'Mean simulated probability'),
                TableTup(1, 0, sim_mins, 'Minimum simulated frequency'),
                TableTup(1, 1, sim_maxs, 'Maximum simulated frequency'),
                TableTup(1, 2, sim_stds, 'Simulated frequencies\' Std.'),
                ]

            for tbl in tbls:
                ax = ax_arr[tbl.i, tbl.j]

                if not tbl.i:
                    tcol_labs = self._tws
                    x_label = None
                    col_colors = col_hdr_clrs
                    n_tab_rows = self._rps.shape[0] + 1

                else:
                    tcol_labs = None
                    x_label = 'Time window'
                    col_colors = None

                    n_tab_rows = self._rps.shape[0]

                if not tbl.j:
                    trow_labs = row_lab_strs
                    row_colors = row_hdr_clrs

                else:
                    trow_labs = None
                    row_colors = None

                if tbl.j == (n_fig_cols - 1):
                    y_label = 'Return period (No. of events)'

                else:
                    y_label = None

                ax.table(
                    cellText=tbl.tbl,
                    loc='center',
                    bbox=[0, 0, 1.0, n_tab_rows / max_tab_rows],
                    rowLabels=trow_labs,
                    colLabels=tcol_labs,
                    rowColours=row_colors,
                    colColours=col_colors,
                    )

                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)

                ax.set_xticks([])
                ax.set_yticks([])

                ttl = ax.set_title(tbl.lab)
                ttl.set_position(
                    (ttl.get_position()[0], n_tab_rows / max_tab_rows))

                ax.yaxis.set_label_position('right')

                [ax.spines[spine].set_visible(False)
                 for spine in ax.spines]

            plt.suptitle(
                f'Reference station: {stn}, Nebor station: {neb_stn}\n'
                f'No. of steps: {n_steps}, No. of simulations: {n_sims}',
                x=0.5,
                y=n_tab_rows / max_tab_rows,
                va='bottom')

            plt.tight_layout()

            plt.savefig(
                sim_figs_dir / f'simult_ext_stats_{stn}_{neb_stn}.png',
                bbox_inches='tight')
            plt.close()

        if self._n_cpus > 1:
            self._h5_hdl.close()
            self._h5_hdl = None
        return
