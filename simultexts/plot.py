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
from parse import search
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from pathos.multiprocessing import ProcessPool

from .misc import print_sl, print_el

plt.ioff()


class SimultaneousExtremesPlot:

    def __init__(self, verbose=True):

        self._vb = verbose

        self._h5_hdl = None

        self._n_cpus = 1

        self._mp_pool = None

        self._plot_freqs_flag = False
        self._plot_dendrs_flag = False

        self._sim_freq_figs_dir = None

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

            print(
                f'INFO: Set the number of running processes to: '
                f'{self._n_cpus}')

            print_el()

        self._set_misc_sett_flag = True
        return

    def set_plot_type_flags(self, frequencies_flag, dendrograms_flag):

        assert isinstance(frequencies_flag, bool), (
            'frequencies_flag not a boolean value!')

        assert isinstance(dendrograms_flag, bool), (
            'dendrograms_flag not a boolean value!')

        self._plot_freqs_flag = frequencies_flag
        self._plot_dendrs_flag = dendrograms_flag

        if self._vb:
            print_sl()

            print(
                f'INFO: Set the following plot flags:\n'
                f'\tPlot frequencies flag: {self._plot_freqs_flag}\n',
                f'\tPlot dendrograms flag: {self._plot_dendrs_flag}')

            print_el()
        return

    def verify(self):

        assert self._set_out_dir_flag, 'Outputs directory not set!'
        assert self._h5_path_set_flag, 'Path to HDF5 not set!'

        assert any([self._plot_freqs_flag, self._plot_dendrs_flag]), (
            'None of the plotting flags are True!')

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

        if self._plot_freqs_flag:
            self._sim_freq_figs_dir = self._out_dir / 'simultexts_freqs_figs'
            self._sim_freq_figs_dir.mkdir(exist_ok=True)

        if self._plot_dendrs_flag:
            self._sim_dendro_figs_dir = self._out_dir / 'simultexts_dendro_figs'
            self._sim_dendro_figs_dir.mkdir(exist_ok=True)
        return

    def plot(self):

        plot_beg_time = default_timer()

        assert self._set_plot_verify_flag, 'Unverified plotting state!'

        self._prepare()

        if self._vb:
            print_sl()

            print(f'Plotting simultaneous extremes\' simulation results...')

            print_el()

        sims_grp = self._h5_hdl['simultexts_sims']

        PSE = PlotSimultaneousExtremesMP(self)

        PSE_gen = (stn_comb for stn_comb in sims_grp)

        if self._mp_pool is not None:
            plot_rets = list(self._mp_pool.uimap(PSE.plot, PSE_gen))

            self._mp_pool.clear()

        else:
            plot_rets = list(map(PSE.plot, PSE_gen))

        if self._plot_dendrs_flag:
            plot_ret_dict = {}

            for plot_ret in plot_rets:
                plot_ret_dict.update(plot_ret)

            PSE.plot_dendrograms(plot_ret_dict)

        self._h5_hdl.close()
        self._h5_hdl = None

        if self._vb:
            print_sl()

            print(
                f'Done plotting simultaneous extremes\' simulation results\n'
                f'Total plotting time was: '
                f'{default_timer() - plot_beg_time:0.3f} seconds')

            print_el()
        return

    __verify = verify


class PlotSimultaneousExtremesMP:

    '''To be used in SimultaneousExtremesPlot only'''

    def __init__(self, SEP_cls):

        take_sep_cls_var_labs = [
            '_vb',
            '_rps',
            '_tws',
            '_n_cpus',
            '_out_dir',
            '_h5_path',
            '_h5_hdl',
            '_sim_freq_figs_dir',
            '_sim_dendro_figs_dir',
            '_plot_freqs_flag',
            '_plot_dendrs_flag',
            ]

        for _var in take_sep_cls_var_labs:
            setattr(self, _var, getattr(SEP_cls, _var))

        assert any([self._plot_freqs_flag, self._plot_dendrs_flag]), (
            'None of the plotting flags are True!')

        if self._n_cpus > 1:
            self._vb = False
            self._h5_hdl = None
        return

    def _prepare_data(self):

        if self._h5_hdl is None:
            self._h5_hdl = h5py.File(self._h5_path, 'r', driver='core')

        self._stn_comb_grp = self._h5_hdl['simultexts_sims'][self._stn_comb]

        self._ref_evts_arr = self._stn_comb_grp['ref_evts'][...]

        self._n_steps = self._stn_comb_grp['n_steps'][...]

        self._n_sims = self._stn_comb_grp['n_sims'][...]

        self._freqs_tups = {}

        FreqTup = namedtuple(
            'FreqTup', (
                'obs_vals',
                'sim_avgs',
                'sim_maxs',
                'sim_mins',
                'sim_stds',
                'mean_probs',
                'min_probs',
                'max_probs'))

        for stn in self._stn_labs:
            neb_evts_arr = self._stn_comb_grp[f'neb_evts_{stn}'][...]

            obs_vals = neb_evts_arr[0]

            sim_avgs = np.round(neb_evts_arr[1:].mean(axis=0)).astype(int)
            sim_maxs = np.round(neb_evts_arr[1:].max(axis=0)).astype(int)
            sim_mins = np.round(neb_evts_arr[1:].min(axis=0)).astype(int)
            sim_stds = np.round(neb_evts_arr[1:].std(axis=0), 2)

            mean_probs = np.round(
                sim_avgs / self._ref_evts_arr.reshape(-1, 1), 3)

            min_probs = np.round(
                sim_mins / self._ref_evts_arr.reshape(-1, 1), 3)

            max_probs = np.round(
                sim_maxs / self._ref_evts_arr.reshape(-1, 1), 3)

            self._freqs_tups[stn] = FreqTup(
                obs_vals,
                sim_avgs,
                sim_maxs,
                sim_mins,
                sim_stds,
                mean_probs,
                min_probs,
                max_probs,
                )

        if self._n_cpus > 1:
            self._h5_hdl.close()
            self._h5_hdl = None
        return

    def plot(self, stn_comb):

        self._stn_comb = stn_comb
        self._stn_labs = eval(self._stn_comb)

        assert len(self._stn_labs) == 2, 'Only configured for pairs!'

        self._prepare_data()

        if self._plot_freqs_flag:
            self._plot_frequencies()

        plot_ret = None
        if self._plot_dendrs_flag:
            plot_ret = {self._stn_comb:self._freqs_tups}

        return plot_ret

    def _get_rp_tw_dicts(self, stn_combs_data_dict):

        rp_tw_dicts = {}

        for rp_idx, rp in enumerate(self._rps):
            for tw_idx, tw in enumerate(self._tws):

                dendro_dict = {}
                for stn_comb in stn_combs_data_dict:
                    for stn in stn_combs_data_dict[stn_comb]:

                        crd_val = (stn_combs_data_dict[
                            stn_comb][stn].mean_probs[rp_idx, tw_idx])

                        dendro_dict[
                            f'{stn_comb}_{stn}_{crd_val:0.3f}'] = crd_val

#                         dendro_dict[f'{stn_comb}'] = 1 - (
#                             stn_combs_data_dict[stn_comb
#                                 ][stn].mean_probs[rp_idx, tw_idx])
#
#                         break

                rp_tw_dicts[f'{rp}_{tw}'] = dendro_dict

        return rp_tw_dicts

    def _plot_dendros(self, rp_tw_dicts):

        fig_size = (16, 8)

        prs_pt = '(\'{}\', \'{}\')_{}_{:0.3f}'

        for rp_tw_comb in rp_tw_dicts:
            dendro_labs = []
            for stn_comb_mean_prob in rp_tw_dicts[rp_tw_comb].keys():
                stn_1, stn_2, ref_stn, mean_prob = search(
                    prs_pt, stn_comb_mean_prob)

                dendro_labs.append(
                    f'{stn_1} & {stn_2} ({ref_stn}, {mean_prob})')

            mean_probs = np.array(list(rp_tw_dicts[rp_tw_comb].values()))

            linkage = hierarchy.linkage(mean_probs.reshape(-1, 1), 'median')

            plt.figure(figsize=fig_size)

            _dendro = hierarchy.dendrogram(
                linkage,
                labels=dendro_labs,
                leaf_rotation=90,
                )

            plt.xlabel(
                'Station combination '
                '(reference station, simulated probability)')

            plt.ylabel('Linkage distance')

            rp, tw = rp_tw_comb.split('_')

            plt.title(
                f'Simulated simultaneous extreme event occurence clusters '
                f'for event exeecedance probabilty: {rp} and time window: '
                f'{tw} steps')

            fig_name = f'dendrogram_EP{rp}_TW{tw}.png'
            fig_path = str(self._sim_dendro_figs_dir / fig_name)

            plt.tight_layout()

            plt.savefig(fig_path, bbox_inches='tight')
            plt.close()

        return

    def plot_dendrograms(self, stn_combs_data_dict):

        rp_tw_dicts = self._get_rp_tw_dicts(stn_combs_data_dict)

        self._plot_dendros(rp_tw_dicts)
        return

    def _plot_frequencies(self):

        # TODO: save tables as a csv

        stn_idxs_swth = (1, 0)

        TableTup = namedtuple('TableTup', ['i', 'j', 'tbl', 'lab'])

        n_fig_rows = 2
        n_fig_cols = 4
        fig_size = (15, 6)

        row_lab_strs = [
            f'{self._rps[i]} ({self._ref_evts_arr[i]})'
            for i in range(self._rps.shape[0])]

        col_hdr_clrs = [[0.75] * 4] * self._tws.shape[0]
        row_hdr_clrs = [[0.75] * 4] * self._rps.shape[0]

        max_tab_rows = self._rps.shape[0]

        for stn_idx, stn in enumerate(self._stn_labs):

            neb_stn = self._stn_labs[stn_idxs_swth[stn_idx]]

            obs_vals = self._freqs_tups[stn].obs_vals

            sim_avgs = self._freqs_tups[stn].sim_avgs
            sim_maxs = self._freqs_tups[stn].sim_maxs
            sim_mins = self._freqs_tups[stn].sim_mins
            sim_stds = self._freqs_tups[stn].sim_stds

            mean_probs = self._freqs_tups[stn].mean_probs
            min_probs = self._freqs_tups[stn].min_probs
            max_probs = self._freqs_tups[stn].max_probs

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
                TableTup(0, 3, min_probs, 'Min. simulated probability'),
                TableTup(1, 3, max_probs, 'Max. simulated probability'),
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
                    y_label = 'Exceedance Probability\n(No. of events)'

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
                f'No. of steps: {self._n_steps}, No. of simulations: '
                f'{self._n_sims}',
                x=0.5,
                y=n_tab_rows / max_tab_rows,
                va='bottom')

            plt.tight_layout()

            sim_freq_fig_name = f'simult_ext_stats_{stn}_{neb_stn}.png'

            plt.savefig(
                str(self._sim_freq_figs_dir / sim_freq_fig_name),
                bbox_inches='tight')
            plt.close()
        return
