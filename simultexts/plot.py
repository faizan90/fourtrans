'''
Created on Feb 7, 2019

@author: Faizan-Uni
'''
import psutil
from pathlib import Path
from timeit import default_timer
from collections import namedtuple

import h5py
import numpy as np
import pandas as pd
import shapefile as shpf
import matplotlib.pyplot as plt
from descartes import PolygonPatch
from pathos.multiprocessing import ProcessPool

from .misc import print_sl, print_el, ret_mp_idxs

plt.ioff()


class SimultaneousExtremesPlot:

    def __init__(self, verbose=True):

        self._vb = verbose

        self._h5_hdl = None

        self._n_cpus = 1
        self._n_cpus_extra = 0

        self._mp_pool = None

        self._stn_comb_freq_tups = None
        self._cluster_feats_dict = None

        self._plot_freqs_flag = False
        self._plot_clusters_flag = False
        self._plot_sim_cdfs_flag = False
        self._plot_auto_corrs_flag = False
        self._plot_ft_cumm_corrs_flag = False

        self._out_dirs_dict = {}
        self._clusters_shp_loc = None
        self._clusters_shp_fld = None

        self._set_out_dir_flag = False
        self._h5_path_set_flag = False
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
                f'INFO: Set the number of running processes to '
                f'{self._n_cpus}')

            print_el()
        return

    def set_plot_type_flags(
            self,
            frequencies_flag=False,
            sim_cdfs_flag=False,
            sim_auto_corrs_flag=False,
            sim_ft_corrs_flag=False):

        assert isinstance(frequencies_flag, bool), (
            'frequencies_flag not a boolean value!')

        assert isinstance(sim_cdfs_flag, bool), (
            'sim_cdfs_flag not a boolean value!')

        assert isinstance(sim_auto_corrs_flag, bool), (
            'sim_auto_corrs_flag not a boolean value!')

        assert isinstance(sim_ft_corrs_flag, bool), (
            'sim_ft_corrs_flag not a boolean value!')

        self._plot_freqs_flag = frequencies_flag
        self._plot_sim_cdfs_flag = sim_cdfs_flag
        self._plot_auto_corrs_flag = sim_auto_corrs_flag
        self._plot_ft_cumm_corrs_flag = sim_ft_corrs_flag

        if self._vb:
            print_sl()

            print(
                f'INFO: Set the following plot flags:\n'
                f'\tPlot frequencies flag: {self._plot_freqs_flag}\n',
                f'\tPlot simulation CDFs flag: {self._plot_sim_cdfs_flag}\n',
                f'\tPlot simulation auto corrs flag: '
                f'{self._plot_auto_corrs_flag}\n',
                f'\tPlot Fourier cummulative correlations: '
                f'{self._plot_ft_cumm_corrs_flag}')

            print_el()
        return

    def set_clusters_shapefile_path(self, path_to_shapefile, label_field):

        assert isinstance(path_to_shapefile, (str, Path)), (
            'path_to_shapefile not a string or a pathlib.Path object!')

        path_to_shapefile = Path(path_to_shapefile).absolute()

        assert path_to_shapefile.exists(), (
            'path_to_shapefile does not exist!')

        assert isinstance(label_field, str), 'label_field not a string!'

        self._clusters_shp_loc = path_to_shapefile
        self._clusters_shp_fld = label_field

        if self._vb:
            print_sl()

            print(
                f'INFO: Set the path to clusters shapefile as: '
                f'{self._clusters_shp_loc}')

            print_el()

        self._plot_clusters_flag = True
        return

    def verify(self):

        assert self._set_out_dir_flag, 'Outputs directory not set!'
        assert self._h5_path_set_flag, 'Path to HDF5 not set!'

        assert any([
            self._plot_freqs_flag,
            self._plot_clusters_flag,
            self._plot_sim_cdfs_flag,
            self._plot_auto_corrs_flag,
            self._plot_ft_cumm_corrs_flag]), (
                'None of the plotting flags are True!')

        self._set_plot_verify_flag = True
        return

    def plot(self):

        plot_beg_time = default_timer()

        assert self._set_plot_verify_flag, 'Unverified plotting state!'

        self._prepare()

        if self._vb:
            print_sl()

            print(f'Plotting simultaneous extremes\' simulation results...')

            print_el()

        sims_grp = list(self._h5_hdl['simultexts_sims'].keys())

        self._h5_hdl.close()
        self._h5_hdl = None

        # have before initiating PSE
        self._n_cpus_extra = max(0, self._n_cpus - len(sims_grp))

        self._n_cpus = self._n_cpus - self._n_cpus_extra

        if (self._n_cpus > 1) and (self._mp_pool is None):
            self._mp_pool = ProcessPool(self._n_cpus)

        PSEPD = PrepareSimultaneousExtremesPlottingData(self)

        self._stn_comb_freq_tups = PSEPD.get_data()

        if self._plot_clusters_flag:
            self._cluster_feats_dict = PSEPD.get_cluster_plot_data()

        PSE = PlotSimultaneousExtremesMP(self)

        PSE_gen = (item for item in self._stn_comb_freq_tups.items())

        if self._mp_pool is not None:
            list(self._mp_pool.uimap(PSE.plot, PSE_gen))

            self._mp_pool.clear()

        else:
            list(map(PSE.plot, PSE_gen))

        if self._mp_pool is not None:
            self._mp_pool.join()
            self._mp_pool = None

        if self._vb:
            print_sl()

            print(
                f'Done plotting simultaneous extremes\' simulation results\n'
                f'Total plotting time was: '
                f'{default_timer() - plot_beg_time:0.3f} seconds')

            print_el()
        return

    def _var_chk(self):

        main_vars = [
            'excd_probs',
            'time_windows',
            'n_sims',
            'simultexts_sims']

        exst_list = [x in self._h5_hdl for x in main_vars]

        if not all(exst_list):
            print_sl()

            print('WARNING: The following variables are not in the HDF5:')
            print(
                '\t',
                [main_vars[i]
                 for i in range(len(main_vars)) if not exst_list[i]])

            print_el()

        assert all(exst_list), (
            'Some or all the required variables are not in '
            'the input HDF5 file!')
        return

    def _prepare(self):

        if not self._out_dir.exists():
            self._out_dir.mkdir(exist_ok=True)

        if self._h5_hdl is None:
            self._h5_hdl = h5py.File(self._h5_path, 'r', driver=None)

        self._var_chk()

        self._eps = self._h5_hdl['excd_probs'][...]
        self._tws = self._h5_hdl['time_windows'][...]
        self._n_sims = self._h5_hdl['n_sims'][...]

        saved_sim_cdfs_flag = bool(self._h5_hdl['save_sim_cdfs_flag'][...])

        saved_sim_acorrs_flag = bool(
            self._h5_hdl['save_sim_acorrs_flag'][...])

        saved_sim_ft_cumm_corrs_flag = bool(
            self._h5_hdl['save_ft_cumm_corrs_flag'][...])

        if self._plot_freqs_flag:
            self._out_dirs_dict['freq_figs'] = (
                self._out_dir / 'simultexts_freqs_figs')

            self._out_dirs_dict['freq_figs'].mkdir(exist_ok=True)

            self._out_dirs_dict['freq_tabs'] = (
                self._out_dir / 'simultexts_freqs_tables')

            self._out_dirs_dict['freq_tabs'].mkdir(exist_ok=True)

        if self._plot_clusters_flag:
            self._out_dirs_dict['binary_cluster_figs'] = (
                self._out_dir / 'simultexts_binary_cluster_figs')

            self._out_dirs_dict['binary_cluster_figs'].mkdir(exist_ok=True)

            self._out_dirs_dict['nD_cluster_figs'] = (
                self._out_dir / 'simultexts_nD_cluster_figs')

            self._out_dirs_dict['nD_cluster_figs'].mkdir(exist_ok=True)

        if self._plot_sim_cdfs_flag:
            assert saved_sim_cdfs_flag, (
                'CDFs data not saved inside the HDF5!')

            self._out_dirs_dict['cdfs_figs'] = (
                self._out_dir / 'simultexts_sim_cdfs_figs')

            self._out_dirs_dict['cdfs_figs'].mkdir(exist_ok=True)

        if self._plot_auto_corrs_flag:
            assert saved_sim_acorrs_flag, (
                'Auto correlation data not saved inside the HDF5!')

            self._out_dirs_dict['pcorr_figs'] = (
                self._out_dir / 'simultexts_sim_pcorr_figs')

            self._out_dirs_dict['pcorr_figs'].mkdir(exist_ok=True)

            self._out_dirs_dict['scorr_figs'] = (
                self._out_dir / 'simultexts_sim_scorr_figs')

            self._out_dirs_dict['scorr_figs'].mkdir(exist_ok=True)

        if self._plot_ft_cumm_corrs_flag:
            assert saved_sim_ft_cumm_corrs_flag, (
                'Fourier cummulative correlation not saved inside HDF5!')

            self._out_dirs_dict['ft_corr_figs'] = (
                self._out_dir / 'simultexts_ft_pcorr_figs')

            self._out_dirs_dict['ft_corr_figs'].mkdir(exist_ok=True)
        return

    __verify = verify


class PrepareSimultaneousExtremesPlottingData:

    def __init__(self, SEP_cls):

        take_sep_cls_var_labs = [
            '_eps',
            '_tws',
            '_h5_path',
            '_plot_freqs_flag',
            '_plot_clusters_flag',
            '_clusters_shp_loc',
            '_clusters_shp_fld',
            '_out_dirs_dict',
            ]

        for _var in take_sep_cls_var_labs:
            setattr(self, _var, getattr(SEP_cls, _var))

        self._FreqTup = namedtuple(
            'FreqTup', (
                'obs_vals',
                'sim_avgs',
                'sim_maxs',
                'sim_mins',
                'sim_stds',
                'avg_probs',
                'min_probs',
                'max_probs',

                ))

        self._StnCombData = namedtuple(
            'StnCombData', (
                'freqs_tups',
                'ref_evts_arr',
                'ref_evts_ext_arr',
                'n_steps',
                'n_steps_ext',
                'comb_lab'))

        self._h5_hdl = None
        self._stn_combs = None
        return

    def get_data(self):

        self._h5_hdl = h5py.File(self._h5_path, 'r', driver=None)

        stn_combs = list(self._h5_hdl['simultexts_sims'].keys())

        stn_comb_freq_tups = {}
        for i, stn_comb in enumerate(stn_combs):
            stn_comb_freq_tups[stn_comb] = self._get_comb_data(
                stn_comb, f'C{i:02d}')

        self._h5_hdl.close()
        self._h5_hdl = None
        return stn_comb_freq_tups

    def _get_comb_data(self, stn_comb, comb_lab):

        '''
        Called only when freqs or clusters flags are True.

        The data is for a given combination only. Each call gets there own
        space. The combined output for each combination can then be used
        by plot_clusters.
        '''

        stn_labs = eval(stn_comb)

        stn_idxs_swth_dict = {}
        for i, ref_stn in enumerate(stn_labs):
            bools = np.ones(len(stn_labs), dtype=bool)
            bools[i] = False
            stn_idxs_swth_dict[ref_stn] = np.array(stn_labs)[bools]

        assert stn_comb in self._h5_hdl['simultexts_sims'], (
            f'Given combination {self._stn_comb} not in the input HDF5!')

        stn_comb_grp = self._h5_hdl['simultexts_sims'][stn_comb]

        assert 'ref_evts' in stn_comb_grp, (
            f'Required variable ref_evts for the given combination '
            f'not in the input HDF5!')

        assert 'ref_evts_ext' in stn_comb_grp, (
            f'Required variable ref_evts_ext for the given combination '
            f'not in the input HDF5!')

        assert 'n_steps' in stn_comb_grp, (
            f'Required variable n_steps for the given combination '
            f'not in the input HDF5!')

        assert 'n_steps_ext' in stn_comb_grp, (
            f'Required variable n_steps_ext for the given combination '
            f'not in the input HDF5!')

        ref_evts_arr = stn_comb_grp['ref_evts'][...]
        ref_evts_ext_arr = stn_comb_grp['ref_evts_ext'][...]

        n_steps = stn_comb_grp['n_steps'][...]
        n_steps_ext = stn_comb_grp['n_steps_ext'][...]

        _tab_labs = [
            'obs_frq',
            'avg_sim_freq',
            'min_sim_freq',
            'max_sim_freq',
            'std_sim_freq',
            'avg_sim_prob',
            'min_sim_prob',
            'max_sim_prob']

        tws_tile = np.tile(self._tws, len(_tab_labs))
        tws_rpt = np.repeat(_tab_labs, self._tws.shape[0])

        assert tws_rpt.shape[0] == tws_tile.shape[0]

        tab_header = ['exd_prob', 'n_ref_evts', 'n_ref_ext_evts'] + [
            f'{tws_rpt[i]}_TW{tws_tile[i]}'
            for i in range(tws_tile.shape[0])]

        ref_evts_rshp = ref_evts_arr.reshape(-1, 1).copy()
        ref_evts_rshp[~ref_evts_rshp.astype(bool)] = 1
        ref_evts_ext_scl_rshp = (
            ref_evts_ext_arr.reshape(-1, 1) / ref_evts_rshp)

        ref_evts_ext_scl_rshp[~np.isfinite(ref_evts_ext_scl_rshp)] = 1
        ref_evts_ext_scl_rshp[~ref_evts_ext_scl_rshp.astype(bool)] = 1

        freqs_tups = {}

        for ref_stn in stn_labs:
            assert f'neb_evts_{ref_stn}' in stn_comb_grp, (
                f'Required variable neb_evts_{ref_stn} not in the '
                f'input HDF5!')

            neb_evts_arr = stn_comb_grp[f'neb_evts_{ref_stn}'][...]

            for neb_stn_i, neb_stn in enumerate(
                stn_idxs_swth_dict[ref_stn]):

                obs_vals = neb_evts_arr[0, neb_stn_i]

                sim_avgs = np.round(
                    neb_evts_arr[1:, neb_stn_i].mean(axis=0) /
                    ref_evts_ext_scl_rshp).astype(int)

                sim_maxs = np.round(
                    neb_evts_arr[1:, neb_stn_i].max(axis=0) /
                    ref_evts_ext_scl_rshp).astype(int)

                sim_mins = np.round(
                    neb_evts_arr[1:, neb_stn_i].min(axis=0) /
                    ref_evts_ext_scl_rshp).astype(int)

                sim_stds = np.round(
                    neb_evts_arr[1:, neb_stn_i].std(axis=0) /
                     ref_evts_ext_scl_rshp, 2)

                avg_probs = np.round(sim_avgs / ref_evts_rshp , 3)
                min_probs = np.round(sim_mins / ref_evts_rshp , 3)
                max_probs = np.round(sim_maxs / ref_evts_rshp , 3)

                freqs_tups[(ref_stn, neb_stn)] = self._FreqTup(
                    obs_vals,
                    sim_avgs,
                    sim_maxs,
                    sim_mins,
                    sim_stds,
                    avg_probs,
                    min_probs,
                    max_probs,
                    )

                if not self._plot_freqs_flag:
                    continue

                table_concat = np.concatenate((
                    self._eps.reshape(-1, 1),
                    ref_evts_arr.reshape(-1, 1),
                    ref_evts_ext_arr.reshape(-1, 1),
                    obs_vals,
                    sim_avgs,
                    sim_mins,
                    sim_maxs,
                    sim_stds,
                    avg_probs,
                    min_probs,
                    max_probs,
                    ), axis=1)

                out_stats_df = pd.DataFrame(
                    data=table_concat,
                    columns=tab_header)

                tab_name = (
                    f'simult_ext_stats_{comb_lab}_{ref_stn}_{neb_stn}.csv')

                out_stats_df.to_csv(
                    self._out_dirs_dict['freq_tabs'] / tab_name,
                    sep=';',
                    float_format='%0.8f')

        return self._StnCombData(
            freqs_tups,
            ref_evts_arr,
            ref_evts_ext_arr,
            n_steps,
            n_steps_ext,
            comb_lab)

    def get_cluster_plot_data(self):

        shp_ds = shpf.Reader(str(self._clusters_shp_loc))
        shp_fields = [str(field[0]) for field in shp_ds.fields[1:]]

        lab_fld_idx = shp_fields.index(self._clusters_shp_fld)

        assert shp_fields

        assert self._clusters_shp_fld in shp_fields, (
            'Given label_field not in the shapefile!')

        shp_xx = {}
        shp_yy = {}
        patches = {}

        names = []

        for shape in shp_ds.shapeRecords():
            assert 'Polygon' in shape.shape.__geo_interface__['type'], (
                'Shape not a polygon!')

            name = str(shape.record[lab_fld_idx])

            names.append(name)

            shp_xx[name] = np.array([i[0] for i in shape.shape.points[:]])
            shp_yy[name] = np.array([i[1] for i in shape.shape.points[:]])
            patches[name] = shape.shape.__geo_interface__

        cluster_feats_dict = {
            'xx': shp_xx,
            'yy': shp_yy,
            'names': names,
            'patches': patches,
            'xx_mean': {name:np.median(shp_xx[name]) for name in names},
            'yy_mean': {name:np.median(shp_yy[name]) for name in names}}

        shp_ds = None
        return cluster_feats_dict


class PlotSimultaneousExtremesMP:

    '''To be used in SimultaneousExtremesPlot only'''

    def __init__(self, SEP_cls):

        take_sep_cls_var_labs = [
            '_vb',
            '_eps',
            '_tws',
            '_n_sims',
            '_n_cpus',
            '_n_cpus_extra',
            '_h5_path',
            '_out_dirs_dict',
            '_plot_freqs_flag',
            '_plot_clusters_flag',
            '_plot_sim_cdfs_flag',
            '_plot_auto_corrs_flag',
            '_plot_ft_cumm_corrs_flag',
            '_cluster_feats_dict',
            ]

        for _var in take_sep_cls_var_labs:
            setattr(self, _var, getattr(SEP_cls, _var))

        assert any([
            self._plot_freqs_flag,
            self._plot_clusters_flag,
            self._plot_sim_cdfs_flag,
            self._plot_auto_corrs_flag,
            self._plot_ft_cumm_corrs_flag]), (
                'None of the plotting flags are True!')

        if self._n_cpus > 1:
            self._vb = False

        self._h5_hdl = None
        return

    def plot(self, stn_comb_data_item):

        stn_comb, stn_comb_data = stn_comb_data_item

        stn_labs = np.array(eval(stn_comb))

        stn_idxs_swth_dict = {}
        for i, stn in enumerate(stn_labs):
            bools = np.ones(len(stn_labs), dtype=bool)
            bools[i] = False
            stn_idxs_swth_dict[stn] = np.array(stn_labs)[bools]

        n_extra_cpus_per_comb = self._n_cpus_extra // self._n_cpus

        if n_extra_cpus_per_comb:
            stn_chunks_idxs = ret_mp_idxs(
                len(stn_labs), n_extra_cpus_per_comb)

            sub_mp_pool = ProcessPool(n_extra_cpus_per_comb)

            self._vb = False

        else:
            stn_chunks_idxs = np.array([0, len(stn_labs)])

            sub_mp_pool = None

        plot_gen = (
            (stn_labs[stn_chunks_idxs[i]:stn_chunks_idxs[i + 1]],
             stn_comb_data,
             stn_comb,
             stn_idxs_swth_dict)

            for i in range(stn_chunks_idxs.shape[0] - 1))

        if sub_mp_pool is not None:
            list(sub_mp_pool.uimap(self._plot, plot_gen))

        else:
            list(map(self._plot, plot_gen))

        if self._plot_clusters_flag:
            self._plot_nD_clusters((stn_comb, stn_comb_data.comb_lab))
        return

    def _plot(self, args):

        (ref_stns,
         stn_comb_data,
         stn_comb,
         stn_idxs_swth_dict) = args

        h5_hdl = h5py.File(self._h5_path, 'r', driver=None)
        stn_comb_grp = h5_hdl['simultexts_sims'][stn_comb]

        for ref_stn in ref_stns:
            if self._plot_freqs_flag:
                self._plot_frequencies(
                    ref_stn, stn_comb_data, stn_idxs_swth_dict)

            if self._plot_clusters_flag:
                self._plot_binary_clusters(
                    ref_stn,
                    stn_idxs_swth_dict,
                    stn_comb_data.freqs_tups,
                    stn_comb_data.comb_lab)

            if self._plot_ft_cumm_corrs_flag:
                self._plot_ft_cumm_diff_corrs(
                    ref_stn, stn_comb_grp, stn_comb_data, 'cumm')

                self._plot_ft_cumm_diff_corrs(
                    ref_stn, stn_comb_grp, stn_comb_data, 'diff')

            if self._plot_sim_cdfs_flag or self._plot_auto_corrs_flag:
                self._plot_sim_cdfs__corrs(
                    ref_stn, stn_comb_grp, stn_comb_data)

        h5_hdl.close()
        return

    def _get_cluster_simult_evts_data(self, stn_comb):

        h5_hdl = h5py.File(self._h5_path, 'r', driver=None)
        stn_comb_grp = h5_hdl['simultexts_sims'][stn_comb]

        assert 'simult_ext_evts_cts' in stn_comb_grp, (
            f'Required variable simult_ext_evts_cts for the given '
            f'combination not in the input HDF5!')

        assert 'all_stn_combs' in stn_comb_grp, (
            f'Required variable all_stn_combs for the given '
            f'combination not in the input HDF5!')

        simult_ext_evts_cts = stn_comb_grp['simult_ext_evts_cts'][...]
        all_stn_combs = stn_comb_grp['all_stn_combs'][...]

        h5_hdl.close()
        return (simult_ext_evts_cts, all_stn_combs)

    def _plot_ft_cumm_diff_corrs(
            self, ref_stn, stn_comb_grp, stn_comb_data, corr_type):

        fig_size = (18, 8)

        comb_lab = stn_comb_data.comb_lab

        ft_ccorrs_arr = stn_comb_grp[f'ft_ccorrs_{ref_stn}'][...]

        n_ft_sims, n_ft_corrs = ft_ccorrs_arr.shape

        n_zoom_freqs = max(int(0.01 * n_ft_corrs), 2)

        obs_corrs = ft_ccorrs_arr[0, :]
        sim_corrs = ft_ccorrs_arr[1:, :].reshape(-1, n_ft_corrs)

        if corr_type == 'cumm':
            type_labs = ['Cummulative', 'cumm']

        elif corr_type == 'diff':

            type_labs = ['Differential', 'diff']

            obs_corrs = np.concatenate(
                ([obs_corrs[0]], obs_corrs[1:] - obs_corrs[:-1]))

            sim_corrs = np.concatenate(
                (sim_corrs[:, 0].reshape(n_ft_sims - 1, 1),
                 (sim_corrs[:, 1:] - sim_corrs[:, :-1]).reshape(
                     n_ft_sims - 1, n_ft_corrs - 1)),
                axis=1)

        else:
            raise ValueError(corr_type)

        min_sim_corrs = sim_corrs.min(axis=0)
        max_sim_corrs = sim_corrs.max(axis=0)
        mean_sim_corrs = sim_corrs.mean(axis=0)

        corr_within_bds_arr = np.ones_like(obs_corrs, dtype=int)
        corr_within_bds_arr[obs_corrs < min_sim_corrs] = 0
        corr_within_bds_arr[obs_corrs > max_sim_corrs] = 2

        pcent_abv_sim = 100 * (corr_within_bds_arr == 2).sum() / n_ft_corrs
        pcent_within_sim = 100 * (corr_within_bds_arr == 1).sum() / n_ft_corrs
        pcent_bel_sim = 100 * (corr_within_bds_arr == 0).sum() / n_ft_corrs

        fig = plt.figure(figsize=fig_size)

        axs = (
            plt.subplot2grid((6, 2), (0, 0), 5, 1, fig=fig),
            plt.subplot2grid((6, 2), (5, 0), 1, 1, fig=fig),
            plt.subplot2grid((6, 2), (0, 1), 5, 1, fig=fig),
            plt.subplot2grid((6, 2), (5, 1), 1, 1, fig=fig))

        axs[0].plot(
            obs_corrs, color='r', alpha=0.9, lw=1, label='obs.', zorder=3)

        axs[0].plot(
            mean_sim_corrs,
            color='b',
            alpha=0.6,
            lw=2.0,
            label='sim_mean',
            zorder=2)

        axs[0].fill_between(
            np.arange(n_ft_corrs),
            min_sim_corrs,
            max_sim_corrs,
            color='b',
            alpha=0.3,
            label='sim_bds',
            zorder=1)

        axs[0].set_ylabel(f'{type_labs[0]} correlation (-)')

        axs[0].set_xticklabels([])

        axs[0].legend()
        axs[0].grid()

        axs[1].plot(corr_within_bds_arr, color='C0', alpha=0.7, lw=1)
        axs[1].set_yticks(np.arange(3))
        axs[1].set_yticklabels(
            ['Obs. below sim.', 'Obs. within sim.', 'Obs. above sim.'])

        axs[1].grid()

        axs[1].set_xlabel('Frequency (steps)')

        axs[2].plot(
            obs_corrs[:n_zoom_freqs],
            color='r',
            alpha=0.9,
            lw=1,
            label='obs.',
            zorder=3)

        axs[2].plot(
            mean_sim_corrs[:n_zoom_freqs],
            color='b',
            alpha=0.6,
            lw=2.0,
            label='sim_mean',
            zorder=2)

        axs[2].fill_between(
            np.arange(n_zoom_freqs),
            min_sim_corrs[:n_zoom_freqs],
            max_sim_corrs[:n_zoom_freqs],
            color='b',
            alpha=0.3,
            label='sim_bds',
            zorder=1)

        axs[2].set_xticklabels([])

        axs[2].grid()

        axs[2].yaxis.tick_right()

        axs[3].plot(
            corr_within_bds_arr[:n_zoom_freqs],
            color='C0',
            alpha=0.7,
            lw=1)

        axs[3].set_yticks(np.arange(3))
        axs[3].set_yticklabels([])

        axs[3].grid()

        axs[3].set_xlabel('Frequency (steps)')

        plt.suptitle(
            f'{type_labs[0]} Fourier transform pearson correlations '
            f'for observed and simulated series of station {ref_stn} '
            f'in combination {comb_lab}\n'
            f'No. of common steps: {stn_comb_data.n_steps}, '
            f'No. of extended steps: {stn_comb_data.n_steps_ext}, '
            f'No. of simulations: {self._n_sims}, '
            f'Total no. of simulated series: {n_ft_sims - 1}\n'
            f'Observed {pcent_abv_sim:0.1f}% above, '
            f'{pcent_within_sim:0.1f}% within and '
            f'{pcent_bel_sim:0.1f}% below simulation, '
            f'No. of frequencies: {n_ft_corrs}, '
            f'Left: All frequencies, '
            f'Right: Top 1 percent frequencies')

        plt.subplots_adjust(wspace=0.02)

        axs[0].tick_params(bottom=False)
        axs[2].tick_params(bottom=False, left=False, right=True)
        axs[3].tick_params(left=False)

        fig_name = (
            f'simult_ext_ft_{type_labs[1]}_coors_{comb_lab}_{ref_stn}.png')

        plt.savefig(
            str(self._out_dirs_dict['ft_corr_figs'] / fig_name),
            bbox_inches='tight')

        plt.close()
        return

    def _plot_binary_clusters(
            self,
            ref_stn,
            stn_idxs_swth_dict,
            freqs_tups,
            comb_lab):

        fig_size = (13, 10)

        cmap = plt.get_cmap('Blues')

        cmap_mappable = plt.cm.ScalarMappable(cmap=cmap)
        cmap_mappable.set_array([])

        for ep_i, ep in enumerate(self._eps):
            for tw_i, tw in enumerate(self._tws):
                neb_stns = stn_idxs_swth_dict[ref_stn]

                print(ep_i, ep, tw_i, tw)

                mean_probs = [
                    freqs_tups[(ref_stn, neb_stn)].avg_probs[ep_i, tw_i]
                    for neb_stn in neb_stns]

                min_probs = [
                    freqs_tups[(ref_stn, neb_stn)].min_probs[ep_i, tw_i]
                    for neb_stn in neb_stns]

                max_probs = [
                    freqs_tups[(ref_stn, neb_stn)].max_probs[ep_i, tw_i]
                    for neb_stn in neb_stns]

                fig = plt.figure(figsize=fig_size)

                map_ax = plt.subplot2grid((1, 25), (0, 0), 1, 24, fig=fig)
                cb_ax = plt.subplot2grid((1, 25), (0, 24), 1, 1, fig=fig)

                for neb_stn_i, neb_stn in enumerate(neb_stns):
                    patch = PolygonPatch(
                        self._cluster_feats_dict['patches'][neb_stn],
                        alpha=0.9,
                        fc=cmap(mean_probs[neb_stn_i]),
                        ec='#999999')

                    map_ax.add_patch(patch)

                    map_ax.plot(
                        self._cluster_feats_dict['xx'][neb_stn],
                        self._cluster_feats_dict['yy'][neb_stn],
                        alpha=0.5,
                        color='grey')

                    stn_text = (
                        f'({min_probs[neb_stn_i]:0.2f}, '
                        f'{mean_probs[neb_stn_i]:0.2f}, '
                        f'{max_probs[neb_stn_i]:0.2f})'
                        )

                    map_ax.text(
                        self._cluster_feats_dict['xx_mean'][neb_stn],
                        self._cluster_feats_dict['yy_mean'][neb_stn],
                        f'{neb_stn}\n{stn_text}',
                        ha='center',
                        va='center',
                        alpha=1.0,
                        color='k')

                map_ax.add_patch(
                    PolygonPatch(
                        self._cluster_feats_dict['patches'][ref_stn],
                        alpha=0.9,
                        fc='#999999',
                        ec='#999999'))

                map_ax.plot(
                    self._cluster_feats_dict['xx'][ref_stn],
                    self._cluster_feats_dict['yy'][ref_stn],
                    alpha=0.9,
                    color='black')

                map_ax.text(
                    self._cluster_feats_dict['xx_mean'][ref_stn],
                    self._cluster_feats_dict['yy_mean'][ref_stn],
                    f'{ref_stn}',
                    ha='center',
                    va='center',
                    alpha=1.0,
                    color='k')

                map_ax.grid()

                map_ax.set_aspect('equal', 'datalim')

                map_ax.tick_params(axis='x', labelrotation=90)

                map_ax.set_xlabel('x-coordinates')
                map_ax.set_ylabel('y-coordinates')

                cb = plt.colorbar(
                    cmap_mappable,
                    cax=cb_ax,
                    orientation='vertical',
                    fraction=0.1)

                cb.set_label('Mean simulated probability')

                map_ax.set_title(
                    f'Mean extremes binary simulated probability for '
                    f'station {ref_stn} in combination {comb_lab}\n'
                    f'Event exeecedance probability: {ep}'
                    f', time window: {tw} steps')

                fig_name = (
                    f'binary_clusters_{comb_lab}_{ref_stn}_EP{ep}_TW{tw}.png')

                plt.savefig(
                    str(self._out_dirs_dict['binary_cluster_figs'] / fig_name),
                    bbox_inches='tight')

                plt.close()
        return

    def _plot_nD_clusters(self, arg):

        stn_comb_str, comb_lab = arg

        stn_comb = eval(stn_comb_str)

        simult_ext_evts_cts, all_stn_combs = (
            self._get_cluster_simult_evts_data(stn_comb_str))

        fig_size = (13, 10)

        cmap = plt.get_cmap('Blues')

        cmap_mappable = plt.cm.ScalarMappable(cmap=cmap)
        cmap_mappable.set_array([])

        for ep_i, ep in enumerate(self._eps):
            for tw_i, tw in enumerate(self._tws):
                for ep_tw_stn_comb_i, ep_tw_stn_comb_str in enumerate(
                    all_stn_combs):

                    ep_tw_stn_comb = eval(ep_tw_stn_comb_str)

                    ep_tw_stn_comb_cts = simult_ext_evts_cts[
                        :, ep_i, tw_i, ep_tw_stn_comb_i]

                    all_probs = np.array([
                        ep_tw_stn_comb_ct[1]
                        for ep_tw_stn_comb_ct in ep_tw_stn_comb_cts[1:]])

                    obs_prob = ep_tw_stn_comb_cts[0][1]
                    min_prob = all_probs.min()
                    max_prob = all_probs.max()
                    mean_prob = all_probs.mean()

                    fig = plt.figure(figsize=fig_size)

                    map_ax = plt.subplot2grid((1, 25), (0, 0), 1, 24, fig=fig)
                    cb_ax = plt.subplot2grid((1, 25), (0, 24), 1, 1, fig=fig)

                    for stn in stn_comb:
                        if stn in ep_tw_stn_comb:
                            patch = PolygonPatch(
                                self._cluster_feats_dict['patches'][stn],
                                alpha=0.9,
                                fc=cmap(mean_prob),
                                ec='#999999')

                            stn_text_clr = 'black'

                        else:
                            patch = PolygonPatch(
                                self._cluster_feats_dict['patches'][stn],
                                alpha=0.2,
                                fc='#999999',
                                ec='#999999')

                            stn_text_clr = 'grey'

                        map_ax.add_patch(patch)

                        map_ax.plot(
                            self._cluster_feats_dict['xx'][stn],
                            self._cluster_feats_dict['yy'][stn],
                            alpha=0.5,
                            color='grey')

                        map_ax.text(
                            self._cluster_feats_dict['xx_mean'][stn],
                            self._cluster_feats_dict['yy_mean'][stn],
                            f'{stn}',
                            ha='center',
                            va='center',
                            alpha=1.0,
                            color=stn_text_clr)

                    map_ax.grid()

                    map_ax.set_aspect('equal', 'datalim')

                    map_ax.tick_params(axis='x', labelrotation=90)

                    map_ax.set_xlabel('x-coordinates')
                    map_ax.set_ylabel('y-coordinates')

                    cb = plt.colorbar(
                        cmap_mappable,
                        cax=cb_ax,
                        orientation='vertical',
                        fraction=0.1)

                    cb.set_label('Mean simulated probability')

                    stns_str = ', '.join(ep_tw_stn_comb)

                    map_ax.set_title(
                        f'Mean extremes nD simulated probability '
                        f'in combination {comb_lab}. '
                        f'Event exeecedance probability: {ep}'
                        f', time window: {tw} steps\n'
                        f'Obs. prob: {obs_prob:0.4f}, Simulated min: '
                        f'{min_prob:0.4f}, mean: {mean_prob:0.4f}, '
                        f'max: {max_prob:0.4f}\n'
                        f'Stations: {stns_str}')

                    fig_name = (
                        f'nD_clusters_{comb_lab}_EP{ep}_TW{tw}_'
                        f'{ep_tw_stn_comb_i}.png')

                    plt.savefig(
                        str(self._out_dirs_dict['nD_cluster_figs'] / fig_name),
                        bbox_inches='tight')

                    plt.close()
        return

    def _plot_sim_cdfs__corrs(self, ref_stn, stn_comb_grp, stn_comb_data):

        fig_size = (15, 8)

        comb_lab = stn_comb_data.comb_lab

        probs = (
            np.arange(1, 1 + stn_comb_data.n_steps, dtype=float) /
            (stn_comb_data.n_steps + 1))

        probs_ext = (
            np.arange(1, 1 + stn_comb_data.n_steps_ext, dtype=float) /
            (stn_comb_data.n_steps_ext + 1))

        if self._plot_sim_cdfs_flag:
            cdfs_arr = stn_comb_grp[f'sim_cdfs_{ref_stn}'][...]

            # mean, minima and maxima
            sort_stn_refr_ser = cdfs_arr[0, :stn_comb_data.n_steps]
            sort_avg_stn_sim_sers = cdfs_arr[1, :]
            sort_min_stn_sim_sers = cdfs_arr[2, :]
            sort_max_stn_sim_sers = cdfs_arr[3, :]

            plt.figure(figsize=fig_size)

            plt.plot(
                sort_stn_refr_ser,
                probs,
                color='r',
                alpha=0.7,
                label='obs',
                lw=1.5)

            plt.plot(
                sort_avg_stn_sim_sers,
                probs_ext,
                color='b',
                alpha=0.7,
                label='sim_mean',
                lw=1.5)

            plt.plot(
                sort_min_stn_sim_sers,
                probs_ext,
                color='C0',
                alpha=0.5,
                label='sim_min',
                lw=1)

            plt.plot(
                sort_max_stn_sim_sers,
                probs_ext,
                color='C1',
                alpha=0.5,
                label='sim_max',
                lw=1)

            plt.ylabel('Probability')
            plt.xlabel('Value')

            plt.legend()
            plt.grid()

            plt.title(
                f'CDFs for observed and simulated series of station '
                f'{ref_stn} in combination {comb_lab}\n'
                f'No. of common steps: {stn_comb_data.n_steps}, '
                f'No. of extended steps: {stn_comb_data.n_steps_ext}, '
                f'No. of simulations: {self._n_sims}')

            plt.tight_layout()

            fig_name = f'simult_ext_cdfs_{comb_lab}_{ref_stn}.png'

            plt.savefig(
                str(self._out_dirs_dict['cdfs_figs'] / fig_name),
                bbox_inches='tight')

            plt.close()

        if self._plot_auto_corrs_flag:
            acorrs_arr = stn_comb_grp[f'sim_acorrs_{ref_stn}'][...]

            # pearson corr
            stn_refr_pcorr = acorrs_arr[0, :]
            avg_stn_sim_pcorr = acorrs_arr[1, :]
            min_stn_sim_pcorr = acorrs_arr[2, :]
            max_stn_sim_pcorr = acorrs_arr[3, :]

            n_corrs = stn_refr_pcorr.shape[0]

            pcorr_within_bds_arr = np.ones_like(stn_refr_pcorr, dtype=int)
            pcorr_within_bds_arr[stn_refr_pcorr < min_stn_sim_pcorr] = 0
            pcorr_within_bds_arr[stn_refr_pcorr > max_stn_sim_pcorr] = 2

            pcent_abv_sim = 100 * (pcorr_within_bds_arr == 2).sum() / n_corrs
            pcent_within_sim = 100 * (pcorr_within_bds_arr == 1).sum() / n_corrs
            pcent_bel_sim = 100 * (pcorr_within_bds_arr == 0).sum() / n_corrs

            fig = plt.figure(figsize=fig_size)

            axs = (
                plt.subplot2grid((6, 1), (0, 0), 5, 1, fig=fig),
                plt.subplot2grid((6, 1), (5, 0), 1, 1, fig=fig))

            axs[0].plot(
                stn_refr_pcorr,
                color='r',
                alpha=0.7,
                label='obs',
                lw=1.5,
                zorder=3)

            axs[0].plot(
                avg_stn_sim_pcorr,
                color='b',
                alpha=0.7,
                label='sim_mean',
                lw=1.5,
                zorder=3)

            axs[0].fill_between(
                np.arange(min_stn_sim_pcorr.shape[0]),
                min_stn_sim_pcorr,
                max_stn_sim_pcorr,
                color='C0',
                alpha=0.3,
                label='sim_bds',
                lw=1,
                zorder=3)

            axs[0].set_xticklabels([])

            axs[0].set_ylabel('Pearson correlation')

            axs[0].legend()
            axs[0].grid()

            axs[1].plot(pcorr_within_bds_arr, color='C0', alpha=0.7, lw=1)
            axs[1].set_yticks(np.arange(3))
            axs[1].set_yticklabels(
                ['Obs. below sim.', 'Obs. within sim.', 'Obs. above sim.'])

            axs[1].grid()

            axs[1].set_xlabel('Lag (step)')

            axs[0].set_title(
                f'Pearson autocorrelations for observed and simulated '
                f'series of station {ref_stn} in combination {comb_lab}\n'
                f'No. of common steps: {stn_comb_data.n_steps}, '
                f'No. of extended steps: {stn_comb_data.n_steps_ext}, '
                f'No. of simulations: {self._n_sims}, '
                f'Observed {pcent_abv_sim:0.1f}% above, '
                f'{pcent_within_sim:0.1f}% within and '
                f'{pcent_bel_sim:0.1f}% below simulation')

            axs[0].tick_params(bottom=False)

            plt.tight_layout()

            fig_name = f'simult_ext_pcorrs_{comb_lab}_{ref_stn}.png'

            plt.savefig(
                str(self._out_dirs_dict['pcorr_figs'] / fig_name),
                bbox_inches='tight')

            plt.close()

            # spearman corr
            stn_refr_scorr = acorrs_arr[4, :]
            avg_stn_sim_scorr = acorrs_arr[5, :]
            min_stn_sim_scorr = acorrs_arr[6, :]
            max_stn_sim_scorr = acorrs_arr[7, :]

            scorr_within_bds_arr = np.ones_like(stn_refr_scorr, dtype=int)
            scorr_within_bds_arr[stn_refr_scorr < min_stn_sim_scorr] = 0
            scorr_within_bds_arr[stn_refr_scorr > max_stn_sim_scorr] = 2

            pcent_abv_sim = 100 * (scorr_within_bds_arr == 2).sum() / n_corrs
            pcent_within_sim = 100 * (scorr_within_bds_arr == 1).sum() / n_corrs
            pcent_bel_sim = 100 * (scorr_within_bds_arr == 0).sum() / n_corrs

            fig = plt.figure(figsize=fig_size)

            axs = (
                plt.subplot2grid((6, 1), (0, 0), 5, 1, fig=fig),
                plt.subplot2grid((6, 1), (5, 0), 1, 1, fig=fig))

            axs[0].fill_between(
                np.arange(min_stn_sim_scorr.shape[0]),
                min_stn_sim_scorr,
                max_stn_sim_scorr,
                color='C0',
                alpha=0.3,
                label='sim_bds',
                lw=1)

            axs[0].plot(
                avg_stn_sim_scorr,
                color='b',
                alpha=0.7,
                label='sim_mean',
                lw=1.5)

            axs[0].plot(
                stn_refr_scorr,
                color='r',
                alpha=0.7,
                label='obs',
                lw=1.5)

            axs[0].set_xticklabels([])

            axs[0].set_ylabel('Spearman correlation')

            axs[0].legend()
            axs[0].grid()

            axs[1].plot(scorr_within_bds_arr, color='C0', alpha=0.7, lw=1)
            axs[1].set_yticks(np.arange(3))
            axs[1].set_yticklabels(
                ['Obs. below sim.', 'Obs. within sim.', 'Obs. above sim.'])

            axs[1].grid()

            axs[1].set_xlabel('Lag (step)')

            axs[0].set_title(
                f'Spearman autocorrelations for observed and simulated '
                f'series of station {ref_stn} in combination {comb_lab}\n'
                f'No. of common steps: {stn_comb_data.n_steps}, '
                f'No. of extended steps: {stn_comb_data.n_steps_ext}, '
                f'No. of simulations: {self._n_sims}, '
                f'Observed {pcent_abv_sim:0.1f}% above, '
                f'{pcent_within_sim:0.1f}% within and '
                f'{pcent_bel_sim:0.1f}% below simulation')

            axs[0].tick_params(bottom=False)

            plt.tight_layout()

            fig_name = f'simult_ext_scorrs_{comb_lab}_{ref_stn}.png'

            plt.savefig(
                str(self._out_dirs_dict['scorr_figs'] / fig_name),
                bbox_inches='tight')

            plt.close()
        return

    def _plot_frequencies(self, ref_stn, stn_comb_data, stn_idxs_swth_dict):

        TableTup = namedtuple('TableTup', ['i', 'j', 'tbl', 'lab'])

        n_fig_rows = 2
        n_fig_cols = 4
        fig_size = (15, 6)

        comb_lab = stn_comb_data.comb_lab

        row_lab_strs = [
            f'{self._eps[i]} ({stn_comb_data.ref_evts_arr[i]}, '
            f'{stn_comb_data.ref_evts_ext_arr[i]})'
            for i in range(self._eps.shape[0])]

        col_hdr_clrs = [[0.75] * 4] * self._tws.shape[0]
        row_hdr_clrs = [[0.75] * 4] * self._eps.shape[0]

        max_tab_rows = self._eps.shape[0]

        for neb_stn in stn_idxs_swth_dict[ref_stn]:

            freqs_tups = stn_comb_data.freqs_tups[(ref_stn, neb_stn)]

            obs_vals = freqs_tups.obs_vals

            sim_avgs = freqs_tups.sim_avgs
            sim_maxs = freqs_tups.sim_maxs
            sim_mins = freqs_tups.sim_mins
            sim_stds = freqs_tups.sim_stds

            avg_probs = freqs_tups.avg_probs
            min_probs = freqs_tups.min_probs
            max_probs = freqs_tups.max_probs

            ax_arr = plt.subplots(
                nrows=n_fig_rows,
                ncols=n_fig_cols,
                sharex=True,
                sharey=True,
                figsize=fig_size)[1]

            tbls = [
                TableTup(0, 0, obs_vals, 'Observed frequency'),
                TableTup(0, 1, sim_avgs, 'Mean simulated frequency'),
                TableTup(0, 2, avg_probs, 'Mean simulated probability'),
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
                    n_tab_rows = self._eps.shape[0] + 1

                else:
                    tcol_labs = None
                    x_label = 'Time window'
                    col_colors = None

                    n_tab_rows = self._eps.shape[0]

                if not tbl.j:
                    trow_labs = row_lab_strs
                    row_colors = row_hdr_clrs

                else:
                    trow_labs = None
                    row_colors = None

                if tbl.j == (n_fig_cols - 1):
                    y_label = (
                        'Exceedance Probability\n(No. of common events,\n'
                        'No. of extended common events)')

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
                f'Reference station: {ref_stn}, Nebor station: {neb_stn}, '
                f'Combination: {comb_lab}\n'
                f'No. of common steps: {stn_comb_data.n_steps}, '
                f'No. of extended steps: {stn_comb_data.n_steps_ext}, '
                f'No. of simulations: {self._n_sims}',
                x=0.5,
                y=n_tab_rows / max_tab_rows,
                va='bottom')

            plt.tight_layout()

            sim_freq_fig_name = (
                f'simult_ext_stats_{comb_lab}_{ref_stn}_{neb_stn}.png')

            plt.savefig(
                str(self._out_dirs_dict['freq_figs'] / sim_freq_fig_name),
                bbox_inches='tight')
            plt.close()
        return
