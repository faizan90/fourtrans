'''
Created on Feb 4, 2019

@author: Faizan-Uni
'''

from timeit import default_timer
from multiprocessing import Manager

import h5py
import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm
from pathos.multiprocessing import ProcessPool

from .misc import print_sl, print_el, ret_mp_idxs
from .data import SimultaneousExtremesDataAndSettings as SEDS


class SimultaneousExtremesAlgorithm(SEDS):

    def __init__(self, verbose=True, overwrite=True):

        SEDS.__init__(self, verbose=verbose)

        self._owr_flag = bool(overwrite)

        self._mp_pool = None

        self._max_acorr_steps = 500

        self._n_cpus_extra = 0

        self._mp_lock = Manager().Lock()

        self._set_alg_verify_flag = False
        return

    def verify(self):

        SEDS._SimultaneousExtremesDataAndSettings__verify(self)

        self._set_alg_verify_flag = True
        return

    def cmpt_simult_exts_freqs(self):

        assert self._set_alg_verify_flag, 'Unverified algorithm state!'

        self._prepare()

        if self._vb:
            print_sl()

            print(
                f'Computing simultaneous extremes frequencies for '
                f'{len(self._stn_combs)} combination(s)...')

            print_el()

        SEFC = SimultaneousExtremesFrequencyComputerMP(self)

        main_sim_beg_time = default_timer()

        SEFC_gen = (
            self._data_df.loc[:, stn_comb].dropna(axis=0, how='any')
            for stn_comb in self._stn_combs)

        if self._mp_pool is not None:
            list(self._mp_pool.uimap(SEFC.get_stn_comb_freqs, SEFC_gen))

            self._mp_pool.clear()

        else:
            list(map(SEFC.get_stn_comb_freqs, SEFC_gen))

        self._finalize()

        if self._vb:
            print_sl()
            print(
                f'Done computing simultaneous extremes frequencies\n'
                f'Total simulation time was: '
                f'{default_timer() - main_sim_beg_time:0.3f} seconds')
            print_el()
        return

    def _finalize(self):

        h5_hdl = h5py.File(self._h5_path, mode='r+', driver='core')

        h5_hdl['excd_probs'] = self._eps
        h5_hdl['time_windows'] = self._tws
        h5_hdl['n_sims'] = self._n_sims
        h5_hdl['n_stn_combs'] = len(self._stn_combs)

        h5_hdl['save_sim_cdfs_flag'] = int(self._save_sim_cdfs_flag)
        h5_hdl['save_sim_acorrs_flag'] = int(self._save_sim_acorrs_flag)
        h5_hdl['save_ft_cumm_corrs_flag'] = (
            int(self._save_sim_ft_cumm_corrs_flag))

        h5_hdl.flush()
        h5_hdl.close()

        if self._mp_pool is not None:
            self._mp_pool.join()
            self._mp_pool = None
        return

    def _prepare(self):

#         self._stn_combs = tuple(combinations(self._data_df.columns, 2))

        self._stn_combs = (self._data_df.columns,)

        n_stn_combs = len(self._stn_combs)

        assert all([len(stn_comb) >= 2 for stn_comb in self._stn_combs]), (
            'Only configured for combinations of two series!')

        self._out_dir.mkdir(exist_ok=True)

        if self._owr_flag or (not self._h5_path.exists()):
            h5_hdl = h5py.File(self._h5_path, mode='w', driver='core')

        else:
            h5_hdl = h5py.File(self._h5_path, mode='r+', driver='core')

        if 'simultexts_sims' not in h5_hdl:
            h5_hdl.create_group('simultexts_sims')

            h5_hdl.flush()

        h5_hdl.close()

        if n_stn_combs < self._n_cpus:
            self._n_cpus_extra = self._n_cpus - n_stn_combs
            self._n_cpus = n_stn_combs

        if (self._n_cpus > 1) and (self._mp_pool is None):
            self._mp_pool = ProcessPool(self._n_cpus)
        return

    __verify = verify


class SimultaneousExtremesFrequencyComputerMP:

    '''Meant for use by SimultaneousExtremesAlgorithm only'''

    def __init__(self, SEA_cls):

        take_sea_cls_var_labs = [
            '_vb',
            '_eps',
            '_tws',
            '_n_sims',
            '_n_cpus',
            '_n_cpus_extra',
            '_save_sim_cdfs_flag',
            '_save_sim_acorrs_flag',
            '_save_sim_ft_cumm_corrs_flag',
            '_ext_steps',
            '_max_acorr_steps',
            '_h5_path',
            '_mp_lock',
            ]

        for _var in take_sea_cls_var_labs:
            setattr(self, _var, getattr(SEA_cls, _var))

        self._vb_old = self._vb

        if self._n_cpus > 1:
            self._vb = False
        return

    def get_stn_comb_freqs(self, obs_vals_df):

        sim_beg_time = default_timer()

        n_extra_cpus_per_comb = self._n_cpus_extra // self._n_cpus

        if n_extra_cpus_per_comb:
            sim_chunks_idxs = ret_mp_idxs(self._n_sims, n_extra_cpus_per_comb)

            sub_mp_pool = ProcessPool(n_extra_cpus_per_comb)

            self._vb = False

        else:
            sim_chunks_idxs = np.array([0, self._n_sims])

            sub_mp_pool = None

        sim_chunks_idxs[-1] += 1  # for simulation zero

        sim_chunk_gen = ((
            obs_vals_df,
            (sim_chunks_idxs[i], sim_chunks_idxs[i + 1]),
            self._mp_lock,)

            for i in range(sim_chunks_idxs.shape[0] - 1))

        if sub_mp_pool is not None:
            list(sub_mp_pool.uimap(self._get_stn_comb_freqs, sim_chunk_gen))
            sub_mp_pool.clear()

        else:
            list(map(self._get_stn_comb_freqs, sim_chunk_gen))

        if self._save_sim_cdfs_flag or self._save_sim_acorrs_flag:
            with self._mp_lock:
                self._write_stats_to_hdf5(tuple(obs_vals_df.columns))

        if self._vb_old:
            if self._vb_old and not self._vb:
                print_sl()

            print(
                f'INFO: Finished in '
                f'{default_timer() - sim_beg_time:0.3f} seconds.')

            print_el()
        return

    def _get_stn_comb_freqs(self, args):

        obs_vals_df, sim_chunk_idxs, mp_lock = args

        n_sims_chunk = sim_chunk_idxs[1] - sim_chunk_idxs[0]

        assert isinstance(obs_vals_df, pd.DataFrame)

        stn_comb = tuple(obs_vals_df.columns)

        n_combs = obs_vals_df.shape[1]

        assert n_combs >= 2

        if obs_vals_df.shape[0] % 2:
            obs_vals_df = obs_vals_df.iloc[:-1]

        n_steps = obs_vals_df.shape[0]

        assert np.all(obs_vals_df.count().values > 0)

        if self._ext_steps:
            n_steps_ext = int(n_steps * np.ceil(self._ext_steps / n_steps))

        else:
            n_steps_ext = n_steps

        _obs_vals_tile = np.tile(
            obs_vals_df.values, (n_steps_ext // n_steps, 1))

        obs_sort_df = pd.DataFrame(
            data=np.sort(_obs_vals_tile, axis=0), columns=stn_comb)

        del _obs_vals_tile

        if self._vb:
            print_sl()

            print(f'INFO: {n_combs} stations in this combination')

            if len(stn_comb) < 10:
                print(f'INFO: Combination {stn_comb}')

        if self._vb:
            print('Commmon steps among these stations:', n_steps)

            if self._ext_steps:
                print(
                    f'Number of steps to simulate after extension: '
                    f'{n_steps_ext}')

        assert not (n_steps_ext % n_steps), (
            'n_steps_ext not a multiple of n_steps!')

        if not obs_vals_df.shape[0] > ((2 * min(self._tws)) + 1):
            if self._vb:
                print(
                    'WARNING: Not enough steps to iterate through, in '
                    'this combination!')
            return

        max_ep = self._eps.max()
        for ep in self._eps:
            if ((1 // ep) > n_steps_ext) or (ep < max_ep):
                continue

            max_ep = ep

        if self._vb:
            print(
                f'INFO: Maximum exceedance probability for this '
                f'combination: {max_ep}')

        max_tw = (2 * min(self._tws)) + 1
        for tw in self._tws:
            if (((2 * tw) + 1) > n_steps_ext) or (((2 * tw) + 1) < max_tw):
                continue

            max_tw = tw

        if self._vb:
            print(
                f'INFO: Maximum time window size for this '
                f'combination: {max_tw}')

        ft_steps = 1 + (n_steps // 2)

        obs_ft_df = pd.DataFrame(
            data=np.full((ft_steps, n_combs), np.nan, dtype=complex),
            columns=stn_comb)

        tfm = 'prob'

        if tfm == 'obs':
            ft_input_df = obs_vals_df.copy()

        elif tfm == 'prob':
            ft_input_df = obs_vals_df.rank(ascending=True) / (n_steps + 1.0)

        elif tfm == 'norm':
            ft_input_df = pd.DataFrame(
                data=norm.ppf(
                    obs_vals_df.rank(ascending=True) / (n_steps + 1.0)),
                columns=stn_comb)

        else:
            raise ValueError(tfm)

        for i in range(n_combs):
            obs_ft_df.iloc[:, i] = np.fft.rfft(ft_input_df.iloc[:, i])

        obs_ft_mags_df = pd.DataFrame(
            data=np.abs(obs_ft_df.values), columns=stn_comb)

        obs_ft_phas_df = pd.DataFrame(
            data=np.angle(obs_ft_df.values), columns=stn_comb)

        assert np.all(np.isfinite(obs_ft_df))
        assert np.all(np.isfinite(obs_ft_mags_df))
        assert np.all(np.isfinite(obs_ft_phas_df))

        sim_phas_mult_vec = np.ones(ft_steps, dtype=int)
        sim_phas_mult_vec[0] = 0
        sim_phas_mult_vec[ft_steps - 1] = 0

        sim_ft_df = pd.DataFrame(
            data=np.full((ft_steps, n_combs), np.nan, dtype=complex),
            columns=stn_comb)

        sim_vals_df = pd.DataFrame(
            data=np.full((n_steps_ext, n_combs), np.nan, dtype=float),
            columns=stn_comb)

        arrs_dict = {
            **{f'neb_evts_{stn}':
                    np.full(
                        (n_sims_chunk,
                         n_combs - 1,
                         len(self._eps),
                         len(self._tws)),
                        np.nan,
                        dtype=int)
                for stn in stn_comb},

            'ref_evts': np.array(self._eps * n_steps, dtype=int),
            'ref_evts_ext': np.array(self._eps * n_steps_ext, dtype=int),
            'n_steps': n_steps,
            'n_steps_ext': n_steps_ext,
            }

        if self._save_sim_cdfs_flag or self._save_sim_acorrs_flag:
            stns_sims_dict = {
                f'sim_sers_{stn}':
                    np.full(
                        (n_sims_chunk, n_steps_ext),
                        np.nan,
                        dtype=float)
                for stn in stn_comb}

            arrs_dict.update(stns_sims_dict)

        if self._save_sim_ft_cumm_corrs_flag:
            ft_sims_ctr = 0

            n_ft_sims = (n_sims_chunk * (n_steps_ext // n_steps))

            stns_ft_ccorrs_dict = {
                f'ft_ccorrs_{stn}':
                    np.full((n_ft_sims, ft_steps - 2), np.nan, dtype=float)
                for stn in stn_comb}

            arrs_dict.update(stns_ft_ccorrs_dict)

        stn_idxs_swth_dict = {}
        for i, stn in enumerate(stn_comb):
            bools = np.ones(n_combs, dtype=bool)
            bools[i] = False
            stn_idxs_swth_dict[stn] = np.array(stn_comb)[bools]

        for sim_no_idx, sim_no in enumerate(
            range(sim_chunk_idxs[0], sim_chunk_idxs[1])):

            for ext_step_idx in range(0, n_steps_ext, n_steps):
                # first sim is the observed data
                if not sim_no:
                    sim_vals_df.iloc[:n_steps, :] = ft_input_df.values
                    sim_vals_df.iloc[n_steps:, :] = np.nan

                else:
                    sim_phases = -np.pi + (
                        (2 * np.pi) * np.random.random(ft_steps))

                    sim_phases *= sim_phas_mult_vec

                    sim_ft_phas_df = obs_ft_phas_df.apply(
                        lambda x: x + sim_phases)

                    sim_ft_phas_cos_df = np.cos(sim_ft_phas_df)
                    sim_ft_phas_sin_df = np.sin(sim_ft_phas_df)

                    for i in range(n_combs):
                        reals = (
                            obs_ft_mags_df.iloc[:, i] *
                            sim_ft_phas_cos_df.iloc[:, i]).values

                        imags = (
                            obs_ft_mags_df.iloc[:, i] *
                            sim_ft_phas_sin_df.iloc[:, i]).values

                        sim_ft_df.iloc[:, i].values.real = reals
                        sim_ft_df.iloc[:, i].values.imag = imags

                    for i in range(n_combs):
                        sim_vals_df.iloc[
                            ext_step_idx:ext_step_idx + n_steps, i] = (
                                np.fft.irfft(sim_ft_df.iloc[:, i]))

                if self._save_sim_ft_cumm_corrs_flag:
                    for stn in stn_comb:
                        if not sim_no:
                            _vals = self._get_ft_cumm_corrs(obs_vals_df[stn])

                        else:
                            _ranks = sim_vals_df[stn].iloc[
                                ext_step_idx:ext_step_idx + n_steps].rank(
                                    ).astype(int).values

                            _sort_obs_ser = obs_vals_df[stn].sort_values()

                            _vals = self._get_ft_cumm_corrs(
                                _sort_obs_ser.iloc[_ranks - 1])

                        stns_ft_ccorrs_dict[
                            f'ft_ccorrs_{stn}'][ft_sims_ctr, :] = _vals

                    assert np.all(np.isfinite(stns_ft_ccorrs_dict[
                        f'ft_ccorrs_{stn}'][ft_sims_ctr, :]))

                    ft_sims_ctr += 1

                if not sim_no:
                    break

            sim_ranks_df = sim_vals_df.rank(ascending=True, method='max')

            if not sim_no:
                sim_vals_probs_df = (
                    n_steps - sim_ranks_df + 1) / (n_steps + 1.0)

            else:
                sim_vals_probs_df = (
                    n_steps_ext - sim_ranks_df + 1) / (n_steps_ext + 1.0)

            assert np.all(sim_vals_probs_df.min() > 0)
            assert np.all(sim_vals_probs_df.max() < 1)

            if self._save_sim_cdfs_flag or self._save_sim_acorrs_flag:
                for i in range(n_combs):
                    key = f'sim_sers_{stn_comb[i]}'

                    if not sim_no:
                        stns_sims_dict[key][sim_no_idx, :n_steps] = (
                            obs_vals_df.iloc[:, i])

                    else:
                        _vals = obs_sort_df.iloc[:, i][
                            sim_ranks_df.iloc[:, i].values - 1]

                        assert np.all(np.isfinite(_vals))

                        stns_sims_dict[key][sim_no_idx, :] = _vals

            self._fill_freqs_arr(
                stn_comb,
                sim_vals_probs_df,
                max_ep,
                arrs_dict,
                stn_idxs_swth_dict,
                sim_no_idx)

        with mp_lock:
            self._write_freqs_data_to_hdf5(stn_comb, arrs_dict, sim_chunk_idxs)
        return

    def _fill_freqs_arr(
            self,
            stn_comb,
            sim_vals_probs_df,
            max_ep,
            arrs_dict,
            stn_idxs_swth_dict,
            sim_no_idx):

        for ref_stn_idx, ref_stn in enumerate(stn_comb):
            max_ep_ge_idxs = (
                sim_vals_probs_df.iloc[:, ref_stn_idx] <= max_ep).values

            max_ep_sim_df = sim_vals_probs_df.loc[max_ep_ge_idxs]

            freqs_arr = arrs_dict[f'neb_evts_{ref_stn}']

            for neb_stn_i, neb_stn in enumerate(
                stn_idxs_swth_dict[ref_stn]):

                for ep_idx, ep in enumerate(self._eps):
                    neb_evt_ctrs = {tw:0 for tw in self._tws}

                    for evt_idx_i, evt_idx in enumerate(
                        max_ep_sim_df.index):

                        if max_ep_sim_df.iloc[evt_idx_i, ref_stn_idx] > ep:
                            continue

                        for tw in self._tws:
                            back_idx = max(0, evt_idx - tw)  # can take the end values as well?
                            forw_idx = evt_idx + tw

                            neb_evt_idxs = (max_ep_sim_df.loc[
                                back_idx:forw_idx, neb_stn] <= ep).values

                            neb_evt_sum = neb_evt_idxs.sum()
                            if not neb_evt_sum:
                                continue

                            neb_evt_ctrs[tw] += 1

                    for tw_idx, tw in enumerate(self._tws):
                        freqs_arr[sim_no_idx, neb_stn_i, ep_idx, tw_idx] = (
                            neb_evt_ctrs[tw])
        return

    def _write_freqs_data_to_hdf5(self, stn_comb, arrs_dict, sim_chunk_idxs):

        '''Must be called with a Lock'''

        h5_hdl = h5py.File(self._h5_path, mode='r+', driver='core')

        sims_grp = h5_hdl['simultexts_sims']

        stn_comb_str = str(stn_comb)

        if stn_comb_str  not in sims_grp:
            stn_grp = sims_grp.create_group(stn_comb_str)

        else:
            stn_grp = sims_grp[stn_comb_str]

        for stn in stn_comb:
            freq_key = f'neb_evts_{stn}'

            if freq_key not in stn_grp:
                sim_freq_ds = stn_grp.create_dataset(
                    freq_key,
                    (self._n_sims + 1,
                     len(stn_comb) - 1,
                     len(self._eps),
                     len(self._tws)),
                    dtype=int)

            else:
                sim_freq_ds = stn_grp[freq_key]

            sim_freq_ds[
                sim_chunk_idxs[0]:sim_chunk_idxs[1]] = arrs_dict[freq_key]

            del arrs_dict[freq_key]

            for key in ['ref_evts', 'ref_evts_ext', 'n_steps', 'n_steps_ext']:
                if key in stn_grp:
                    continue

                stn_grp[key] = arrs_dict[key]

            n_steps = stn_grp['n_steps'][...]
            n_steps_ext = stn_grp['n_steps_ext'][...]

            if self._save_sim_ft_cumm_corrs_flag:
                corrs_key = f'ft_ccorrs_{stn}'

                idxs_scale = n_steps_ext // n_steps

                if corrs_key not in stn_grp:
                    n_ft_sims = (self._n_sims * idxs_scale) + 1

                    ft_steps = 1 + (n_steps // 2)

                    sim_corrs_ds = stn_grp.create_dataset(
                        corrs_key,
                        (n_ft_sims, ft_steps - 2),
                        dtype=float)

                else:
                    sim_corrs_ds = stn_grp[corrs_key]

                sim_corrs_ds[
                    sim_chunk_idxs[0] * idxs_scale:
                    sim_chunk_idxs[1] * idxs_scale] = arrs_dict[corrs_key]

                del arrs_dict[corrs_key]

            if self._save_sim_cdfs_flag or self._save_sim_acorrs_flag:
                sims_key = f'sim_sers_{stn}'

                if sims_key not in stn_grp:
                    sims_ds = stn_grp.create_dataset(
                        sims_key,
                        (self._n_sims + 1, n_steps_ext),
                        dtype=float)

                else:
                    sims_ds = stn_grp[sims_key]

                sims_ds[
                    sim_chunk_idxs[0]:sim_chunk_idxs[1]] = arrs_dict[sims_key]

                del arrs_dict[sims_key]

        h5_hdl.flush()
        h5_hdl.close()
        return

    def _get_ft_cumm_corrs(self, vals_ser):

        n_steps = vals_ser.shape[0]

        ft = np.fft.rfft(vals_ser)[1:n_steps // 2]

        mags = np.absolute(ft)

        cov_arr = mags ** 2

        tot_cov = cov_arr.sum()

        cumm_pcorr = np.cumsum(cov_arr) / tot_cov

        return cumm_pcorr

    def _write_stats_to_hdf5(self, stn_comb):

        '''Must be called with a Lock'''

        assert any([self._save_sim_cdfs_flag, self._save_sim_acorrs_flag])

        stn_comb_str = str(stn_comb)

        h5_hdl = h5py.File(self._h5_path, mode='r+', driver='core')

        sims_grp = h5_hdl['simultexts_sims']

        stn_grp = sims_grp[stn_comb_str]

        n_steps = stn_grp['n_steps'][...]

        n_steps_ext = stn_grp['n_steps_ext'][...]

        for stn in stn_comb:
            sims_key = f'sim_sers_{stn}'

            stn_sims = stn_grp[sims_key][...]

            stn_refr_ser = stn_sims[0, :]
            stn_sim_sers = stn_sims[1:, :]

            if self._save_sim_cdfs_flag:
                sort_stn_refr_ser = np.sort(stn_refr_ser[:n_steps])
                sort_stn_sim_sers = np.sort(stn_sim_sers, axis=1)
                sort_avg_stn_sim_sers = sort_stn_sim_sers.mean(axis=0)
                sort_min_stn_sim_sers = sort_stn_sim_sers.min(axis=0)
                sort_max_stn_sim_sers = sort_stn_sim_sers.max(axis=0)

                # mins, means and maxs sorted values (cdfs)
                cdfs_arr = np.full((4, n_steps_ext), np.nan, dtype=float)
                cdfs_arr[0, :n_steps] = sort_stn_refr_ser
                cdfs_arr[1, :] = sort_avg_stn_sim_sers
                cdfs_arr[2, :] = sort_min_stn_sim_sers
                cdfs_arr[3, :] = sort_max_stn_sim_sers

                stn_grp[f'sim_cdfs_{stn}'] = cdfs_arr

                sort_stn_sim_sers = None

            if self._save_sim_acorrs_flag:
                n_corr_steps = min(self._max_acorr_steps, n_steps)

                auto_pcorrs = np.full(
                    (self._n_sims + 1, n_corr_steps), np.nan)

                auto_scorrs = auto_pcorrs.copy()

                for i in range(self._n_sims + 1):
                    sim_ser = stn_sims[i, :]

                    if not i:
                        sim_ser = sim_ser[:n_steps]

                    rank_sim = rankdata(sim_ser)

                    for j in range(n_corr_steps):
                        auto_pcorrs[i, j] = np.corrcoef(
                            sim_ser, np.roll(sim_ser, j))[0, 1]

                        auto_scorrs[i, j] = np.corrcoef(
                            rank_sim, np.roll(rank_sim, j))[0, 1]

                acorrs_arr = np.full((8, n_corr_steps), np.nan, dtype=float)

                # pearson
                acorrs_arr[0] = auto_pcorrs[0, :]
                acorrs_arr[1] = auto_pcorrs[1:, :].mean(axis=0)
                acorrs_arr[2] = auto_pcorrs[1:, :].min(axis=0)
                acorrs_arr[3] = auto_pcorrs[1:, :].max(axis=0)

                # spearman
                acorrs_arr[4] = auto_scorrs[0, :]
                acorrs_arr[5] = auto_scorrs[1:, :].mean(axis=0)
                acorrs_arr[6] = auto_scorrs[1:, :].min(axis=0)
                acorrs_arr[7] = auto_scorrs[1:, :].max(axis=0)

                stn_grp[f'sim_acorrs_{stn}'] = acorrs_arr

            # del stn_grp[sims_key]  # comment this to keep the simulations
            stn_refr_ser = stn_sim_sers = stn_sims = None

        h5_hdl.flush()
        h5_hdl.close()
        return
