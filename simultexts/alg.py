'''
Created on Feb 4, 2019

@author: Faizan-Uni
'''
from timeit import default_timer
from itertools import combinations

import h5py
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from pathos.multiprocessing import ProcessPool

from .misc import print_sl, print_el
from .data import SimultaneousExtremesDataAndSettings as SEDS


class SimultaneousExtremesAlgorithm(SEDS):

    def __init__(self, verbose=True, overwrite=True):

        SEDS.__init__(self, verbose=verbose)

        self._owr_flag = bool(overwrite)

        self._h5_hdl = None
        self._mp_pool = None

        self._max_acorr_steps = 500

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
                f'{len(self._stn_combs)} pairs...')

            print_el()

        sims_grp = self._h5_hdl.create_group('simultexts_sims')

        SEFC = SimultaneousExtremesFrequencyComputerMP(self)

        main_sim_beg_time = default_timer()

        SEFC_gen = (
            self._data_df.loc[:, stn_comb].dropna(axis=0, how='any')
            for stn_comb in self._stn_combs)

        if self._mp_pool is not None:
            stn_combs__arrs_dicts = list(
                self._mp_pool.uimap(SEFC.get_stn_comb_freqs, SEFC_gen))

            self._mp_pool.clear()

        else:
            stn_combs__arrs_dicts = map(SEFC.get_stn_comb_freqs, SEFC_gen)

        for stn_comb_arrs_dict in stn_combs__arrs_dicts:

            if stn_comb_arrs_dict is None:
                continue

            stn_comb, arrs_dict = stn_comb_arrs_dict

            stn_comb_grp = sims_grp.create_group(str(stn_comb))
            for key in arrs_dict:
                stn_comb_grp[key] = arrs_dict[key]

            self._h5_hdl.flush()

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

        self._h5_hdl['excd_probs'] = self._eps
        self._h5_hdl['time_windows'] = self._tws
        self._h5_hdl['n_sims'] = self._n_sims
        self._h5_hdl['n_stn_combs'] = len(self._stn_combs)

        self._h5_hdl['save_sim_cdfs_flag'] = int(self._save_sim_cdfs_flag)
        self._h5_hdl['save_sim_acorrs_flag'] = int(self._save_sim_acorrs_flag)
        self._h5_hdl['save_ft_cumm_corrs_flag'] = (
            int(self._save_sim_ft_cumm_corrs_flag))

        self._h5_hdl.flush()

        self._h5_hdl.close()
        self._h5_hdl = None

        if self._mp_pool is not None:
            self._mp_pool.join()
            self._mp_pool = None
        return

    def _prepare(self):

        self._stn_combs = tuple(combinations(self._data_df.columns, 2))

        n_stn_combs = len(self._stn_combs)

        assert all([len(stn_comb) == 2 for stn_comb in self._stn_combs]), (
            'Only configured for combinations of two series!')

        self._out_dir.mkdir(exist_ok=True)

        if self._owr_flag or (not self._h5_path.exists()):
            self._h5_hdl = h5py.File(self._h5_path, mode='w', driver='core')

        else:
            self._h5_hdl = h5py.File(self._h5_path, mode='r+', driver='core')

        if n_stn_combs < self._n_cpus:
            if self._vb:
                print_sl()

                print(
                    f'INFO: Reduced the number of running processes from '
                    f'{self._n_cpus} to {n_stn_combs}')

                print_el()

            self._n_cpus = n_stn_combs

        if (self._n_cpus > 1) and (self._mp_pool is None):
            self._mp_pool = ProcessPool(self._n_cpus)
        return

    __verify = verify


class SimultaneousExtremesFrequencyComputerMP:

    '''Not meant for use outside of this script'''

    def __init__(self, SEA_cls):

        take_sea_cls_var_labs = [
            '_vb',
            '_eps',
            '_tws',
            '_n_sims',
            '_n_cpus',
            '_save_sim_cdfs_flag',
            '_save_sim_acorrs_flag',
            '_save_sim_ft_cumm_corrs_flag',
            '_ext_steps',
            '_max_acorr_steps',
            ]

        for _var in take_sea_cls_var_labs:
            setattr(self, _var, getattr(SEA_cls, _var))

        self._vb_old = self._vb

        if self._n_cpus > 1:
            self._vb = False

        return

    def get_stn_comb_freqs(self, obs_vals_df):

        return self._get_stn_comb_freqs(obs_vals_df)

    def _get_stn_comb_freqs(self, obs_vals_df):

        # hard coded for 2D!

        assert isinstance(obs_vals_df, pd.DataFrame)

        sim_beg_time = default_timer()

        stn_comb = tuple(obs_vals_df.columns)

        n_combs = obs_vals_df.shape[1]

        assert len(stn_comb) == 2

        if obs_vals_df.shape[0] % 2:
            obs_vals_df = obs_vals_df.iloc[:-1]

        n_steps = obs_vals_df.shape[0]

        assert np.any(obs_vals_df.count().values)

        if self._ext_steps:
            n_steps_ext = int(n_steps * np.ceil(self._ext_steps / n_steps))

        else:
            n_steps_ext = n_steps

        _obs_vals_tile = np.tile(
            obs_vals_df.values, (n_steps_ext // n_steps, 1))

        obs_sort_df = pd.DataFrame(
            data=np.sort(_obs_vals_tile, axis=0), columns=stn_comb)

        if self._vb:
            print_sl()

            print(f'INFO: Going through the {stn_comb} combination...')

        if self._vb:
            print('Commmon steps between these stations:', n_steps)

            if self._ext_steps:
                print(
                    f'Number of steps to simulate after extension: '
                    f'{n_steps_ext}')

        assert not (n_steps_ext % n_steps), (
            'n_steps_ext not a multiple of n_steps!')

        if not obs_vals_df.shape[0] > ((2 * min(self._tws)) + 1):
            if self._vb:
                print(
                    'WARNING: No steps to iterate through, in '
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
            data=np.full((ft_steps, len(stn_comb)), np.nan, dtype=complex),
            columns=stn_comb)

        obs_probs_df = obs_vals_df.rank(ascending=True) / (n_steps + 1.0)

        for i in range(n_combs):
            obs_ft_df.iloc[:, i] = np.fft.rfft(obs_probs_df.iloc[:, i])

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
            data=np.full((ft_steps, len(stn_comb)), np.nan, dtype=complex),
            columns=stn_comb)

        sim_vals_df = pd.DataFrame(
            data=np.full((n_steps_ext, len(stn_comb)), np.nan, dtype=float),
            columns=stn_comb)

        arrs_dict = {
            **{f'neb_evts_{stn}':
                    np.full(
                        (self._n_sims + 1,
                         len(self._eps),
                         len(self._tws)),
                        np.nan,
                        dtype=int)
                for stn in stn_comb},

            'ref_evts':np.array(self._eps * n_steps, dtype=int),
            'ref_evts_ext':np.array(self._eps * n_steps_ext, dtype=int),
            'n_steps': n_steps,
            'n_steps_ext': n_steps_ext,
            }

        if self._save_sim_cdfs_flag or self._save_sim_acorrs_flag:
            stns_sims_dict = {
                f'sim_sers_{stn}':
                    np.full(
                        (self._n_sims + 1, n_steps_ext),
                        np.nan,
                        dtype=float)
                for stn in stn_comb}

        if self._save_sim_ft_cumm_corrs_flag:
            ft_sims_ctr = 0

            n_ft_sims = (self._n_sims * (n_steps_ext // n_steps)) + 1

            stns_ft_ccorrs_dict = {
                f'ft_ccorrs_{stn}':
                    np.full((n_ft_sims, ft_steps - 2), np.nan, dtype=float)
                for stn in stn_comb}

            arrs_dict.update(stns_ft_ccorrs_dict)

        _stn_idxs_swth = (1, 0)  # never change this

        for sim_no in range(self._n_sims + 1):
            for ext_step_idx in range(0, n_steps_ext, n_steps):
                # first sim is the observed data
                if not sim_no:
                    sim_vals_df.iloc[:n_steps, :] = obs_vals_df.values
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
                        stns_ft_ccorrs_dict[
                            f'ft_ccorrs_{stn}'][ft_sims_ctr, :] = (
                                self._get_ft_cumm_corrs(
                                    sim_vals_df[stn].iloc[
                        ext_step_idx:ext_step_idx + n_steps]))

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

            if self._save_sim_cdfs_flag or self._save_sim_acorrs_flag:
                for i in range(n_combs):
                    key = f'sim_sers_{stn_comb[i]}'

                    if not sim_no:
                        stns_sims_dict[key][sim_no, :n_steps] = (
                            obs_vals_df.iloc[:, i])

                    else:
                        _vals = obs_sort_df.iloc[:, i][
                            sim_ranks_df.iloc[:, i].values - 1]

                        assert np.all(np.isfinite(_vals))

                        stns_sims_dict[key][sim_no, :] = _vals

            for ref_stn_idx, ref_stn in enumerate(stn_comb):
                max_ep_ge_idxs = (
                    sim_vals_probs_df.iloc[:, ref_stn_idx] <= max_ep).values

                neb_stn = stn_comb[_stn_idxs_swth[ref_stn_idx]]

                max_ep_sim_df = sim_vals_probs_df.loc[max_ep_ge_idxs]

                freqs_arr = arrs_dict[f'neb_evts_{ref_stn}']

                for ep_idx, ep in enumerate(self._eps):
                    neb_evt_ctrs = {tw:0 for tw in self._tws}

                    for evt_idx_i, evt_idx in enumerate(max_ep_sim_df.index):
                        if max_ep_sim_df.iloc[evt_idx_i, ref_stn_idx] > ep:
                            continue

                        for tw in self._tws:
                            back_idx = max(0, evt_idx - tw)
                            forw_idx = evt_idx + tw

                            neb_evt_idxs = (max_ep_sim_df.loc[
                                back_idx:forw_idx, neb_stn] <= ep).values

                            neb_evt_sum = neb_evt_idxs.sum()
                            if not neb_evt_sum:
                                continue

                            neb_evt_ctrs[tw] += 1

                    for tw_idx, tw in enumerate(self._tws):
                        freqs_arr[sim_no, ep_idx, tw_idx] = neb_evt_ctrs[tw]

        if self._save_sim_cdfs_flag or self._save_sim_acorrs_flag:
            arrs_dict.update(self._get_stats_dict(
                stn_comb, stns_sims_dict, n_steps, n_steps_ext))

        if self._vb_old:
            if self._vb_old and not self._vb:
                print_sl()

            print(
                f'INFO: Done with the combination {stn_comb} in '
                f'{default_timer() - sim_beg_time:0.3f} seconds.')

            print_el()
        return stn_comb, arrs_dict

    def _get_ft_cumm_corrs(self, vals_ser):

        n_steps = vals_ser.shape[0]

        ft = np.fft.rfft(vals_ser)[1:n_steps // 2]

        mags = np.absolute(ft)

        cov_arr = mags ** 2

        tot_cov = cov_arr.sum()

        cumm_pcorr = np.cumsum(cov_arr) / tot_cov

        return cumm_pcorr

    def _get_stats_dict(
            self, stn_comb, stns_sims_dict, n_steps, n_steps_ext):

        assert any([self._save_sim_cdfs_flag, self._save_sim_acorrs_flag])

        stn_stats_dict = {}

        for stn in stn_comb:
            sims_key = f'sim_sers_{stn}'

            stn_sims = stns_sims_dict[sims_key]

            stn_refr_ser = stn_sims[0, :]
            stn_sim_sers = stn_sims[1:, :]

            if self._save_sim_cdfs_flag:
                sort_stn_refr_ser = np.sort(stn_refr_ser[:n_steps])
                sort_stn_sim_sers = np.sort(stn_sim_sers)
                sort_avg_stn_sim_sers = sort_stn_sim_sers.mean(axis=0)
                sort_min_stn_sim_sers = sort_stn_sim_sers.min(axis=0)
                sort_max_stn_sim_sers = sort_stn_sim_sers.max(axis=0)

                # mins, means and maxs sortred values (cdfs)
                cdfs_arr = np.full((4, n_steps_ext), np.nan, dtype=float)
                cdfs_arr[0, :n_steps] = sort_stn_refr_ser
                cdfs_arr[1, :] = sort_avg_stn_sim_sers
                cdfs_arr[2, :] = sort_min_stn_sim_sers
                cdfs_arr[3, :] = sort_max_stn_sim_sers

                stn_stats_dict[f'sim_cdfs_{stn}'] = cdfs_arr

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

                acorrs_arr = np.full((8, n_steps_ext), np.nan, dtype=float)

                # pearson
                acorrs_arr[0, :n_corr_steps] = auto_pcorrs[0, :]
                acorrs_arr[1, :n_corr_steps] = auto_pcorrs[1:, :].mean(axis=0)
                acorrs_arr[2, :n_corr_steps] = auto_pcorrs[1:, :].min(axis=0)
                acorrs_arr[3, :n_corr_steps] = auto_pcorrs[1:, :].max(axis=0)

                # spearman
                acorrs_arr[4, :n_corr_steps] = auto_scorrs[0, :]
                acorrs_arr[5, :n_corr_steps] = auto_scorrs[1:, :].mean(axis=0)
                acorrs_arr[6, :n_corr_steps] = auto_scorrs[1:, :].min(axis=0)
                acorrs_arr[7, :n_corr_steps] = auto_scorrs[1:, :].max(axis=0)

                stn_stats_dict[f'sim_acorrs_{stn}'] = acorrs_arr

            stns_sims_dict[sims_key] = None
            stn_refr_ser = stn_sim_sers = None

        return stn_stats_dict
