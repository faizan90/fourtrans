'''
Created on Feb 4, 2019

@author: Faizan-Uni
'''
from timeit import default_timer
from itertools import combinations

import h5py
import numpy as np
import pandas as pd
from pathos.multiprocessing import ProcessPool

from .misc import print_sl, print_el
from .data import SimultaneousExtremesDataAndSettings as SEDS


class SimultaneousExtremesAlgorithm(SEDS):

    def __init__(self, verbose=True, overwrite=True):

        SEDS.__init__(self, verbose=verbose)
        self._owr_flag = bool(overwrite)

        self._h5_hdl = None
        self._mp_pool = None

        self._set_alg_verify_flag = False
        self._simult_exts_probs_done_flag = False
        return

    def _prepare(self):

        self._stn_combs = list(combinations(self._data_df.columns, 2))

        assert all([len(stn_comb) == 2 for stn_comb in self._stn_combs]), (
            'Only configured for combinations of two series!')

        self._out_dir.mkdir(exist_ok=True)

        self._h5_path = self._out_dir / 'simultexts_db.hdf5'

        if self._owr_flag or (not self._h5_path.exists()):
            self._h5_hdl = h5py.File(self._h5_path, mode='w', driver='core')

        else:
            self._h5_hdl = h5py.File(self._h5_path, mode='r+', driver='core')

        self._h5_hdl['return_periods'] = self._rps
        self._h5_hdl['time_windows'] = self._tws

        n_stn_combs = len(self._stn_combs)

        if n_stn_combs < self._n_cpus:
            if self._vb:
                print_sl()

                print(f'INFO: Reduced the number of running processes from '
                    f'{self._n_cpus} to {n_stn_combs}')

                print_el()

            self._n_cpus = n_stn_combs

        if (self._n_cpus > 1) and (self._mp_pool is None):
            self._mp_pool = ProcessPool(self._n_cpus)

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
                f'Computing simultaneous extremes frequency for '
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
                self._mp_pool.uimap(SEFC.get_freqs, SEFC_gen))

            self._mp_pool.clear()

        else:
            stn_combs__arrs_dicts = map(SEFC.get_freqs, SEFC_gen)

        for stn_comb_arrs_dict in stn_combs__arrs_dicts:

            if stn_comb_arrs_dict is None:
                continue

            stn_comb, arrs_dict = stn_comb_arrs_dict

            stn_comb_grp = sims_grp.create_group(str(stn_comb))
            for key in arrs_dict:
                stn_comb_grp[key] = arrs_dict[key]

            self._h5_hdl.flush()

        self._finalize_sims()

        self._simult_exts_probs_done_flag = True

        if self._vb:
            print_sl()
            print(
                f'Done computing simultaneous extremes frequency\n'
                f'Total simulation time was: '
                f'{default_timer() - main_sim_beg_time:0.3f} seconds')
            print_el()
        return

    def _finalize_sims(self):

        self._h5_hdl.close()
        self._h5_hdl = None
        return

    __verify = verify


class SimultaneousExtremesFrequencyComputerMP:

    '''Not meant for use outside of this script'''

    def __init__(self, SEA_cls):

        take_sea_cls_var_labs = [
            '_vb',
            '_rps',
            '_tws',
            '_n_sims',
            '_n_cpus',
            ]

        for _var in take_sea_cls_var_labs:
            setattr(self, _var, getattr(SEA_cls, _var))

        self._vb_old = self._vb

        if self._n_cpus > 1:
            self._vb = False
        return

    def get_freqs(self, obs_vals_df):
        sim_beg_time = default_timer()

        stn_comb = tuple(obs_vals_df.columns)

        if obs_vals_df.shape[0] % 2:
            obs_vals_df = obs_vals_df.iloc[:-1]

        assert len(stn_comb) == 2

        if self._vb:
            print_sl()

            print(f'INFO: Going through the {stn_comb} combination...')

        if self._vb:
            print(
                'Commmon steps between these stations:',
                obs_vals_df.shape[0])

        n_steps = obs_vals_df.shape[0]

        assert obs_vals_df.shape[0] > min(self._tws)

        if not n_steps:
            if self._vb:
                print(
                    'WARNING: no steps to iterate through, in '
                    'this combination!')
            return

        max_rp = self._rps.max()
        for rp in self._rps:
            if ((1 // rp) > n_steps) or (rp < max_rp):
                continue

            max_rp = rp

        if self._vb:
            print(
                f'INFO: Maximum return period for this '
                f'combination: {max_rp}')

        max_tw = min(self._tws)
        for tw in self._tws:
            if (tw > n_steps) or (tw < max_tw):
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

        for stn in stn_comb:
            obs_ft_df.loc[:, stn] = np.fft.rfft(obs_vals_df.loc[:, stn])

        obs_ft_mags_df = pd.DataFrame(
            data=np.abs(obs_ft_df.values), columns=stn_comb)

        obs_ft_phas_df = pd.DataFrame(
            data=np.angle(obs_ft_df.values), columns=stn_comb)

        assert np.all(np.isfinite(obs_ft_df))
        assert np.all(np.isfinite(obs_ft_mags_df))
        assert np.all(np.isfinite(obs_ft_phas_df))

        sim_phas_mult_vec = np.ones(ft_steps, dtype=np.int8)
        sim_phas_mult_vec[0] = 0
        sim_phas_mult_vec[ft_steps - 1] = 0

        sim_ft_df = pd.DataFrame(
            data=np.full((ft_steps, len(stn_comb)), np.nan, dtype=complex),
            columns=stn_comb)

        sim_vals_df = pd.DataFrame(
            data=np.full(
                (n_steps, len(stn_comb)), np.nan, dtype=float),
            columns=stn_comb)

        arrs_dict = {
            **{f'neb_evts_{stn}':
                np.full(
                    (self._n_sims + 1,
                     len(self._rps),
                     len(self._tws)),
                    np.nan,
                    dtype=int)
                for stn in stn_comb},

            'ref_evts':np.array(self._rps * n_steps, dtype=int),
            'n_steps': n_steps,
            'n_sims': self._n_sims,
            }

        for sim_no in range(self._n_sims + 1):
            # first sim is the observed data
            if not sim_no:
                sim_ft_phas_df = obs_ft_phas_df

            else:
                sim_phases = -np.pi + (
                    (2 * np.pi) * np.random.random(ft_steps))

                sim_phases *= sim_phas_mult_vec

                sim_ft_phas_df = obs_ft_phas_df.apply(
                    lambda x: x + sim_phases)

            sim_ft_phas_cos_df = np.cos(sim_ft_phas_df)
            sim_ft_phas_sin_df = np.sin(sim_ft_phas_df)

            for stn in stn_comb:
                reals = (
                    obs_ft_mags_df.loc[:, stn] *
                    sim_ft_phas_cos_df.loc[:, stn]).values

                imags = (
                    obs_ft_mags_df.loc[:, stn] *
                    sim_ft_phas_sin_df.loc[:, stn]).values

                sim_ft_df.loc[:, stn].values.real = reals
                sim_ft_df.loc[:, stn].values.imag = imags

            for stn in stn_comb:
                sim_vals_df.loc[:, stn] = np.fft.irfft(
                    sim_ft_df.loc[:, stn])

            # hard coded for 2D, make changes from here on for more dims

            sim_vals_probs_df = sim_vals_df.rank(
                ascending=False) / (n_steps + 1.0)

            for stn_1 in stn_comb:
                max_rp_ge_idxs = (
                    sim_vals_probs_df.loc[:, stn_1] <= max_rp).values

                max_rp_sim_df = sim_vals_probs_df.loc[max_rp_ge_idxs]

                for rp_idx, rp in enumerate(self._rps):
                    neb_evt_ctrs = {tw:0 for tw in self._tws}

                    for evt_idx in max_rp_sim_df.index:
                        if max_rp_sim_df.loc[evt_idx, stn_1] > rp:
                            continue

                        for stn_2 in stn_comb:
                            if stn_1 == stn_2:
                                continue

                            for tw in self._tws:
                                back_idx = max(0, evt_idx - tw)
                                forw_idx = evt_idx + tw

                                neb_evt_idxs = (
                                    max_rp_sim_df.loc[
                                    back_idx:forw_idx, stn_2] <= rp).values

                                neb_evt_sum = neb_evt_idxs.sum()
                                if not neb_evt_sum:
                                    continue

                                neb_evt_ctrs[tw] += 1

                    for tw_idx, tw in enumerate(self._tws):
                        arrs_dict[f'neb_evts_{stn_1}'][
                            sim_no, rp_idx, tw_idx] = neb_evt_ctrs[tw]

        if self._vb_old:
            if self._vb_old and not self._vb:
                print_sl()

            print(
                f'INFO: Done with the combination {stn_comb} in '
                f'{default_timer() - sim_beg_time:0.3f} seconds.')

            print_el()
        return stn_comb, arrs_dict
