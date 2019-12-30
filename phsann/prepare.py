'''
Created on Dec 27, 2019

@author: Faizan
'''

import numpy as np
from scipy.stats import rankdata, norm

from ..simultexts.misc import print_sl, print_el
from ..cyth import get_asymms_sample

from .settings import PhaseAnnealingSettings as PAS


class PhaseAnnealingPrepare(PAS):

    def __init__(self, verbose=True):

        PAS.__init__(self, verbose)

        self._ref_rnk = None
        self._ref_nrm = None
        self._ref_ft = None
        self._ref_phs_spec = None
        self._ref_mag_spec = None
        self._ref_scorrs = None
        self._ref_asymms_1 = None
        self._ref_asymms_2 = None

        self._sim_rnk = None
        self._sim_nrm = None
        self._sim_ft = None
        self._sim_phs_spec = None
        self._sim_mag_spec = None
        self._sim_scorrs = None
        self._sim_asymms_1 = None
        self._sim_asymms_2 = None

        self._prep_ref_aux_flag = False
        self._prep_sim_aux_flag = False

        self._prep_prep_flag = False

        self._prep_verify_flag = False
        return

    def _get_ranks_probs(self, data):

        ranks = rankdata(data, method='average')

        probs = ranks / (self._data_ref_shape[0] + 1.0)

        assert np.all((0 < probs) & (probs < 1))

        return ranks, probs

    def _get_ranks_probs_norms(self, data):

        ranks, probs = self._get_ranks_probs(data)

        norms = norm.ppf(probs, loc=0.0, scale=1.0)

        assert np.all(np.isfinite(norms))

        return ranks, probs, norms

    def _get_scorrs_asymms(self, probs):

        scorrs = np.full(self._sett_obj_lag_steps, np.nan)
        asymms_1 = scorrs.copy()
        asymms_2 = scorrs.copy()

        for i in range(self._sett_obj_lag_steps):
            rolled_probs = np.roll(probs, i + 1)
            scorrs[i] = np.corrcoef(probs, rolled_probs)[0, 1]
            asymms_1[i], asymms_2[i] = get_asymms_sample(probs, rolled_probs)

        assert np.all(np.isfinite(scorrs))
        assert np.all(np.isfinite(asymms_1))
        assert np.all(np.isfinite(asymms_2))

        return scorrs, asymms_1, asymms_2

    def _gen_ref_aux_data(self):

        if self._data_ref_data.ndim != 1:
            raise NotImplementedError('Implementation for 1D only!')

        ranks, probs, norms = self._get_ranks_norms(self._data_ref_data)

        ft = np.fft.rfft(norms)

        phs_spec = np.angle(ft)
        mag_spec = np.abs(ft)

        assert np.all(np.isfinite(ft))
        assert np.all(np.isfinite(phs_spec))
        assert np.all(np.isfinite(mag_spec))

        self._ref_rnk = ranks
        self._ref_nrm = norms

        self._ref_ft = ft
        self._ref_phs_spec = phs_spec
        self._ref_mag_spec = mag_spec

        scorrs, asymms_1, asymms_2 = self._get_scorrs_asymms(probs)

        if self._sett_obj_rank_corr_flag:
            self._ref_scorrs = scorrs

        if self._sett_obj_symm_type_1_flag:
            self._ref_asymms_1 = asymms_1

        if self._sett_obj_symm_type_2_flag:
            self._ref_asymms_2 = asymms_2

        self._prep_ref_aux_flag = True
        return

    def _gen_sim_aux_data(self):

        assert self._prep_ref_aux_flag

        if self._data_ref_data.ndim != 1:
            raise NotImplementedError('Implementation for 1D only!')

        rands = np.random.random((self._data_ref_shape[0] // 2) - 1)

        phs_spec = -np.pi + (2 * np.pi * rands)

        ft = np.full(
            1 + (self._data_ref_shape // 2), np.nan, dtype=np.complex128)

        ft[+0] = self._ref_ft[+0]
        ft[-1] = self._ref_ft[-1]

        ft.real[1:-1] = np.cos(phs_spec) * self._ref_mag_spec[1:-1]
        ft.imag[1:-1] = np.sin(phs_spec) * self._ref_mag_spec[1:-1]

        assert np.all(np.isfinite(ft))
        assert np.all(np.isfinite(phs_spec))

        data = np.fft.irfft(ft)

        assert np.all(np.isfinite(data))

        ranks, probs, norms = self._get_ranks_probs_norms(data)

        self._sim_rnk = ranks
        self._sim_nrm = norms

        self._sim_ft = ft
        self._sim_phs_spec = phs_spec
        self._sim_mag_spec = self._ref_mag_spec.copy()

        scorrs, asymms_1, asymms_2 = self._get_scorrs_asymms(probs)

        if self._sett_obj_rank_corr_flag:
            self._sim_scorrs = scorrs

        if self._sett_obj_symm_type_1_flag:
            self._sim_asymms_1 = asymms_1

        if self._sett_obj_symm_type_2_flag:
            self._sim_asymms_2 = asymms_2

        self._prep_sim_aux_flag = True
        return

    def prepare(self):

        PAS.PhaseAnnealingSettings__verify(self)
        assert self._sett_verify_flag

        self._gen_ref_aux_data()
        assert self._prep_ref_aux_flag

        self._gen_sim_aux_data()
        assert self._prep_sim_aux_flag

        self._prep_prep_flag = True
        return

    def verify(self):

        assert self._prep_prep_flag

        if self._vb:
            print_sl()

            print(f'Phase annealing preparation done successfully!')

            print_el()

        self._prep_verify_flag = True
        return

    __verify = verify