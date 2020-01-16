'''
Created on Dec 27, 2019

@author: Faizan
'''

import numpy as np
from scipy.stats import rankdata, norm

from ..simultexts.misc import print_sl, print_el
from ..cyth import (
    get_asymms_sample,
    get_asymm_1_sample,
    get_asymm_2_sample,
    fill_bi_var_cop_dens,
    )

from .settings import PhaseAnnealingSettings as PAS


class PhaseAnnealingPrepare(PAS):

    '''Prepare derived variables required by phase annealing here'''

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
        self._ref_ecop_dens_arrs = None

        # Be careful with these
        self._sim_rnk = None
        self._sim_nrm = None
        self._sim_ft = None
        self._sim_phs_spec = None
        self._sim_mag_spec = None
        self._sim_scorrs = None
        self._sim_asymms_1 = None
        self._sim_asymms_2 = None
        self._sim_ecop_dens_arrs = None

        self._prep_ref_aux_flag = False
        self._prep_sim_aux_flag = False

        self._prep_prep_flag = False

        self._prep_verify_flag = False
        return

    def _get_ranks_probs(self, data):

        ranks = rankdata(data, method='average')

        probs = ranks / (self._data_ref_shape[0] + 1.0)

        assert np.all((0 < probs) & (probs < 1)), 'probs out of range!'

        return ranks, probs

    def _get_ranks_probs_norms(self, data):

        ranks, probs = self._get_ranks_probs(data)

        norms = norm.ppf(probs, loc=0.0, scale=1.0)

        assert np.all(np.isfinite(norms)), 'Invalid values in norms!'

        return ranks, probs, norms

    def _get_asymm_1_max(self, scorr):

        if self._sett_obj_asymms_normalize_flag:
            a_max = (
                0.5 * (1 - scorr)) * (1 - ((0.5 * (1 - scorr)) ** (1.0 / 3.0)))

        else:
            a_max = 1.0

        return a_max

    def _get_asymm_2_max(self, scorr):

        if self._sett_obj_asymms_normalize_flag:
            a_max = (
                0.5 * (1 + scorr)) * (1 - ((0.5 * (1 + scorr)) ** (1.0 / 3.0)))

        else:
            a_max = 1.0

        return a_max

    def _get_obj_vars(self, probs):

        if (self._sett_obj_scorr_flag or
            self._sett_obj_asymm_type_1_flag or
            self._sett_obj_asymm_type_1_flag):

            scorrs = np.full(self._sett_obj_lag_steps.size, np.nan)

        else:
            scorrs = None

        if self._sett_obj_asymm_type_1_flag:
            asymms_1 = np.full(self._sett_obj_lag_steps.size, np.nan)

        else:
            asymms_1 = None

        if self._sett_obj_asymm_type_2_flag:
            asymms_2 = np.full(self._sett_obj_lag_steps.size, np.nan)

        else:
            asymms_2 = None

        if self._sett_obj_ecop_dens_flag:
            ecop_dens_arrs = np.full(
                (self._sett_obj_lag_steps.size,
                 self._sett_obj_ecop_dens_bins,
                 self._sett_obj_ecop_dens_bins),
                 np.nan,
                 dtype=np.float64)

        else:
            ecop_dens_arrs = None

        if (self._sett_obj_asymm_type_1_flag and
            self._sett_obj_asymm_type_2_flag):

            double_flag = True

        else:
            double_flag = False

        for i, lag in enumerate(self._sett_obj_lag_steps):
            rolled_probs = np.roll(probs, lag)

            if scorrs is not None:
                scorrs[i] = np.corrcoef(probs, rolled_probs)[0, 1]

            if double_flag:
                asymms_1[i], asymms_2[i] = get_asymms_sample(
                    probs, rolled_probs)

            else:
                if asymms_1 is not None:
                    asymms_1[i] = get_asymm_1_sample(probs, rolled_probs)

                if asymms_2 is not None:
                    asymms_2[i] = get_asymm_2_sample(probs, rolled_probs)

            if asymms_1 is not None:
                asymms_1[i] = asymms_1[i] / self._get_asymm_1_max(scorrs[i])

            if asymms_2 is not None:
                asymms_2[i] = asymms_2[i] / self._get_asymm_2_max(scorrs[i])

            if ecop_dens_arrs is not None:
                fill_bi_var_cop_dens(
                    probs, rolled_probs, ecop_dens_arrs[i, :, :])

        if scorrs is not None:
            assert np.all(np.isfinite(scorrs)), 'Invalid values in scorrs!'

            assert np.all((scorrs >= -1.0) & (scorrs <= +1.0)), (
                'scorrs out of range!')

        if not self._sett_obj_scorr_flag:
            scorrs = None

        if asymms_1 is not None:
            assert np.all(np.isfinite(asymms_1)), 'Invalid values in asymms_1!'

            assert np.all((asymms_1 >= -1.0) & (asymms_1 <= +1.0)), (
                'asymms_1 out of range!')

        if asymms_2 is not None:
            assert np.all(np.isfinite(asymms_2)), 'Invalid values in asymms_2!'

            assert np.all((asymms_2 >= -1.0) & (asymms_2 <= +1.0)), (
                'asymms_2 out of range!')

        if ecop_dens_arrs is not None:
            assert np.all(np.isfinite(ecop_dens_arrs)), (
                'Invalid values in ecop_dens_arrs!')

        return scorrs, asymms_1, asymms_2, ecop_dens_arrs

    def _gen_ref_aux_data(self):

        if self._data_ref_rltzn.ndim != 1:
            raise NotImplementedError('Implementation for 1D only!')

        ranks, probs, norms = self._get_ranks_probs_norms(self._data_ref_rltzn)

        ft = np.fft.rfft(norms)

        phs_spec = np.angle(ft)
        mag_spec = np.abs(ft)

        assert np.all(np.isfinite(ft)), 'Invalid values in ft!'
        assert np.all(np.isfinite(phs_spec)), 'Invalid values in phs_spec!'
        assert np.all(np.isfinite(mag_spec)), 'Invalid values in mag_spec!'

        self._ref_rnk = ranks
        self._ref_nrm = norms

        self._ref_ft = ft
        self._ref_phs_spec = phs_spec
        self._ref_mag_spec = mag_spec

        scorrs, asymms_1, asymms_2, ecop_dens_arrs = self._get_obj_vars(probs)

        self._ref_scorrs = scorrs
        self._ref_asymms_1 = asymms_1
        self._ref_asymms_2 = asymms_2
        self._ref_ecop_dens_arrs = ecop_dens_arrs

        self._prep_ref_aux_flag = True
        return

    def _gen_sim_aux_data(self):

        assert self._prep_ref_aux_flag, 'Call _gen_ref_aux_data first!'

        if self._data_ref_rltzn.ndim != 1:
            raise NotImplementedError('Implementation for 1D only!')

        rands = np.random.random((self._data_ref_shape[0] // 2) - 1)

        phs_spec = -np.pi + (2 * np.pi * rands)

        ft = np.full(
            1 + (self._data_ref_shape[0] // 2), np.nan, dtype=np.complex)

        ft[+0] = self._ref_ft[+0]
        ft[-1] = self._ref_ft[-1]

        ft.real[1:-1] = np.cos(phs_spec) * self._ref_mag_spec[1:-1]
        ft.imag[1:-1] = np.sin(phs_spec) * self._ref_mag_spec[1:-1]

        assert np.all(np.isfinite(ft)), 'Invalid values in ft!'
        assert np.all(np.isfinite(phs_spec)), 'Invalid values in phs_spec!'

        data = np.fft.irfft(ft)

        assert np.all(np.isfinite(data)), 'Invalid values in data!'

        ranks, probs, norms = self._get_ranks_probs_norms(data)

        self._sim_rnk = ranks
        self._sim_nrm = norms

        self._sim_ft = ft
        self._sim_phs_spec = phs_spec
        self._sim_mag_spec = self._ref_mag_spec.copy()

        scorrs, asymms_1, asymms_2, ecop_dens_arrs = self._get_obj_vars(probs)

        self._sim_scorrs = scorrs
        self._sim_asymms_1 = asymms_1
        self._sim_asymms_2 = asymms_2
        self._sim_ecop_dens_arrs = ecop_dens_arrs

        self._prep_sim_aux_flag = True
        return

    def prepare(self):

        '''Generate data required before phase annealing starts'''

        PAS._PhaseAnnealingSettings__verify(self)
        assert self._sett_verify_flag, 'Settings in an unverfied state!'

        self._gen_ref_aux_data()
        assert self._prep_ref_aux_flag, (
            'Apparently, _gen_ref_aux_data did not finish as expected!')

        self._gen_sim_aux_data()
        assert self._prep_sim_aux_flag, (
            'Apparently, _gen_sim_aux_data did not finish as expected!')

        self._prep_prep_flag = True
        return

    def verify(self):

        assert self._prep_prep_flag, 'Call prepare first!'

        if self._vb:
            print_sl()

            print(f'Phase annealing preparation done successfully!')

            print_el()

        self._prep_verify_flag = True
        return

    __verify = verify
