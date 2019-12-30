'''
Created on Dec 27, 2019

@author: Faizan
'''

import numpy as np

from .prepare import PhaseAnnealingPrepare as PAP


class PhaseAnnealingAlgorithm(PAP):

    def __init__(self, verbose=True):

        PAP.__init__(self, verbose)

        self._alg_reals = None

        return

    def _get_obj_ftn_val(self):

        obj_val = 0.0

        if self._sett_obj_scorr_flag:
            obj_val += ((self._ref_scorrs - self._sim_scorrs) ** 2).sum()

        if self._sett_obj_symm_type_1_flag:
            obj_val += ((self._ref_asymms_1 - self._sim_asymms_1) ** 2).sum()

        if self._sett_obj_symm_type_2_flag:
            obj_val += ((self._ref_asymms_2 - self._sim_asymms_2) ** 2).sum()

        assert np.isfinite(obj_val)

        return obj_val

    def _update_sim(self, index, phs):

        self._sim_phs_spec[index] = phs

        self._sim_ft[index].real = np.cos(phs) * self._sim_mag_spec[index]
        self._sim_ft[index].imag = np.sin(phs) * self._sim_mag_spec[index]

        data = np.fft.rfft(self._sim_ft)

        ranks, probs, norms = self._get_ranks_norms(data)

        self._sim_rnk = ranks
        self._sim_nrm = norms

        scorrs, asymms_1, asymms_2 = self._get_scorrs_asymms(probs)

        if self._sett_obj_rank_corr_flag:
            self._sim_scorrs = scorrs

        if self._sett_obj_symm_type_1_flag:
            self._sim_asymms_1 = asymms_1

        if self._sett_obj_symm_type_2_flag:
            self._sim_asymms_2 = asymms_2

        return

    def _get_iter_index(self):

        index = np.random.random()
        index *= ((self._data_ref_shape[0] // 2) - 2)
        index += 1
        return index

    def _get_realization(self):

        if self._data_ref_data.ndim != 1:
            raise NotImplementedError('Implemention for 1D only!')

        runn_iter = 0
        iters_wo_accept = 0

        curr_temp = self._sett_ann_init_temp

        old_obj_val = self._get_obj_ftn_val()

        old_index = self._get_iter_index()

        while (runn_iter < self._sett_ann_max_iters) and (iters_wo_accept < self._sett_ann_max_iter_wo_chng):

            new_index = self._get_iter_index()

            index_ctr = 0
            while (old_index == new_index):
                new_index = self._get_iter_index()

                if index_ctr > 100:
                    raise RuntimeError('Something wrong is!')

                index_ctr += 1

            assert 0 < new_index < (self._data_ref_shape[0] // 2)

            old_phs = self._sim_phs_spec[new_index]
            new_phs = -np.pi + (2 * np.pi * np.random.random())

            self._update_sim(new_index, new_phs)

            new_obj_val = self._get_obj_ftn_val()

            if new_obj_val < old_obj_val:
                accept_flag = True

            else:
                rand_p = np.random.random()
                boltz_p = np.exp((old_obj_val - new_obj_val) / curr_temp)

                if rand_p < boltz_p:
                    accept_flag = True

                else:
                    accept_flag = False

            if accept_flag:
                old_index = new_index

                old_obj_val = new_obj_val

                iters_wo_accept = 0

            else:
                self._update_sim(new_index, old_phs)

                iters_wo_accept += 1

            runn_iter += 1

        return (
            self._sim_ft,
            self._sim_rnk,
            self._sim_rnk,
            self._sim_corrs,
            self._sim_asymms_1,
            self._sin_asymms_2)

    def generate_realizations(self):

        self._alg_reals = []

        for i in range(self._sett_misc_nreals):
            self._alg_reals.append(self._get_realization())

        return
