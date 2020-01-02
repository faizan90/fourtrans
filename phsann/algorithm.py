'''
Created on Dec 27, 2019

@author: Faizan
'''
from timeit import default_timer
from collections import deque

import numpy as np

from ..simultexts.misc import print_sl, print_el

from .prepare import PhaseAnnealingPrepare as PAP


class PhaseAnnealingAlgorithm(PAP):

    def __init__(self, verbose=True):

        PAP.__init__(self, verbose)

        self._alg_verify_flag = False
        return

    def _get_obj_ftn_val(self):

        obj_val = 0.0

        if self._sett_obj_scorr_flag:
            obj_val += ((self._ref_scorrs - self._sim_scorrs) ** 2).sum()

        if self._sett_obj_asymm_type_1_flag:
            obj_val += ((self._ref_asymms_1 - self._sim_asymms_1) ** 2).sum()

        if self._sett_obj_asymm_type_2_flag:
            obj_val += ((self._ref_asymms_2 - self._sim_asymms_2) ** 2).sum()

        if self._sett_obj_ecop_dens_flag:
            obj_val += (
                (self._ref_ecop_dens_arrs -
                 self._sim_ecop_dens_arrs) ** 2).sum()

        assert np.isfinite(obj_val)

        return obj_val

    def _update_sim(self, index, phs):

        self._sim_phs_spec[index] = phs

        self._sim_ft.real[index] = np.cos(phs) * self._sim_mag_spec[index]
        self._sim_ft.imag[index] = np.sin(phs) * self._sim_mag_spec[index]

        data = np.fft.irfft(self._sim_ft)

        ranks, probs, norms = self._get_ranks_probs_norms(data)

        self._sim_rnk = ranks
        self._sim_nrm = norms

        scorrs, asymms_1, asymms_2, ecop_dens_arrs = self._get_obj_vars(probs)

        self._sim_scorrs = scorrs
        self._sim_asymms_1 = asymms_1
        self._sim_asymms_2 = asymms_2
        self._sim_ecop_dens_arrs = ecop_dens_arrs

        return

    def _get_sim_index(self):

        index = np.random.random()
        index *= ((self._data_ref_shape[0] // 2) - 2)
        index += 1

        assert 0 < index < (self._data_ref_shape[0] // 2)

#         index = np.random.random()
#         index *= ((self._data_ref_shape[0] // 2) - 1)
#
#         assert 0 <= index < (self._data_ref_shape[0] // 2)

        return int(index)

    def _get_realization(self, args):

        real_iter, = args

        if self._vb:
            timer_beg = default_timer()

            print(f'Starting realization at index {real_iter}...')

        if self._data_ref_data.ndim != 1:
            raise NotImplementedError('Implemention for 1D only!')

        self._gen_sim_aux_data()

        runn_iter = 1  # 1-index due to temp_ratio
        iters_wo_accept = 0
        tol = np.inf

        curr_temp = self._sett_ann_init_temp

        old_obj_val = self._get_obj_ftn_val()

        old_index = self._get_sim_index()
        new_index = old_index

        tols = deque(maxlen=self._sett_ann_obj_tol_iters)

        all_tols = []
        all_obj_vals = []

        stopp_criteria = (
            (runn_iter <= self._sett_ann_max_iters),
            (iters_wo_accept < self._sett_ann_max_iter_wo_chng),
            (tol > self._sett_ann_obj_tol))

        while all(stopp_criteria):

            if not (runn_iter % self._sett_ann_upt_evry_iter):
                curr_temp *= self._sett_ann_temp_red_ratio
                assert not np.isclose(curr_temp, 0.0)
                assert curr_temp > 0.0

            index_ctr = 0
            while (old_index == new_index):
                new_index = self._get_sim_index()

                if index_ctr > 100:
                    raise RuntimeError('Something wrong is!')

                index_ctr += 1

            old_phs = self._sim_phs_spec[new_index]
            new_phs = -np.pi + (2 * np.pi * np.random.random())

#             assert not np.isclose(old_phs, new_phs), 'What are the chances?'

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

            # location is important
            tols.append(abs(old_obj_val - new_obj_val))

            if runn_iter >= tols.maxlen:
                tol = sum(tols) / float(tols.maxlen)
                assert np.isfinite(tol)

                all_tols.append(tol)

            all_obj_vals.append(new_obj_val)

#             if self._vb:
#                 print(
#                     runn_iter,
#                     curr_temp,
#                     accept_flag,
#                     old_obj_val,
#                     new_obj_val)

            if accept_flag:
                old_index = new_index

                old_obj_val = new_obj_val

                iters_wo_accept = 0

            else:
                self._update_sim(new_index, old_phs)

                iters_wo_accept += 1

            runn_iter += 1

            stopp_criteria = (
                (runn_iter <= self._sett_ann_max_iters),
                (iters_wo_accept < self._sett_ann_max_iter_wo_chng),
                (tol > self._sett_ann_obj_tol))

        if not self._sett_obj_scorr_flag:
            self._sim_scorrs = np.array([], dtype=float)

        if not self._sett_obj_asymm_type_1_flag:
            self._sim_asymms_1 = np.array([], dtype=float)

        if not self._sett_obj_asymm_type_2_flag:
            self._sim_asymms_2 = np.array([], dtype=float)

        if self._vb:
            timer_end = default_timer()

            print(
                f'Done with realization at index {real_iter} in '
                f'{timer_end - timer_beg:0.3f} seconds.')

        return (
            self._sim_ft.copy(),
            self._sim_rnk.copy(),
            self._sim_nrm.copy(),
            self._sim_scorrs.copy(),
            self._sim_asymms_1.copy(),
            self._sim_asymms_2.copy(),
            runn_iter,
            iters_wo_accept,
            tol,
            curr_temp,
            stopp_criteria,
            np.array(all_tols, dtype=np.float64),
            np.array(all_obj_vals, dtype=np.float64),
            )

    def verify(self):

        PAP._PhaseAnnealingPrepare__verify(self)
        assert self._prep_verify_flag

        if self._vb:
            print_sl()

            print(
                'Phase annealing algorithm requirements verified '
                'successfully!')

            print_el()

        self._alg_verify_flag = True
        return

    __verify = verify
