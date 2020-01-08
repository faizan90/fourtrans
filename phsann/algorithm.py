'''
Created on Dec 27, 2019

@author: Faizan
'''
from collections import deque
from timeit import default_timer

import numpy as np

from ..simultexts.misc import print_sl, print_el

from .prepare import PhaseAnnealingPrepare as PAP


class PhaseAnnealingAlgorithm(PAP):

    '''The main phase annealing algorithm'''

    def __init__(self, verbose=True):

        PAP.__init__(self, verbose)

        self._alg_sim_ann_init_temps = None

        self._alg_ann_runn_auto_init_temp_search_flag = False

        self._alg_verify_flag = False
        return

    def _get_init_temp(
            self,
            auto_init_temp_atpt,
            pre_init_temps,
            pre_acpt_rates,
            init_temp):

        if self._alg_ann_runn_auto_init_temp_search_flag:

            assert isinstance(auto_init_temp_atpt, int), (
                'auto_init_temp_atpt not an integer!')

            assert (
                (auto_init_temp_atpt >= 0) and
                (auto_init_temp_atpt < self._sett_ann_auto_init_temp_atpts)), (
                    'Invalid _sett_ann_auto_init_temp_atpts!')

            assert len(pre_acpt_rates) == len(pre_init_temps), (
                'Unequal size of pre_acpt_rates and pre_init_temps!')

            if auto_init_temp_atpt:
                pre_init_temp = pre_init_temps[-1]
                pre_acpt_rate = pre_acpt_rates[-1]

                assert isinstance(pre_init_temp, float), (
                    'pre_init_temp not a float!')

                assert (
                    (pre_init_temp >=
                     self._sett_ann_auto_init_temp_temp_bd_lo) and
                    (pre_init_temp <=
                     self._sett_ann_auto_init_temp_temp_bd_hi)), (
                         'Invalid pre_init_temp!')

                assert isinstance(pre_acpt_rate, float), (
                    'pre_acpt_rate not a float!')

                assert 0 <= pre_acpt_rate <= 1, 'Invalid pre_acpt_rate!'

            if auto_init_temp_atpt == 0:
                init_temp = self._sett_ann_auto_init_temp_temp_bd_lo

            else:
                temp_lo_bd = self._sett_ann_auto_init_temp_temp_bd_lo
                temp_lo_bd *= (
                    self._sett_ann_auto_init_temp_ramp_rate **
                    (auto_init_temp_atpt - 1))

                temp_hi_bd = (
                    temp_lo_bd * self._sett_ann_auto_init_temp_ramp_rate)

                init_temp = temp_lo_bd + (
                    (temp_hi_bd - temp_lo_bd) * np.random.random())

                assert temp_lo_bd <= init_temp <= temp_hi_bd, (
                    'Invalid init_temp!')

                if init_temp > self._sett_ann_auto_init_temp_temp_bd_hi:
                    init_temp = self._sett_ann_auto_init_temp_temp_bd_hi

            assert (
                self._sett_ann_auto_init_temp_temp_bd_lo <=
                init_temp <=
                self._sett_ann_auto_init_temp_temp_bd_hi), (
                    'Invalid init_temp!')

        else:
            assert isinstance(init_temp, float), 'init_temp not a float!'
            assert 0 <= init_temp, 'Invalid init_temp!'

        return init_temp

    def _get_stopp_criteria(self, test_vars):

        runn_iter, iters_wo_acpt, tol, curr_temp = test_vars

        stopp_criteria = False

        if self._alg_ann_runn_auto_init_temp_search_flag:
            stopp_criteria = (
                (runn_iter <= self._sett_ann_auto_init_temp_niters),
                )

        else:
            stopp_criteria = (
                (runn_iter <= self._sett_ann_max_iters),
                (iters_wo_acpt < self._sett_ann_max_iter_wo_chng),
                (tol > self._sett_ann_obj_tol),
                (not np.isclose(curr_temp, 0.0)),
                )

        return stopp_criteria

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

        assert np.isfinite(obj_val), 'Invalid obj_val!'

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

        assert 0 < index < (self._data_ref_shape[0] // 2), 'Invalid index!'

        return int(index)

    def _get_realization_multi(self, args):

        ((real_iter_beg, real_iter_end),
        ) = args

        reals = []
        pre_init_temps = []
        pre_acpt_rates = []

        for real_iter in range(real_iter_beg, real_iter_end):
            real_args = (
                real_iter,
                pre_init_temps,
                pre_acpt_rates,
                self._alg_sim_ann_init_temps[real_iter],
                )

            real = self._get_realization_single(real_args)

            reals.append(real)

            if self._alg_ann_runn_auto_init_temp_search_flag:
                pre_acpt_rates.append(real[0])
                pre_init_temps.append(real[1])

                if self._vb:
                    print('acpt_rate:', real[0], 'init_temp:', real[1])
                    print('\n')

                if real[0] >= self._sett_ann_auto_init_temp_acpt_bd_hi:
                    if self._vb:
                        print(
                            'Acceptance is at upper bounds, not looking '
                            'for initial temperature anymore!')

                    break

                if real[1] >= self._sett_ann_auto_init_temp_temp_bd_hi:
                    if self._vb:
                        print(
                            'Reached upper bounds of temperature, '
                            'not going any further!')

                    break

        return reals

    def _get_realization_single(self, args):

        (real_iter,
         pre_init_temps,
         pre_acpt_rates,
         init_temp) = args

        assert isinstance(real_iter, int), 'real_iter not integer!'

        if self._alg_ann_runn_auto_init_temp_search_flag:
            assert 0 <= real_iter < self._sett_ann_auto_init_temp_atpts, (
                    'Invalid real_iter!')

        else:
            assert 0 <= real_iter < self._sett_misc_nreals, (
                    'Invalid real_iter!')

        if self._vb:
            timer_beg = default_timer()

            print(f'Starting realization at index {real_iter}...')

        if self._data_ref_data.ndim != 1:
            raise NotImplementedError('Implemention for 1D only!')

        self._gen_sim_aux_data()

        runn_iter = 1  # 1-index due to temp_ratio
        iters_wo_acpt = 0
        tol = np.inf

        curr_temp = self._get_init_temp(
            real_iter, pre_init_temps, pre_acpt_rates, init_temp)

        old_obj_val = self._get_obj_ftn_val()

        old_index = self._get_sim_index()
        new_index = old_index

        tols = deque(maxlen=self._sett_ann_obj_tol_iters)

        all_tols = []
        all_obj_vals = []
        acpts_rjts = []

        stopp_criteria = self._get_stopp_criteria(
            (runn_iter, iters_wo_acpt, tol, curr_temp))

        while all(stopp_criteria):

            index_ctr = 0
            while (old_index == new_index):
                new_index = self._get_sim_index()

                if index_ctr > 100:
                    raise RuntimeError(
                        'Could not get an index that is different than '
                        'the previous!')

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

            acpts_rjts.append(accept_flag)

            tols.append(abs(old_obj_val - new_obj_val))

            if runn_iter >= tols.maxlen:
                tol = sum(tols) / float(tols.maxlen)
                assert np.isfinite(tol), 'Invalid tol!'

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

                iters_wo_acpt = 0

            else:
                self._update_sim(new_index, old_phs)

                iters_wo_acpt += 1

            runn_iter += 1

            if not self._alg_ann_runn_auto_init_temp_search_flag:
                if not (runn_iter % self._sett_ann_upt_evry_iter):
                    curr_temp *= self._sett_ann_temp_red_ratio
                    assert curr_temp >= 0.0, 'Invalid curr_temp!'

            stopp_criteria = self._get_stopp_criteria(
                (runn_iter, iters_wo_acpt, tol, curr_temp))

        if self._alg_ann_runn_auto_init_temp_search_flag:
            acpt_rate = sum(acpts_rjts) / len(acpts_rjts)

            ret = (acpt_rate, curr_temp)

        else:
            if self._sett_obj_scorr_flag:
                sim_scorrs = self._sim_scorrs

            else:
                sim_scorrs = np.array([], dtype=float)

            if self._sett_obj_asymm_type_1_flag:
                sim_asymms_1 = self._sim_asymms_1

            else:
                sim_asymms_1 = np.array([], dtype=float)

            if self._sett_obj_asymm_type_2_flag:
                sim_asymms_2 = self._sim_asymms_2

            else:
                sim_asymms_2 = np.array([], dtype=float)

            if self._sett_obj_ecop_dens_flag:
                sim_ecop_dens_arrs = self._sim_ecop_dens_arrs

            else:
                sim_ecop_dens_arrs = np.array([], dtype=float)

            ret = (
                self._sim_ft.copy(),
                self._sim_rnk.copy(),
                self._sim_nrm.copy(),
                sim_scorrs.copy(),
                sim_asymms_1.copy(),
                sim_asymms_2.copy(),
                sim_ecop_dens_arrs.copy(),
                runn_iter,
                iters_wo_acpt,
                tol,
                curr_temp,
                stopp_criteria,
                np.array(all_tols, dtype=np.float64),
                np.array(all_obj_vals, dtype=np.float64),
                np.array(acpts_rjts, dtype=int),
                )

        if self._vb:
            timer_end = default_timer()

            print(
                f'Done with realization at index {real_iter} in '
                f'{timer_end - timer_beg:0.3f} seconds.')

        return ret

    def verify(self):

        PAP._PhaseAnnealingPrepare__verify(self)
        assert self._prep_verify_flag, 'Prepare in an unverified state!'

        self._alg_sim_ann_init_temps = (
            [self._sett_ann_init_temp] * self._sett_misc_nreals)

        if self._vb:
            print_sl()

            print(
                'Phase annealing algorithm requirements verified '
                'successfully!')

            print_el()

        self._alg_verify_flag = True
        return

    __verify = verify
