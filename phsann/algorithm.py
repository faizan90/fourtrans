'''
Created on Dec 27, 2019

@author: Faizan
'''
from timeit import default_timer
from collections import deque, namedtuple

import numpy as np
from pathos.multiprocessing import ProcessPool

from ..simultexts.misc import print_sl, print_el, ret_mp_idxs
from ..cyth import (
    get_asymms_sample,
    fill_bi_var_cop_dens,
    )

from .prepare import PhaseAnnealingPrepare as PAP

SimRltznData = namedtuple(
    'SimRltznData',
    ['ft',
     'rnk',
     'nrm',
     'scorrs',
     'asymms_1',
     'asymms_2',
     'ecop_dens',
     'runn_iter',
     'iters_wo_acpt',
     'tol',
     'fin_temp',
     'stopp_criteria',
     'tols_all',
     'obj_vals_all',
     'acpts_rjts',
     'acpt_rates',
     'obj_vals_min',
     'phss_all',
     'temps_all',
     'phs_red_rates_all',
    ]
    )


class PhaseAnnealingAlgorithm(PAP):

    '''The main phase annealing algorithm'''

    def __init__(self, verbose=True):

        PAP.__init__(self, verbose)

        self._alg_sim_ann_init_temps = None

        self._alg_ann_runn_auto_init_temp_search_flag = False

        self._alg_rltzns = None

        self._alg_auto_temp_search_ress = None

        self._alg_rltzns_gen_flag = False

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

        (runn_iter,
         iters_wo_acpt,
         tol,
         curr_temp,
         curr_phs_red_rate) = test_vars

        stopp_criteria = (
            (runn_iter < self._sett_ann_max_iters),
            (iters_wo_acpt < self._sett_ann_max_iter_wo_chng),
            (tol > self._sett_ann_obj_tol),
            (not np.isclose(curr_temp, 0.0)),
            (not np.isclose(curr_phs_red_rate, 0.0)),
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

    def _update_at_end(self, rnks):

        probs = rnks / (self._data_ref_shape[0] + 1.0)

        assert np.all((0 < probs) & (probs < 1)), 'probs out of range!'

        scorrs = np.full(self._sett_obj_lag_steps.size, np.nan)

        asymms_1 = np.full(self._sett_obj_lag_steps.size, np.nan)

        asymms_2 = np.full(self._sett_obj_lag_steps.size, np.nan)

        ecop_dens_arrs = np.full(
            (self._sett_obj_lag_steps.size,
             self._sett_obj_ecop_dens_bins,
             self._sett_obj_ecop_dens_bins),
             np.nan,
             dtype=np.float64)

        for i, lag in enumerate(self._sett_obj_lag_steps):
            rolled_probs = np.roll(probs, lag)

            scorrs[i] = np.corrcoef(probs, rolled_probs)[0, 1]

            asymms_1[i], asymms_2[i] = get_asymms_sample(
                probs, rolled_probs)

            asymms_1[i] = asymms_1[i] / self._get_asymm_1_max(scorrs[i])

            asymms_2[i] = asymms_2[i] / self._get_asymm_2_max(scorrs[i])

            fill_bi_var_cop_dens(
                probs, rolled_probs, ecop_dens_arrs[i, :, :])

        assert np.all(np.isfinite(scorrs)), 'Invalid values in scorrs!'

        assert np.all((scorrs >= -1.0) & (scorrs <= +1.0)), (
            'scorrs out of range!')

        assert np.all(np.isfinite(asymms_1)), 'Invalid values in asymms_1!'

        assert np.all((asymms_1 >= -1.0) & (asymms_1 <= +1.0)), (
            'asymms_1 out of range!')

        assert np.all(np.isfinite(asymms_2)), 'Invalid values in asymms_2!'

        assert np.all((asymms_2 >= -1.0) & (asymms_2 <= +1.0)), (
            'asymms_2 out of range!')

        assert np.all(np.isfinite(ecop_dens_arrs)), (
            'Invalid values in ecop_dens_arrs!')

        return scorrs, asymms_1, asymms_2, ecop_dens_arrs

    def _update_ref_at_end(self):

        (self._ref_scorrs,
         self._ref_asymms_1,
         self._ref_asymms_2,
         self._ref_ecop_dens_arrs) = self._update_at_end(self._ref_rnk)

        return

    def _update_sim_at_end(self):

        (self._sim_scorrs,
         self._sim_asymms_1,
         self._sim_asymms_2,
         self._sim_ecop_dens_arrs) = self._update_at_end(self._sim_rnk)

        return

    def _get_new_idx(self):

        index = np.random.random()
        index *= ((self._data_ref_shape[0] // 2) - 2)
        index += 1

        assert 0 < index < (self._data_ref_shape[0] // 2), 'Invalid index!'

        return int(index)

    def _get_rltzn_multi(self, args):

        ((rltzn_iter_beg, rltzn_iter_end),
        ) = args

        rltzns = []
        pre_init_temps = []
        pre_acpt_rates = []

        for rltzn_iter in range(rltzn_iter_beg, rltzn_iter_end):
            rltzn_args = (
                rltzn_iter,
                pre_init_temps,
                pre_acpt_rates,
                self._alg_sim_ann_init_temps[rltzn_iter],
                )

            rltzn = self._get_rltzn_single(rltzn_args)

            rltzns.append(rltzn)

            if self._alg_ann_runn_auto_init_temp_search_flag:
                pre_acpt_rates.append(rltzn[0])
                pre_init_temps.append(rltzn[1])

                if self._vb:
                    print('acpt_rate:', rltzn[0], 'init_temp:', rltzn[1])
                    print('\n')

                if rltzn[0] >= self._sett_ann_auto_init_temp_acpt_bd_hi:
                    if self._vb:
                        print(
                            'Acceptance is at upper bounds, not looking '
                            'for initial temperature anymore!')

                    break

                if rltzn[1] >= self._sett_ann_auto_init_temp_temp_bd_hi:
                    if self._vb:
                        print(
                            'Reached upper bounds of temperature, '
                            'not going any further!')

                    break

        return rltzns

    def _get_new_phs_and_idx(
            self, old_index, new_index, curr_phs_red_rate, runn_iter):

        index_ctr = 0
        while (old_index == new_index):
            new_index = self._get_new_idx()

            if index_ctr > 100:
                raise RuntimeError(
                    'Could not get an index that is different than '
                    'the previous!')

            index_ctr += 1

        old_phs = self._sim_phs_spec[new_index]

        new_phs = -np.pi + (2 * np.pi * np.random.random())

        if not self._alg_ann_runn_auto_init_temp_search_flag:
            if False:
                new_phs *= curr_phs_red_rate

            else:
                new_phs *= (
                    (self._sett_ann_max_iters - runn_iter) /
                    self._sett_ann_max_iters)

            new_phs += old_phs

            pi_ctr = 0
            while not (-np.pi <= new_phs <= +np.pi):
                if new_phs > +np.pi:
                    new_phs = -np.pi + (new_phs - np.pi)

                elif new_phs < -np.pi:
                    new_phs = +np.pi + (new_phs + np.pi)

                if pi_ctr > 100:
                    raise RuntimeError(
                        'Could not get a phase that is in range!')

                pi_ctr += 1

#         assert not np.isclose(old_phs, new_phs), 'What are the chances?'
        return old_phs, new_phs, new_index

    def _get_rltzn_single(self, args):

        (rltzn_iter,
         pre_init_temps,
         pre_acpt_rates,
         init_temp) = args

        assert isinstance(rltzn_iter, int), 'rltzn_iter not integer!'

        if self._alg_ann_runn_auto_init_temp_search_flag:
            assert 0 <= rltzn_iter < self._sett_ann_auto_init_temp_atpts, (
                    'Invalid rltzn_iter!')

        else:
            assert 0 <= rltzn_iter < self._sett_misc_n_rltzns, (
                    'Invalid rltzn_iter!')

        if self._vb:
            timer_beg = default_timer()

            print(f'Starting realization at index {rltzn_iter}...')

        if self._data_ref_rltzn.ndim != 1:
            raise NotImplementedError('Implemention for 1D only!')

        # randomize all phases before starting
        self._gen_sim_aux_data()

        runn_iter = 0
        iters_wo_acpt = 0
        tol = np.inf

        curr_temp = self._get_init_temp(
            rltzn_iter, pre_init_temps, pre_acpt_rates, init_temp)

        curr_phs_red_rate = self._sett_ann_phs_red_rate

        old_obj_val = self._get_obj_ftn_val()

        old_index = self._get_new_idx()
        new_index = old_index

        tols = deque(maxlen=self._sett_ann_obj_tol_iters)

        tols_all = []
        obj_vals_all = []
        obj_vals_min = []
        acpts_rjts = []

        phss_all = []
        temps_all = [[runn_iter, curr_temp]]
        phs_red_rates_all = [[runn_iter, curr_phs_red_rate]]

        stopp_criteria = self._get_stopp_criteria(
            (runn_iter, iters_wo_acpt, tol, curr_temp, curr_phs_red_rate))

        while all(stopp_criteria):

            old_phs, new_phs, new_index = self._get_new_phs_and_idx(
                old_index, new_index, curr_phs_red_rate, runn_iter)

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

            if not self._alg_ann_runn_auto_init_temp_search_flag:
                phss_all.append(new_phs)

            if runn_iter >= tols.maxlen:
                tol = sum(tols) / float(tols.maxlen)
                assert np.isfinite(tol), 'Invalid tol!'

                tols_all.append(tol)

            obj_vals_all.append(new_obj_val)

            if accept_flag:
                old_index = new_index

                old_obj_val = new_obj_val

                iters_wo_acpt = 0

                obj_vals_min.append(new_obj_val)

            else:
                self._update_sim(new_index, old_phs)

                iters_wo_acpt += 1

                obj_vals_min.append(old_obj_val)

            runn_iter += 1

            if not self._alg_ann_runn_auto_init_temp_search_flag:

                if not (runn_iter % self._sett_ann_upt_evry_iter):

                    temps_all.append([runn_iter - 1, curr_temp])

                    curr_temp *= self._sett_ann_temp_red_rate

                    assert curr_temp >= 0.0, 'Invalid curr_temp!'

                    temps_all.append([runn_iter, curr_temp])

                    phs_red_rates_all.append(
                        [runn_iter - 1, curr_phs_red_rate])

                    curr_phs_red_rate *= self._sett_ann_phs_red_rate

                    assert curr_phs_red_rate >= 0.0, (
                        'Invalid curr_phs_red_rate!')

                    phs_red_rates_all.append(
                        [runn_iter, curr_phs_red_rate])

                    iters_wo_acpt = 0

                stopp_criteria = self._get_stopp_criteria(
                    (runn_iter,
                     iters_wo_acpt,
                     tol,
                     curr_temp,
                     curr_phs_red_rate))

            else:
                stopp_criteria = (
                    (runn_iter <= self._sett_ann_auto_init_temp_niters),
                    )

        if self._alg_ann_runn_auto_init_temp_search_flag:
            acpt_rate = sum(acpts_rjts) / len(acpts_rjts)

            ret = (acpt_rate, curr_temp)

        else:
            self._update_sim_at_end()

            acpts_rjts = np.array(acpts_rjts, dtype=bool)

            acpt_rates = (
                np.cumsum(acpts_rjts) /
                np.arange(1, acpts_rjts.size + 1, dtype=float))

            ret = SimRltznData._make((
                self._sim_ft.copy(),
                self._sim_rnk.copy(),
                self._sim_nrm.copy(),
                self._sim_scorrs.copy(),
                self._sim_asymms_1.copy(),
                self._sim_asymms_2.copy(),
                self._sim_ecop_dens_arrs.copy(),
                runn_iter,
                iters_wo_acpt,
                tol,
                curr_temp,
                np.array(stopp_criteria),
                np.array(tols_all, dtype=np.float64),
                np.array(obj_vals_all, dtype=np.float64),
                acpts_rjts,
                acpt_rates,
                np.array(obj_vals_min, dtype=np.float64),
                np.array(phss_all, dtype=np.float64),
                np.array(temps_all, dtype=np.float64),
                np.array(phs_red_rates_all, dtype=np.float64),
                ))

        if self._vb:
            timer_end = default_timer()

            print(
                f'Done with realization at index {rltzn_iter} in '
                f'{timer_end - timer_beg:0.3f} seconds.')

        return ret

    def _auto_temp_search(self):

        if self._vb:
            print_sl()

            print('Generating auto_init_temp realizations...')

            print_el()

        assert self._alg_verify_flag, 'Call verify first!'

        self._alg_ann_runn_auto_init_temp_search_flag = True

        self._alg_sim_ann_init_temps = (
            [self._sett_ann_init_temp] * self._sett_ann_auto_init_temp_atpts)

        ann_init_temps = []
        auto_temp_search_ress = []

        rltzns_gen = (
            (
            (0, self._sett_ann_auto_init_temp_atpts),
            )
            for i in range(self._sett_misc_n_rltzns))

        if self._sett_misc_n_cpus > 1:

            mp_pool = ProcessPool(self._sett_misc_n_cpus)
            mp_pool.restart(True)

            mp_rets = list(
                mp_pool.uimap(self._get_rltzn_multi, rltzns_gen))

            mp_pool.close()
            mp_pool.join()

            mp_pool = None

        else:
            mp_rets = []
            for rltzn_args in rltzns_gen:
                mp_rets.append(self._get_rltzn_multi(rltzn_args))

        if self._vb:
            print(
                'Selected the following temperatures with their '
                'corresponding acceptance rates:')

        not_acptd_ct = 0
        for i in range(self._sett_misc_n_rltzns):
            acpt_rates_temps = np.atleast_2d(mp_rets[i])

            auto_temp_search_ress.append(acpt_rates_temps)

            within_range_idxs = (
                (self._sett_ann_auto_init_temp_acpt_bd_lo <=
                 acpt_rates_temps[:, 0]) &
                (self._sett_ann_auto_init_temp_acpt_bd_hi >=
                 acpt_rates_temps[:, 0]))

            if not within_range_idxs.sum():
                acpt_rate = np.nan
                ann_init_temp = np.nan

            else:
                acpt_rates_temps = np.atleast_2d(
                    acpt_rates_temps[within_range_idxs, :])

                best_acpt_rate_idx = np.argmin(
                    (acpt_rates_temps[:, 0] -
                     self._sett_ann_auto_init_temp_trgt_acpt_rate) ** 2)

                acpt_rate, ann_init_temp = acpt_rates_temps[
                    best_acpt_rate_idx, :]

            ann_init_temps.append(ann_init_temp)

            if not (
                self._sett_ann_auto_init_temp_acpt_bd_lo <=
                acpt_rate <=
                self._sett_ann_auto_init_temp_acpt_bd_hi):

                not_acptd_ct += 1

            if self._vb:
                print(f'Realization {i:04d}:', ann_init_temp, acpt_rate)

        if not_acptd_ct:
            raise RuntimeError(
                f'\nCould not find optimal simulated annealing inital '
                f'temperatures for {not_acptd_ct} out of '
                f'{self._sett_misc_n_rltzns} simulations!')

        self._alg_sim_ann_init_temps = ann_init_temps
        self._alg_auto_temp_search_ress = auto_temp_search_ress

        if self._vb:
            print_sl()

            print('Done generating auto_init_temp realizations.')

            print_el()

        self._alg_ann_runn_auto_init_temp_search_flag = False
        return

    def _gen_rltzns_rglr(self):

        if self._vb:
            print_sl()

            print('Generating regular realizations...')

            print_el()

        assert self._alg_verify_flag, 'Call verify first!'

        self._alg_rltzns = []

        mp_idxs = ret_mp_idxs(self._sett_misc_n_rltzns, self._sett_misc_n_cpus)

        rltzns_gen = (
            (
            (mp_idxs[i], mp_idxs[i + 1]),
            )
            for i in range(mp_idxs.size - 1))

        if self._sett_misc_n_cpus > 1:

            mp_pool = ProcessPool(self._sett_misc_n_cpus)
            mp_pool.restart(True)

            mp_rets = list(
                mp_pool.uimap(self._get_rltzn_multi, rltzns_gen))

            mp_pool.close()
            mp_pool.join()

            mp_pool = None

            for i in range(self._sett_misc_n_cpus):
                self._alg_rltzns.extend(mp_rets[i])

        else:
            for rltzn_args in rltzns_gen:
                self._alg_rltzns.extend(
                    self._get_rltzn_multi(rltzn_args))

        if self._vb:
            print_sl()

            print('Done generating regular realizations.')

            print_el()

        return

    def generate_realizations(self):

        '''Start the phase annealing algorithm'''

        if self._sett_auto_temp_set_flag:
            self._auto_temp_search()

        self._gen_rltzns_rglr()

        self._update_ref_at_end()

        self._alg_rltzns_gen_flag = True
        return

    def get_realizations(self):

        assert self._alg_rltzns_gen_flag, 'Call generate_realizations first!'

        return self._alg_rltzns

    def verify(self):

        PAP._PhaseAnnealingPrepare__verify(self)
        assert self._prep_verify_flag, 'Prepare in an unverified state!'

        self._alg_sim_ann_init_temps = (
            [self._sett_ann_init_temp] * self._sett_misc_n_rltzns)

        if self._vb:
            print_sl()

            print(
                'Phase annealing algorithm requirements verified '
                'successfully!')

            print_el()

        self._alg_verify_flag = True
        return

    __verify = verify
