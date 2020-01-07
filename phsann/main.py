'''
@author: Faizan

Dec 30, 2019

1:25:46 PM
'''

from pathos.multiprocessing import ProcessPool

import numpy as np

from ..simultexts.misc import print_sl, print_el, ret_mp_idxs

from .algorithm import PhaseAnnealingAlgorithm as PAA


class PhaseAnnealing(PAA):

    def __init__(self, verbose=True):

        PAA.__init__(self, verbose)

        self._main_alg_reals = None

        self._main_auto_temp_search_ress = None

        self._main_reals_gen_flag = False
        self._main_verify_flag = False
        return

    def _search_auto_temp(self):

        if self._vb:
            print_sl()

            print('Generating auto_init_temp realizations...')

            print_el()

        assert self._main_verify_flag, 'Call verify first!'

        self._alg_ann_runn_auto_init_temp_search_flag = True

        self._alg_sim_ann_init_temps = (
            [self._sett_ann_init_temp] * self._sett_ann_auto_init_temp_atpts)

        ann_init_temps = []
        auto_temp_search_ress = []

        reals_gen = (
            (
            (0, self._sett_ann_auto_init_temp_atpts),
            )
            for i in range(self._sett_misc_nreals))

        if self._sett_misc_ncpus > 1:

            mp_pool = ProcessPool(self._sett_misc_ncpus)

            mp_rets = list(
                mp_pool.uimap(self._get_realization_multi, reals_gen))

            mp_pool = None

        else:
            mp_rets = []
            for real_args in reals_gen:
                mp_rets.append(self._get_realization_multi(real_args))

        if self._vb:
            print(
                'Selected the following temperatures with their '
                'corresponding acceptance rates:')

        not_acptd_ct = 0
        for i in range(self._sett_misc_nreals):
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

        if self._vb:
            print('\n')

        if not_acptd_ct:
            raise RuntimeError(
                f'Could not find optimal simulated annealing inital '
                f'temperatures for {not_acptd_ct} out of '
                f'{self._sett_misc_nreals} simulations!')

        self._alg_sim_ann_init_temps = ann_init_temps
        self._main_auto_temp_search_ress = auto_temp_search_ress

        if self._vb:
            print_sl()

            print('Done generating auto_init_temp realizations.')

            print_el()

        self._alg_ann_runn_auto_init_temp_search_flag = False
        return

    def _generate_realizations_regular(self):

        if self._vb:
            print_sl()

            print('Generating regular realizations...')

            print_el()

        assert self._main_verify_flag, 'Call verify first!'

        self._main_alg_reals = []

        mp_idxs = ret_mp_idxs(self._sett_misc_nreals, self._sett_misc_ncpus)

        reals_gen = (
            (
            (mp_idxs[i], mp_idxs[i + 1]),
            )
            for i in range(mp_idxs.size - 1))

        if self._sett_misc_ncpus > 1:

            mp_pool = ProcessPool(self._sett_misc_ncpus)

            mp_rets = list(
                mp_pool.uimap(self._get_realization_multi, reals_gen))

            mp_pool = None

            for i in range(self._sett_misc_ncpus):
                self._main_alg_reals.extend(mp_rets[i])

        else:
            for real_args in reals_gen:
                self._main_alg_reals.extend(
                    self._get_realization_multi(real_args))

        if self._vb:
            print_sl()

            print('Done generating regular realizations.')

            print_el()

        return

    def generate_realizations(self):

        if self._sett_auto_temp_set_flag:
            self._search_auto_temp()

        self._generate_realizations_regular()

        self._main_reals_gen_flag = True
        return

    def verify(self):

        PAA._PhaseAnnealingAlgorithm__verify(self)
        assert self._alg_verify_flag, 'Algorithm in an unverified state!'

        self._main_verify_flag = True
        return
