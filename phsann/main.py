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

        self._alg_reals = None

        self._mp_pool = None
        self._sim_ann_init_temps = None
        self._auto_temp_search_ress = None

        self._main_reals_gen_flag = False
        self._main_verify_flag = False
        return

    def _search_auto_temp(self):

        if self._vb:
            print_sl()

            print('Generating auto_init_temp realizations...')

            print_el()

        assert self._main_verify_flag

        self._alg_ann_runn_auto_init_temp_search_flag = True

        ann_init_temps = []
        auto_temp_search_ress = []

        reals_gen = (
            (
            (0, self._sett_ann_auto_init_temp_atpts),
            0,
            None,
            )
            for i in range(self._sett_misc_nreals))

        if self._sett_misc_ncpus > 1:

            self._mp_pool = ProcessPool(self._sett_misc_ncpus)

            # TODO: have it more efficient by having results available
            # as soon as they arrive
            mp_rets = list(
                self._mp_pool.uimap(self._get_realization_multi, reals_gen))

            self._mp_pool = None

        else:
            mp_rets = []
            for real_args in reals_gen:
                mp_rets.append(self._get_realization_multi(real_args))

        mean_acpt_rate = (
            0.5 * (self._sett_ann_auto_init_temp_acpt_bd_lo +
                   self._sett_ann_auto_init_temp_acpt_bd_hi))

        if self._vb:
            print(
                'Selected the following temperatures with their '
                'corresponding acceptance rates:')

        not_acptd_ct = 0
        for i in range(self._sett_misc_nreals):
            auto_temp_search_ress.append(mp_rets[i])

            acpt_rates_temps = np.array(mp_rets[i])

            best_acpt_rate_idx = np.argmin(
                (acpt_rates_temps[:, 0] - mean_acpt_rate) ** 2)

            ann_init_temp = acpt_rates_temps[:, 1][best_acpt_rate_idx]

            ann_init_temps.append(ann_init_temp)

            if not (
                self._sett_ann_auto_init_temp_acpt_bd_lo <=
                acpt_rates_temps[best_acpt_rate_idx][0] <=
                self._sett_ann_auto_init_temp_acpt_bd_hi):

                not_acptd_ct += 1

            if self._vb:
                print(
                    f'Realization {i:04d}:',
                    acpt_rates_temps[best_acpt_rate_idx][1],
                    acpt_rates_temps[best_acpt_rate_idx][0])

                print('\n')

        if not_acptd_ct:
            raise RuntimeError(
                f'Could not find optimal simulated annealing inital '
                f'temperatures for {not_acptd_ct} out of '
                f'{self._sett_misc_nreals} simulations!')

        self._sim_ann_init_temps = ann_init_temps
        self._auto_temp_search_ress = auto_temp_search_ress

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

        assert self._main_verify_flag

        self._alg_reals = []

        mp_idxs = ret_mp_idxs(
            self._sett_misc_nreals, self._sett_misc_ncpus)

        reals_gen = (
            (
            (mp_idxs[i], mp_idxs[i + 1]),
            None,
            0.0
            )
            for i in range(mp_idxs.size - 1))

        if self._sett_misc_ncpus > 1:

            self._mp_pool = ProcessPool(self._sett_misc_ncpus)

            # TODO: have it more efficient by having results available
            # as soon as they arrive
            mp_rets = list(
                self._mp_pool.uimap(self._get_realization_multi, reals_gen))

            self._mp_pool = None

            for i in range(self._sett_misc_ncpus):
                self._alg_reals.extend(mp_rets[i])

        else:
            for real_args in reals_gen:
                self._alg_reals.append(self._get_realization_single(real_args))

        if self._vb:
            print_sl()

            print('Done generating regular realizations.')

            print_el()

        return

    def generate_realizations(self):

        if self._sett_ann_auto_init_temp_search_flag:
            self._sett_ann_init_temp = self._search_auto_temp()

        self._generate_realizations_regular()

        self._main_reals_gen_flag = True
        return

    def verify(self):

        PAA._PhaseAnnealingAlgorithm__verify(self)
        assert self._alg_verify_flag

        self._main_verify_flag = True
        return
