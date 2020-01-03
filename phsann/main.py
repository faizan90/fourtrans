'''
@author: Faizan

Dec 30, 2019

1:25:46 PM
'''

from pathos.multiprocessing import ProcessPool

# import numpy as np

from ..simultexts.misc import print_sl, print_el, ret_mp_idxs

from .algorithm import PhaseAnnealingAlgorithm as PAA


class PhaseAnnealing(PAA):

    def __init__(self, verbose=True):

        PAA.__init__(self, verbose)

        self._alg_reals = None

        self._mp_pool = None

        self._main_reals_gen_flag = False
        self._main_verify_flag = False
        return

    def _generate_realizations_auto_temp(self):

        if self._vb:
            print_sl()

            print('Generating auto_init_temp realizations...')

            print_el()

        assert self._main_verify_flag

        auto_init_temp_rets = []

        reals_gen = (
            (
            (0, self._sett_ann_init_temp_atpts),
            0,
            None,
            None,
            )
            for i in range(self._sett_misc_ncpus))

        if self._sett_misc_ncpus > 1:

            self._mp_pool = ProcessPool(self._sett_misc_ncpus)

            # TODO: have it more efficient by having results available
            # as soon as they arrive
            mp_rets = list(
                self._mp_pool.uimap(self._get_realization_multi, reals_gen))

            self._mp_pool = None

        else:
            mp_rets = [self._get_realization_multi(reals_gen.__next__())]

        for i in range(self._sett_misc_ncpus):
            auto_init_temp_rets.append(mp_rets[i])

        print(auto_init_temp_rets)

        raise Exception

        if self._vb:
            print_sl()

            print('Done generating auto_init_temp realizations.')

            print_el()

        return

    def _generate_realization_regular(self):

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
            None,
            None,
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

            self._alg_ann_runn_auto_init_temp_search_flag = True

            self._sett_ann_init_temp = self._generate_realizations_auto_temp()

            self._alg_ann_runn_auto_init_temp_search_flag = False

        self._generate_realization_regular()

        self._main_reals_gen_flag = True
        return

    def verify(self):

        PAA._PhaseAnnealingAlgorithm__verify(self)
        assert self._alg_verify_flag

        self._main_verify_flag = True
        return
