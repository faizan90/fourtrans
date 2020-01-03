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

    def generate_realizations(self):

        if self._vb:
            print_sl()

            print('Generating realizations...')

            print_el()

        assert self._main_verify_flag

        self._alg_reals = []

        if self._sett_misc_ncpus > 1:

#             old_vb = self._vb
#
#             self._vb = False

            mp_idxs = ret_mp_idxs(
                self._sett_misc_nreals, self._sett_misc_ncpus)

            self._mp_pool = ProcessPool(self._sett_misc_ncpus)

            reals_gen = (
                (
                (mp_idxs[i], mp_idxs[i + 1]),
                )
                for i in range(mp_idxs.size - 1))

            # TODO: have it more efficient by having results available
            # as soon as they arrive
            mp_rets = list(
                self._mp_pool.uimap(self._get_realization_multi, reals_gen))

            self._mp_pool = None

            for i in range(self._sett_misc_ncpus):
                self._alg_reals.extend(mp_rets[i])

#             self._vb = old_vb

        else:
            for i in range(self._sett_misc_nreals):
                self._alg_reals.append(self._get_realization_single((i,)))

        if self._vb:
            print_sl()

            print('Done generating realizations.')

            print_el()

        self._main_reals_gen_flag = True
        return

    def verify(self):

        PAA._PhaseAnnealingAlgorithm__verify(self)
        assert self._alg_verify_flag

        self._main_verify_flag = True
        return
