'''
@author: Faizan

Dec 30, 2019

1:25:46 PM
'''

from .algorithm import PhaseAnnealingAlgorithm as PAA


class PhaseAnnealing(PAA):

    def __init__(self, verbose=True):

        PAA.__init__(self, verbose)

        self._alg_reals = None

        self._main_reals_gen_flag = False
        self._main_verify_flag = False
        return

    def generate_realizations(self):

        assert self._main_verify_flag

        self._alg_reals = []

        for i in range(self._sett_misc_nreals):
            self._alg_reals.append(self._get_realization())

        self._main_reals_gen_flag = True
        return

    def verify(self):

        PAA._PhaseAnnealingAlgorithm__verify(self)
        assert self._alg_verify_flag

        self._main_verify_flag = True
        return
