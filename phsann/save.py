'''
@author: Faizan-Uni

Jan 16, 2020

9:39:36 AM
'''
from fnmatch import fnmatch

import numpy as np
#
# from ..simultexts.misc import print_sl, print_el

from .algorithm import PhaseAnnealingAlgorithm as PAA


class PhaseAnnealingSave(PAA):

    '''
    Save reference, realizations flags, settings to HDF5
    '''

    def __init__(self, verbose):

        PAA.__init__(self, verbose)

        self._save_verify_flag = True
        return

    def _get_flags(self):

        flags = []
        for var in vars(self):
            if not fnmatch(var, '*_flag'):
                continue

            flag = getattr(self, var)

            if not isinstance(flag, bool):
                continue

            flags.append(flag)

        assert flags, 'No flags selected!'

        return np.array(flags, dtype=bool)

    def save_realizations(self):

        assert self._alg_rltzns_gen_flag, 'Call generate realizations first!'

        flags = self._get_flags()

        return

    def verify(self):

        PAA._PhaseAnnealingAlgorithm__verify(self)
        assert self._alg_verify_flag, 'Algorithm in an unverified state!'

        assert self._sett_misc_outs_dir.exists(), 'outputs_dir does not exist!'

        self._save_verify_flag = True
        return

    __verify = verify
