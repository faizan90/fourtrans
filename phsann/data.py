'''
Created on Dec 27, 2019

@author: Faizan
'''

import numpy as np

from ..simultexts.misc import print_sl, print_el


class PhaseAnnealingData:

    def __init__(self, verbose=True):

        assert isinstance(verbose, bool)

        self._vb = verbose

        self._data_ref_data = None
        self._data_ref_shape = None

        self._data_min_pts = 3

        self._data_ref_set_flag = False
        self._data_verify_flag = False
        return

    def set_reference_data(self, ref_data):

        assert isinstance(ref_data, np.ndarray)
        assert ref_data.ndim == 1
        assert np.all(np.isfinite(ref_data))
        assert ref_data.dtype == np.float64

        if ref_data.shape[0] % 2:
            ref_data = ref_data[:-1]

            print_sl()

            print('Warning: dropped last step for even steps!')

            print_el()

        assert 0 < self._data_min_pts <= ref_data.shape[0]

        self._data_ref_data = ref_data
        self._data_ref_shape = ref_data.shape

        if self._vb:
            print_sl()

            print(
                f'Reference data set with a shape of {self._data_ref_shape}!')

            print_el()

        self._data_ref_set_flag = True
        return

    def verify(self):

        assert self._data_ref_set_flag

        if self._vb:
            print_sl()

            print(f'Phase annealing data verified successfully!')

            print_el()

        self._data_verify_flag = True
        return

    __verify = verify
