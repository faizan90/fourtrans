'''
Created on Dec 27, 2019

@author: Faizan
'''

import numpy as np

from ..simultexts.misc import print_sl, print_el


class PhaseAnnealingData:

    def __init__(self, verbose=True):

        assert isinstance(verbose, bool), 'verbose not a boolean!'

        self._vb = verbose

        self._data_ref_data = None
        self._data_ref_shape = None

        self._data_min_pts = 3

        self._data_ref_set_flag = False
        self._data_verify_flag = False
        return

    def set_reference_data(self, ref_data):

        if self._vb:
            print_sl()

            print('Setting reference data for phase annealing...\n')

        assert isinstance(ref_data, np.ndarray), 'ref_data not a numpy array!'
        assert ref_data.ndim == 1, 'ref_data not a 1D array!'
        assert np.all(np.isfinite(ref_data)), 'Invalid values in ref_data!'
        assert ref_data.dtype == np.float64, 'ref_data dtype not np.float64!'

        if ref_data.shape[0] % 2:
            ref_data = ref_data[:-1]

            print('Warning: dropped last step for even steps!\n')

        assert 0 < self._data_min_pts <= ref_data.shape[0], (
            'ref_data has too few steps!')

        self._data_ref_data = ref_data
        self._data_ref_shape = ref_data.shape

        if self._vb:
            print(f'Reference data set with shape: {self._data_ref_shape}')

            print_el()

        self._data_ref_set_flag = True
        return

    def verify(self):

        assert self._data_ref_set_flag, 'Call set_reference_data first!'

        if self._vb:
            print_sl()

            print(f'Phase annealing data verified successfully!')

            print_el()

        self._data_verify_flag = True
        return

    __verify = verify
