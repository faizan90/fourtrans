'''
Created on Dec 27, 2019

@author: Faizan
'''

import numpy as np

from ..simultexts.misc import print_sl, print_el


class PhaseAnnealingData:

    '''Set the reference data'''

    def __init__(self, verbose=True):

        '''
        Parameters
        ----------
        verbose : bool
            Whether to show activity messages
        '''

        assert isinstance(verbose, bool), 'verbose not a boolean!'

        self._vb = verbose

        self._data_ref_rltzn = None
        self._data_ref_shape = None

        self._data_min_pts = 3

        self._data_ref_set_flag = False
        self._data_verify_flag = False
        return

    def set_reference_data(self, ref_rltzn):

        '''
        Set the reference data array

        Parameters
        ----------
        ref_rltzn : 1D float64 np.ndarray
            The reference realization/data array. No NaNs or Infinitys allowed.
        '''

        if self._vb:
            print_sl()

            print('Setting reference data for phase annealing...\n')

        assert isinstance(ref_rltzn, np.ndarray), (
            'ref_rltzn not a numpy array!')

        assert ref_rltzn.ndim == 1, 'ref_rltzn not a 1D array!'
        assert np.all(np.isfinite(ref_rltzn)), 'Invalid values in ref_rltzn!'
        assert ref_rltzn.dtype == np.float64, 'ref_rltzn dtype not np.float64!'

        if ref_rltzn.shape[0] % 2:
            ref_rltzn = ref_rltzn[:-1]

            print('Warning: dropped last step for even steps!\n')

        assert 0 < self._data_min_pts <= ref_rltzn.shape[0], (
            'ref_rltzn has too few steps!')

        self._data_ref_rltzn = ref_rltzn
        self._data_ref_shape = ref_rltzn.shape

        if self._vb:
            print(
                f'Reference realization set with shape: '
                f'{self._data_ref_shape}')

            print_el()

        self._data_ref_set_flag = True
        return

    def verify(self):

        '''Verify if data has been set correctly'''

        assert self._data_ref_set_flag, 'Call set_reference_data first!'

        if self._vb:
            print_sl()

            print(f'Phase annealing data verified successfully!')

            print_el()

        self._data_verify_flag = True
        return

    __verify = verify
