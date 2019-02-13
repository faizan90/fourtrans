'''
Created on Feb 4, 2019

@author: Faizan-Uni
'''

import psutil
from pathlib import Path

import pandas as pd
import numpy as np

from .misc import print_sl, print_el


class SimultaneousExtremesDataAndSettings:

    def __init__(self, verbose=True):

        self._vb = bool(verbose)

        self._n_cpus = 1

        self._set_data_flag = False
        self._set_out_dir_flag = False
        self._set_ret_prd_flag = False
        self._set_tws_flag = False
        self._set_n_sims_flag = False
        self._set_misc_sett_flag = False

        self._set_data_verify_flag = False
        return

    def set_data(self, time_ser_df):

        assert isinstance(time_ser_df, pd.DataFrame), (
            'time_ser_df not a pandas DataFrame object!')

        assert isinstance(time_ser_df.index, pd.DatetimeIndex), (
            'time_ser_df\'s index not a pandas DatetimeIndex object!')

        assert np.issubdtype(time_ser_df.values.dtype, np.floating), (
            'time_ser_df\'s values not of floating type!')

        assert not np.isinf(time_ser_df.values).sum(), (
            'time_ser_df cannot have infinity in it!')

        assert time_ser_df.shape[0], 'No time steps in time_ser_df!'

        assert time_ser_df.shape[1], (
            'Atleast two columns required in time_ser_df!')

        assert not time_ser_df.index.duplicated(keep=False).sum(), (
            'Duplicate time steps in time_ser_df!')

        self._data_df = time_ser_df

        self._data_df.columns = [str(stn) for stn in self._data_df.columns]

        if self._vb:
            print_sl()

            print('INFO: Set input time series dataframe as following:')
            print('\t', f'Number of steps: {self._data_df.shape[0]}')
            print('\t', f'Number of stations: {self._data_df.shape[1]}')
            print('\t', 'Cast the station labels to strings!')

            cts_ser = time_ser_df.count()

            print('\t', 'Stations and their valid values\' count:')
            for stn in cts_ser.index:
                print(2 * '\t', f'{stn:>10s}:{cts_ser[stn]:<10d}')

            print_el()

        self._set_data_flag = True
        return

    def set_outputs_directory(self, out_dir):

        assert isinstance(out_dir, (Path, str)), (
            'out_dir not a string or a Path-like object!')

        out_dir = Path(out_dir).absolute()

        assert out_dir.parents[0].exists(), (
            'Parent directory of the out_dir does not exist!')

        self._out_dir = out_dir

        if self._vb:
            print_sl()

            print('INFO: Set the outputs directory as following:')
            print('\t', f'{str(self._out_dir)}')

            print_el()

        self._set_out_dir_flag = True
        return

    def set_return_periods(self, return_periods):

        assert isinstance(return_periods, np.ndarray), (
            'return_periods not a numpy array!')

        assert return_periods.ndim == 1, (
            'return_periods cannot have more than one dimension!')

        assert np.issubdtype(return_periods.dtype, np.floating), (
            'return_periods\'s values not of floating type!')

        assert np.all(np.isfinite(return_periods)), (
            'Invalid values in return_periods!')

        assert np.all((return_periods > 0) & (return_periods < 1)), (
            'Return periods can\'t be less than or equal to zero or '
            'greater than or equal to one!')

        assert return_periods.shape[0], 'No elements in return_periods!'

        assert np.all(np.unique(return_periods) == np.sort(return_periods)), (
            'Non-unique elements in return_periods!')

        # sorting is important! the algorithm depends on it
        self._rps = np.sort(return_periods)

        if self._vb:
            print_sl()

            print('INFO: Set the return periods as following:')
            print('\t', f'Number of return periods: {self._rps.shape[0]}')
            print('\t', f'{self._rps}')

            print_el()

        self._set_ret_prd_flag = True
        return

    def set_time_windows(self, time_windows):

        assert isinstance(time_windows, np.ndarray), (
            'time_windows not a numpy array!')

        assert time_windows.ndim == 1, (
            'time_windows cannot have more than one dimension!')

        assert np.issubdtype(time_windows.dtype, np.integer), (
            'time_windows\'s values not of integer type!')

        assert np.all(time_windows >= 0), (
            'Time windows can\'t be less than zero!')

        assert time_windows.shape[0], 'No elements in time_windows!'

        assert np.all(
            np.unique(time_windows).shape[0] == time_windows.shape[0]), (
                'Non-unique elements in time_windows!')

        # sorting is important! the algorithm depends on it
        self._tws = np.sort(time_windows)

        if self._vb:
            print_sl()

            print('INFO: Set the time windows as following:')
            print('\t', f'Number of time windows: {self._tws.shape[0]}')
            print('\t', f'{self._tws}')

            print_el()

        self._set_tws_flag = True
        return

    def set_number_of_simulations(self, n_sims):

        assert isinstance(n_sims, int), 'n_sims not an integer!'

        assert n_sims > 0, 'n_sims should be greater than zero!'

        self._n_sims = n_sims

        if self._vb:
            print_sl()

            print(f'INFO: Set the number of simulations to: {self._n_sims}')

            print_el()

        self._set_n_sims_flag = True
        return

    def set_misc_settings(self, n_cpus):

        if isinstance(n_cpus, int):
            assert n_cpus > 0, 'n_cpus has to be one or more!'

        elif isinstance(n_cpus, str):
            assert n_cpus == 'auto'

            n_cpus = max(1, psutil.cpu_count() - 1)

        else:
            raise AssertionError('n_cpus can be an integer or \'auto\' only!')

        self._n_cpus = n_cpus

        if self._vb:
            print_sl()

            print(f'INFO: Set the number of running processes to: '
                f'{self._n_cpus}')

            print_el()

        self._set_misc_sett_flag = True
        return

    def verify(self):

        assert self._set_data_flag, 'Data not set!'
        assert self._set_out_dir_flag, 'Outputs directory not set!'
        assert self._set_ret_prd_flag, 'Return periods not set!'
        assert self._set_tws_flag, 'Time windows not set!'
        assert self._set_n_sims_flag, 'Number of simulations not set!'

        assert int(1.0 / self._rps.min()) < self._data_df.shape[0], (
            'Return period(s) longer than available data time steps!')

        assert np.any(self._data_df.count().values > (2 * self._tws).max()), (
            'Time windows are wider than available data time steps '
            'for all series!')

        if self._vb:
            print_sl()

            print('INFO: All data verified to be correct!')

            print_el()

        self._set_data_verify_flag = True
        return

    __verify = verify
