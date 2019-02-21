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
        self._ext_steps = 0

        self._save_sim_cdfs_flag = False
        self._save_sim_acorrs_flag = False
        self._save_sim_ft_cumm_corrs_flag = False

        self._set_data_flag = False
        self._set_out_dir_flag = False
        self._h5_path_set_flag = False
        self._set_excd_probs_flag = False
        self._set_tws_flag = False
        self._set_n_sims_flag = False

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

        assert time_ser_df.shape[1] >= 2, (
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

    def set_output_hdf5_path(self, hdf5_path):

        assert isinstance(hdf5_path, (str, Path)), (
            'hdf5_path not a string or Path object!')

        hdf5_path = Path(hdf5_path).absolute()

        self._h5_path = hdf5_path

        self._h5_path_set_flag = True
        return

    def set_exceedance_probabilities(self, exceedance_probabilities):

        assert isinstance(exceedance_probabilities, np.ndarray), (
            'exceedance_probabilities not a numpy array!')

        assert exceedance_probabilities.ndim == 1, (
            'exceedance_probabilities cannot have more than one dimension!')

        assert np.issubdtype(exceedance_probabilities.dtype, np.floating), (
            'exceedance_probabilities\'s values not of floating type!')

        assert np.all(np.isfinite(exceedance_probabilities)), (
            'Invalid values in exceedance_probabilities!')

        assert np.all(
            (exceedance_probabilities > 0) &
            (exceedance_probabilities < 1)), (
                'All values in exceedance_probabilities must be between'
                'zero and one!')

        assert exceedance_probabilities.shape[0], (
            'No elements in exceedance_probabilities!')

        assert np.all(
            np.unique(exceedance_probabilities) ==
            np.sort(exceedance_probabilities)), (
                'Non-unique elements in exceedance_probabilities!')

        self._eps = np.sort(exceedance_probabilities)

        if self._vb:
            print_sl()

            print('INFO: Set the exceedance probabilities as following:')

            print(
                '\t',
                f'Number of exceedance probabilities: {self._eps.shape[0]}')

            print('\t', f'Exceedance probabilities: {self._eps}')

            print_el()

        self._set_excd_probs_flag = True
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

        self._tws = np.sort(time_windows)

        if self._vb:
            print_sl()

            print('INFO: Set the time windows as following:')
            print('\t', f'Number of time windows: {self._tws.shape[0]}')
            print('\t', f'Time windows: {self._tws}')

            print_el()

        self._set_tws_flag = True
        return

    def set_number_of_simulations(self, n_sims):

        assert isinstance(n_sims, int), 'n_sims not an integer!'

        assert n_sims > 0, 'n_sims should be greater than zero!'

        self._n_sims = n_sims

        if self._vb:
            print_sl()

            print(f'INFO: Set the number of simulations to {self._n_sims}')

            print_el()

        self._set_n_sims_flag = True
        return

    def set_more_data_flags(
            self,
            sim_cdfs_flag,
            sim_auto_corrs_flag,
            sim_ft_cumm_corrs_flag):

        assert isinstance(sim_cdfs_flag, bool), (
            'sim_cdfs_flag not a boolean value!')

        assert isinstance(sim_auto_corrs_flag, bool), (
            'sim_auto_corrs_flag not a boolean value!')

        assert isinstance(sim_ft_cumm_corrs_flag, bool), (
            'sim_ft_corrs_flag not a boolean value!')

        self._save_sim_cdfs_flag = sim_cdfs_flag
        self._save_sim_acorrs_flag = sim_auto_corrs_flag
        self._save_sim_ft_cumm_corrs_flag = sim_ft_cumm_corrs_flag

        if self._vb:
            print_sl()

            print(
                f'INFO: Set the following additional information '
                f'computation flags:')

            print(
                '\t',
                f'Set sim_cdfs_flag to {self._save_sim_cdfs_flag}')

            print(
                '\t',
                f'Set sim_auto_corrs_flag to {self._save_sim_acorrs_flag}')

            print(
                '\t',
                f'Set sim_ft_cumm_corrs_flag to '
                f'{self._save_sim_ft_cumm_corrs_flag}')

            print_el()
        return

    def set_misc_settings(self, n_cpus=1, extend_steps=0):

        if isinstance(n_cpus, int):
            assert n_cpus > 0, 'n_cpus has to be one or more!'

        elif isinstance(n_cpus, str):
            assert n_cpus == 'auto'

            n_cpus = max(1, psutil.cpu_count() - 1)

        else:
            raise AssertionError('n_cpus can be an integer or \'auto\' only!')

        assert isinstance(extend_steps, int)
        assert extend_steps >= 0, 'extend_steps can not be less than zero!'

        self._n_cpus = n_cpus
        self._ext_steps = extend_steps

        if self._vb:
            print_sl()

            print(f'INFO: Set the following misc. settings:')

            print('\t', f'Number of parallel processes: {self._n_cpus}')

            print('\t', f'Extend each simulation to {self._ext_steps} steps')

            print_el()
        return

    def verify(self):

        assert self._set_data_flag, 'Data not set!'
        assert self._set_out_dir_flag, 'Outputs directory not set!'
        assert self._h5_path_set_flag, 'Output HDF5 path not set!'
        assert self._set_excd_probs_flag, 'Exceedance probabilities not set!'
        assert self._set_tws_flag, 'Time windows not set!'
        assert self._set_n_sims_flag, 'Number of simulations not set!'

        if self._vb:
            print_sl()

            print('INFO: All data verified to be correct!')

            print_el()

        self._set_data_verify_flag = True
        return

    __verify = verify
