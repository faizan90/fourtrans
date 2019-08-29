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

    '''Set the inputs for the computation of probabilities of
    simultaneous extremes occurences using the Fourier transform.
    This class is inheritied by SimultaneousExtremesAlgorithm.
    '''

    def __init__(self, verbose=True):

        self._vb = bool(verbose)

        self._n_cpus = 1
        self._ext_steps = 0

        self._tfm_types = ('obs', 'prob', 'norm', 'prob__no_ann_cyc')

        self._save_sim_cdfs_flag = False
        self._save_sim_acorrs_flag = False
        self._save_sim_ft_cumm_corrs_flag = False

        self._set_data_flag = False
        self._set_out_dir_flag = False
        self._set_excd_probs_flag = False
        self._set_tws_flag = False
        self._set_n_sims_flag = False
        self._set_tfm_type_flag = False

        self._set_data_verify_flag = False
        return

    def set_data(self, time_ser_df):

        '''
        Parameters
        ----------
        time_ser_df : pd.DataFrame
            A DataFrame holding the data of each column (station) as a
            time series. The index should be of the type pd.DatetimeIndex.
            The values should be a subclass of np.floating. The columns
            are cast to strings upon setup. Values may be NaN but not all
            of them for a given station.
        '''

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

        '''
        Parameters
        ----------
        out_dir : str or pathlib.Path
            The directory in which to save all the simulation outputs. It is
            created if non-existent. All outputs from the simulations are
            saved to an HDF5 file "simultexts_db.hdf5" inside this directory.
        '''

        assert isinstance(out_dir, (Path, str)), (
            'out_dir not a string or a Path-like object!')

        out_dir = Path(out_dir).absolute()

        assert out_dir.parents[0].exists(), (
            'Parent directory of the out_dir does not exist!')

        self._out_dir = out_dir

        self._h5_path = self._out_dir / r'simultexts_db.hdf5'

        if self._vb:
            print_sl()

            print('INFO: Set the outputs directory as following:')
            print('\t', f'{str(self._out_dir)}')

            print_el()

        self._set_out_dir_flag = True
        return

    def set_exceedance_probabilities(self, exceedance_probabilities):

        '''
        Parameters
        ----------
        exceedance_probabilities : subtype of numpy.floating dtype 1D array
            A Numpy array holding the exceedance probabilites of events
            for which to compute the simulated probabilites of simultaneous
            occurrences of events. The values range between 0 and 1. The
            smaller the value the lower the chance of it occurring.
            Sorted array of values is used upon setup.
        '''

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

        assert (
            np.unique(exceedance_probabilities).shape[0] ==
            exceedance_probabilities.shape[0]), (
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

        '''
        Parameters
        ----------
        time_windows : subtype of numpy.integer dtype 1D array
            An array with window sizes that represent the step size such
            that events of a given exceedance probability occurring in
            plus-minus this many steps for a given combination of stations
            are considered simultaneous. e.g. time window of zero would
            mean all the events in a given combination have to occur on
            the same step to be considered as simultaneous. A window size
            of 2 means that the difference of steps between all the stations
            in a given combination is not more than 2 steps (5
            steps in total, including step 0).
        '''

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

        '''
        Parameters
        ----------
        n_sims : int
            The number of simulations to perform. First simulation is always
            the observed data. The rest are n_sims.
        '''

        assert isinstance(n_sims, int), 'n_sims not an integer!'

#         assert n_sims > 0, 'n_sims should be greater than zero!'

        self._n_sims = n_sims

        if self._vb:
            print_sl()

            print(f'INFO: Set the number of simulations to {self._n_sims}')

            print_el()

        self._set_n_sims_flag = True
        return

    def set_tfm_type(self, tfm_type):

        '''Set the type of transformation applied to the input data before
        it is Fouriered.

        Parameters
        ----------
        tfm_type : string
            A string identifier for the transformation type.
            obs : No transformation i.e. use original data.
            prob : Use CDF values for each column
            norm : Use standard normal values for each column
            prob__no_ann_cyc : Remove annual cycle from each column and then
            use the CDF values. The annual cycle CDF is added to the Fouriered
            values before reshuffling the simulated series.
        '''

        assert isinstance(tfm_type, str), 'tfm_type not a string object!'

        assert tfm_type in self._tfm_types, (
            f'Given tfm_type: {tfm_type} does not exist in defined tfm_type: '
            f'{self._tfm_types}')

        self._tfm_type = tfm_type

        if self._vb:
            print_sl()

            print(
                f'INFO: Set the data transformation type to {self._tfm_type}')

            print_el()

        self._set_tfm_type_flag = True
        return

    def set_additonal_analysis_flags(
            self,
            sim_cdfs_flag,
            sim_auto_corrs_flag,
            sim_ft_cumm_corrs_flag):

        '''
        Parameters
        ----------
        sim_cdfs_flag : bool
            To save the cummulative distribution of each simulated series.
            This can later be used for verification of the algorithm.
        sim_auto_corrs_flag : bool
            To save the auto correlations of observed and simulated series.
            This can be used to see how close are the simulated data to the
            observed.
        sim_ft_cumm_corrs_flag : bool
            To save cummulative correlation contribution of each fourier
            wave to the observed and simulated series. This can be used to
            see how close are the observed and simulated series.

        The plotting class can plot this data if it is saved in the database
        and the plotting flags are set to True before plotting.
        '''

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

        '''
        Parameters
        ----------
        n_cpus : int
            The number of maximum running processes to use while simulating.
            Default is 1. If set to "auto" the maximum number of running
            processes is set to the number of logical cores minus one.
        extend_steps : int
            Extend each simulated series to this many steps. The final
            length depends on the length of the series in the inputs. The
            length can only be a multiple of the original length of the input.
            If extend_steps is less the original length then the original is
            used. If it is higher than the original, then the extended steps
            are such that they are a multiple of the original and are
            greater than or equal to extend_steps.
        '''

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

        '''
        Verify if all the required inputs are set. The algorithm won't
        proceed if verify hasn't been called and all the required inputs
        aren't set.
        '''

        assert self._set_data_flag, 'Data not set!'
        assert self._set_out_dir_flag, 'Outputs directory not set!'
        assert self._set_excd_probs_flag, 'Exceedance probabilities not set!'
        assert self._set_tws_flag, 'Time windows not set!'
        assert self._set_n_sims_flag, 'Number of simulations not set!'
        assert self._set_tfm_type_flag, 'Transformation type not set!'

        if self._vb:
            print_sl()

            print('INFO: All data verified to be correct!')

            print_el()

        self._set_data_verify_flag = True
        return

    __verify = verify
