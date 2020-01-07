'''
Created on Dec 27, 2019

@author: Faizan
'''
import psutil
from pathlib import Path

import numpy as np

from ..simultexts.misc import print_sl, print_el

from .data import PhaseAnnealingData as PAD


class PhaseAnnealingSettings(PAD):

    def __init__(self, verbose=True):

        PAD.__init__(self, verbose)

        self._sett_obj_scorr_flag = None
        self._sett_obj_asymm_type_1_flag = None
        self._sett_obj_asymm_type_2_flag = None
        self._sett_obj_ecop_dens_flag = None
        self._sett_obj_lag_steps = None
        self._sett_obj_ecop_dens_bins = None

        self._sett_ann_init_temp = None
        self._sett_ann_temp_red_ratio = None
        self._sett_ann_upt_evry_iter = None
        self._sett_ann_max_iters = None
        self._sett_ann_max_iter_wo_chng = None

        self._sett_ann_auto_init_temp_temp_bd_lo = None
        self._sett_ann_auto_init_temp_temp_bd_hi = None
        self._sett_ann_auto_init_temp_atpts = None
        self._sett_ann_auto_init_temp_niters = None
        self._sett_ann_auto_init_temp_acpt_bd_lo = None
        self._sett_ann_auto_init_temp_acpt_bd_hi = None
        self._sett_ann_auto_init_temp_ramp_rate = None
        self._sett_ann_auto_init_temp_trgt_acpt_rate = None

        self._sett_misc_nreals = 1
        self._sett_misc_ncpus = 1

        self._sett_obj_set_flag = False
        self._sett_ann_set_flag = False
        self._sett_auto_temp_set_flag = False
        self._sett_misc_set_flag = False

        self._sett_verify_flag = False
        return

    def set_objective_settings(
            self,
            scorr_flag,
            asymm_type_1_flag,
            asymm_type_2_flag,
            ecop_dens_flag,
            lag_steps,
            ecop_dens_bins=None):

        if self._vb:
            print_sl()

            print(
                'Setting objective function settings for phase annealing...\n')

        assert isinstance(scorr_flag, bool), 'scorr_flag not a boolean!'

        assert isinstance(asymm_type_1_flag, bool), (
            'asymm_type_1_flag not a boolean!')

        assert isinstance(asymm_type_2_flag, bool), (
            'asymm_type_2_flag not a boolean!')

        assert isinstance(ecop_dens_flag, bool), (
            'ecop_dens_flag not a boolean!')

        assert any([
            scorr_flag,
            asymm_type_1_flag,
            asymm_type_2_flag,
            ecop_dens_flag,
            ]), 'All objective function flags are False!'

        assert isinstance(lag_steps, np.ndarray), (
            'lag_steps not a numpy arrray!')

        assert lag_steps.ndim == 1, 'lag_steps not a 1D array!'
        assert lag_steps.size > 0, 'lag_steps is empty!'
        assert lag_steps.dtype == np.int, 'lag_steps has a non-integer dtype!'
        assert np.all(lag_steps > 0), 'lag_steps has non-postive values!'

        assert np.unique(lag_steps).size == lag_steps.size, (
            'Non-unique values in lag_steps!')

        if ecop_dens_flag:
            assert isinstance(ecop_dens_bins, int), (
                'ecop_dens_bins is not an integer!')

            assert ecop_dens_bins > 1, 'Invalid ecop_dens_bins!'

        self._sett_obj_scorr_flag = scorr_flag
        self._sett_obj_asymm_type_1_flag = asymm_type_1_flag
        self._sett_obj_asymm_type_2_flag = asymm_type_2_flag
        self._sett_obj_ecop_dens_flag = ecop_dens_flag

        self._sett_obj_lag_steps = lag_steps

        if ecop_dens_flag:
            self._sett_obj_ecop_dens_bins = ecop_dens_bins

        if self._vb:
            print(
                'Rank correlation flag:',
                self._sett_obj_scorr_flag)

            print(
                'Asymmetry type 1 flag:',
                self._sett_obj_asymm_type_1_flag)

            print(
                'Asymmetry type 2 flag:',
                self._sett_obj_asymm_type_2_flag)

            print(
                'Empirical copula density flag:',
                self._sett_obj_ecop_dens_flag)

            print(
                'Lag steps:',
                self._sett_obj_lag_steps)

            print(
                'Empirical copula density bins:',
                self._sett_obj_ecop_dens_bins)

            print_el()

        self._sett_obj_set_flag = True
        return

    def set_annealing_settings(
            self,
            initial_annealing_temperature,
            temperature_reduction_ratio,
            update_at_every_iteration_no,
            maximum_iterations,
            maximum_without_change_iterations,
            objective_tolerance,
            objective_tolerance_iterations):

        if self._vb:
            print_sl()

            print('Setting annealing settings for phase annealing...\n')

        assert isinstance(initial_annealing_temperature, float), (
            'initial_annealing_temperature not a float!')

        assert isinstance(temperature_reduction_ratio, float), (
            'temperature_reduction_ratio not a float!')

        assert isinstance(update_at_every_iteration_no, int), (
            'update_at_every_iteration_no not an integer!')

        assert isinstance(maximum_iterations, int), (
            'maximum_iterations not an integer!')

        assert isinstance(maximum_without_change_iterations, int), (
            'maximum_without_change_iterations not an integer!')

        assert isinstance(objective_tolerance, float), (
            'objective_tolerance not a float!')

        assert isinstance(objective_tolerance_iterations, int), (
            'objective_tolerance_iterations not an integer!')

        assert 0 < initial_annealing_temperature < np.inf, (
            'Invalid initial_annealing_temperature!')

        assert 0 < temperature_reduction_ratio <= 1, (
            'Invalid temperature_reduction_ratio!')

        assert (
            0 <
            objective_tolerance_iterations <=
            update_at_every_iteration_no <=
            maximum_without_change_iterations <=
            maximum_iterations), (
                'Inconsistent or invalid objective_tolerance_iterations, '
                'update_at_every_iteration_no, '
                'maximum_without_change_iterations, maximum_iterations'
                'values!')

        assert 0 <= objective_tolerance < np.inf, (
            'Invalid objective_tolerance!')

        self._sett_ann_init_temp = initial_annealing_temperature
        self._sett_ann_temp_red_ratio = temperature_reduction_ratio
        self._sett_ann_upt_evry_iter = update_at_every_iteration_no
        self._sett_ann_max_iters = maximum_iterations
        self._sett_ann_max_iter_wo_chng = maximum_without_change_iterations
        self._sett_ann_obj_tol = objective_tolerance
        self._sett_ann_obj_tol_iters = objective_tolerance_iterations

        if self._vb:

            print(
                'Initial annealing temperature:', self._sett_ann_init_temp)

            print(
                'Temperature reduction ratio:', self._sett_ann_temp_red_ratio)

            print(
                'Temperature update iteration:', self._sett_ann_upt_evry_iter)

            print(
                'Maximum iterations:', self._sett_ann_max_iters)

            print(
                'Maximum iterations without change:',
                self._sett_ann_max_iter_wo_chng)

            print(
                'Objective function tolerance:',
                self._sett_ann_obj_tol)

            print(
                'Objective function tolerance iterations:',
                self._sett_ann_obj_tol_iters)

            print_el()

        self._sett_ann_set_flag = True
        return

    def set_annealing_auto_temperature_settings(
            self,
            temperature_lower_bound,
            temperature_upper_bound,
            max_search_attempts,
            n_iterations_per_attempt,
            acceptance_lower_bound,
            acceptance_upper_bound,
            ramp_rate,
            target_acceptance_rate):

        if self._vb:
            print_sl()

            print(
                'Setting automatic annealing initial temperature settings '
                'for phase annealing...\n')

        assert isinstance(temperature_lower_bound, float), (
            'temperature_lower_bound not a float!')

        assert isinstance(temperature_upper_bound, float), (
            'temperature_upper_bound not a float!')

        assert isinstance(max_search_attempts, int), (
            'max_search_attempts not an integer!')

        assert isinstance(n_iterations_per_attempt, int), (
            'n_iterations_per_attempt not an integer!')

        assert isinstance(acceptance_lower_bound, float), (
            'acceptance_lower_bound not a float!')

        assert isinstance(acceptance_upper_bound, float), (
            'acceptance_upper_bound not a float!')

        assert isinstance(ramp_rate, float), (
            'ramp_rate not a float!')

        assert isinstance(target_acceptance_rate, float), (
            'target_acceptance_rate not a float!')

        assert (
            0 < temperature_lower_bound < temperature_upper_bound < np.inf), (
                'Inconsistent or invalid temperature_lower_bound and '
                'temperature_upper_bound!')

        assert 0 < max_search_attempts, 'Invalid max_search_attempts!'

        assert 0 < n_iterations_per_attempt, (
            'Invalid n_iterations_per_attempt!')

        assert (
            0 <
            acceptance_lower_bound <=
            target_acceptance_rate <=
            acceptance_upper_bound <
            1.0), (
                'Invalid or inconsistent acceptance_lower_bound, '
                'target_acceptance_rate or acceptance_upper_bound!')

        assert 0 < ramp_rate < np.inf, 'Invalid ramp_rate!'

        self._sett_ann_auto_init_temp_temp_bd_lo = temperature_lower_bound
        self._sett_ann_auto_init_temp_temp_bd_hi = temperature_upper_bound
        self._sett_ann_auto_init_temp_atpts = max_search_attempts
        self._sett_ann_auto_init_temp_niters = n_iterations_per_attempt
        self._sett_ann_auto_init_temp_acpt_bd_lo = acceptance_lower_bound
        self._sett_ann_auto_init_temp_acpt_bd_hi = acceptance_upper_bound
        self._sett_ann_auto_init_temp_ramp_rate = ramp_rate
        self._sett_ann_auto_init_temp_trgt_acpt_rate = target_acceptance_rate

        if self._vb:
            print(
                'Lower tmeperature bounds:',
                self._sett_ann_auto_init_temp_temp_bd_lo)

            print(
                'Upper temperature bounds:',
                self._sett_ann_auto_init_temp_temp_bd_hi)

            print(
                'Maximum temperature search attempts:',
                self._sett_ann_auto_init_temp_atpts)

            print(
                'Number of iterations per attempt:',
                self._sett_ann_auto_init_temp_niters)

            print(
                'Lower acceptance bounds:',
                self._sett_ann_auto_init_temp_acpt_bd_lo)

            print(
                'Upper acceptance bounds:',
                self._sett_ann_auto_init_temp_acpt_bd_hi)

            print(
                'Temperature ramp rate:',
                self._sett_ann_auto_init_temp_ramp_rate
                )

            print(
                'Target acceptance rate:',
                self._sett_ann_auto_init_temp_trgt_acpt_rate)

            print_el()

        self._sett_auto_temp_set_flag = True
        return

    def set_misc_settings(self, n_reals, outputs_dir, n_cpus=1):

        if self._vb:
            print_sl()

            print('Setting misc. settings for phase annealing...\n')

        assert isinstance(n_reals, int), 'n_reals not an integer!'
        assert 0 < n_reals, 'Invalid n_reals!'

        outputs_dir = Path(outputs_dir)

        assert outputs_dir.is_absolute()

        assert outputs_dir.parents[0].exists(), (
            'Parent directory of outputs dir does not exist!')

        if not outputs_dir.exists:
            outputs_dir.mkdir(exist_ok=True)

        if isinstance(n_cpus, str):
            assert n_cpus == 'auto', 'Invalid n_cpus!'

            n_cpus = max(1, psutil.cpu_count() - 1)

        else:
            assert isinstance(n_cpus, int), 'n_cpus is not an integer!'

            assert n_cpus > 0, 'Invalid n_cpus!'

        if n_reals < n_cpus:
            n_cpus = n_reals

        self._sett_misc_nreals = n_reals
        self._sett_misc_outs_dir = outputs_dir
        self._sett_misc_ncpus = n_cpus

        if self._vb:
            print('Number of realizations:', self._sett_misc_nreals)

            print('Outputs directory:', self._sett_misc_outs_dir)

            print(
                'Number of maximum process(es) to use:', self._sett_misc_ncpus)

            print_el()

        self._sett_misc_set_flag = True
        return

    def verify(self):

        PAD._PhaseAnnealingData__verify(self)
        assert self._data_verify_flag, 'Data in an unverified state!'

        assert self._sett_obj_set_flag, 'Call set_objective_settings first!'
        assert self._sett_ann_set_flag, 'Call set_annealing_settings first!'
        assert self._sett_misc_set_flag, 'Call set_misc_settings first!'

        if self._data_ref_data.ndim != 1:
            raise NotImplementedError('Algorithm meant for 1D only!')

        if self._sett_obj_scorr_flag:
            assert np.all(
                self._sett_obj_lag_steps < self._data_ref_shape[0]), (
                    'At least one of the lag_steps is >= the size '
                    'of ref_data!')

        if self._sett_obj_ecop_dens_flag:
            assert self._sett_obj_ecop_dens_bins <= self._data_ref_shape[0], (
               'ecop_dens_bins have to be <= size of ref_data!')

        if self._sett_auto_temp_set_flag:
            self._sett_ann_init_temp = None

            if self._vb:
                print_sl()

                print(
                    'Unset Phase annealing initial temperature to None due '
                    'to auto search!')

                print_el()

        if self._vb:
            print_sl()

            print(f'Phase annealing settings verified successfully!')

            print_el()

        self._sett_verify_flag = True
        return

    __verify = verify
