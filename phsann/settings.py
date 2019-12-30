'''
Created on Dec 27, 2019

@author: Faizan
'''
from pathlib import Path

import numpy as np

from ..simultexts.misc import print_sl, print_el

from .data import PhaseAnnealingData as PAD


class PhaseAnnealingSettings(PAD):

    def __init__(self, verbose=True):

        PAD.__init__(self, verbose)

        self._sett_obj_scorr_flag = None
        self._sett_obj_symm_type_1_flag = None
        self._sett_obj_symm_type_2_flag = None
        self._sett_obj_lag_steps = None

        self._sett_ann_init_temp = None
        self._sett_ann_temp_red_ratio = None
        self._sett_ann_upt_evry_iter = None
        self._sett_ann_max_iters = None
        self._sett_ann_max_iter_wo_chng = None

        self._sett_misc_nreals = 1

        self._sett_obj_set_flag = False
        self._sett_ann_set_flag = False

        self._sett_verify_flag = False
        return

    def set_objective_settings(
            self,
            scorr_flag,
            symm_type_1_flag,
            symm_type_2_flag,
            lag_steps):

        assert isinstance(scorr_flag, bool)
        assert isinstance(symm_type_1_flag, bool)
        assert isinstance(symm_type_2_flag, bool)

        assert any(
            [scorr_flag, symm_type_1_flag, symm_type_2_flag])

        assert isinstance(lag_steps, int)
        assert lag_steps > 0

        self._sett_obj_scorr_flag = scorr_flag
        self._sett_obj_symm_type_1_flag = symm_type_1_flag
        self._sett_obj_symm_type_2_flag = symm_type_2_flag

        self._sett_obj_lag_steps = lag_steps

        if self._vb:
            print_sl()

            print('Set the following objective function flags:')

            print('Rank correlation flag:', self._sett_obj_scorr_flag)
            print('Symmetry type 1 flag:', self._sett_obj_symm_type_1_flag)
            print('Symmetry type 2 flag:', self._sett_obj_symm_type_2_flag)
            print(f'Lag steps:', self._sett_obj_lag_steps)

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

        assert isinstance(initial_annealing_temperature, float)
        assert isinstance(temperature_reduction_ratio, float)
        assert isinstance(update_at_every_iteration_no, int)
        assert isinstance(maximum_iterations, int)
        assert isinstance(maximum_without_change_iterations, int)
        assert isinstance(objective_tolerance, float)
        assert isinstance(objective_tolerance_iterations, int)

        assert 0 < initial_annealing_temperature < np.inf

        assert 0 < temperature_reduction_ratio < 1

        assert (
            0 <
            objective_tolerance_iterations <=
            update_at_every_iteration_no <=
            maximum_without_change_iterations <=
            maximum_iterations)

        assert 0 < objective_tolerance < np.inf

        self._sett_ann_init_temp = initial_annealing_temperature
        self._sett_ann_temp_red_ratio = temperature_reduction_ratio
        self._sett_ann_upt_evry_iter = update_at_every_iteration_no
        self._sett_ann_max_iters = maximum_iterations
        self._sett_ann_max_iter_wo_chng = maximum_without_change_iterations
        self._sett_ann_obj_tol = objective_tolerance
        self._sett_ann_obj_tol_iters = objective_tolerance_iterations

        if self._vb:
            print_sl()

            print(
                'Set the following simulated annealing algorithm parameters:')

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

    def set_misc_settings(self, n_reals, outputs_dir):

        assert isinstance(n_reals, int)
        assert 0 < n_reals

        outputs_dir = Path(outputs_dir)

        assert outputs_dir.is_absolute()

        assert outputs_dir.parents[0].exists()

        if not outputs_dir.exists:
            outputs_dir.mkdir(exist_ok=True)

        self._sett_misc_nreals = n_reals
        self._sett_misc_outs_dir = outputs_dir

        if self._vb:
            print_sl()

            print('Set the following misc. variables:')

            print('Number of realizations:', self._sett_misc_nreals)

            print('Outputs directory:', self._sett_misc_outs_dir)

            print_el()

        return

    def verify(self):

        PAD.PhaseAnnealing__verify(self)
        assert self._data_verify_flag

        assert self._sett_obj_set_flag
        assert self._sett_ann_set_flag

        if self._sett_obj_scorr_flag:
            assert self._sett_obj_lag_steps < self._data_ref_shape[0]

        if self._data_ref_data.ndim != 1:
            raise NotImplementedError('Algorithm meant for 1D only!')

        if self._vb:
            print_sl()

            print(f'Phase annealing settings verified successfully!')

            print_el()

        self._sett_verify_flag = True
        return

    __verify = verify
