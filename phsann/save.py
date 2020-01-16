'''
@author: Faizan-Uni

Jan 16, 2020

9:39:36 AM
'''
from pathlib import Path
from fnmatch import fnmatch

import h5py
import numpy as np
#
from ..simultexts.misc import print_sl, print_el

from .algorithm import PhaseAnnealingAlgorithm as PAA


class PhaseAnnealingSave(PAA):

    '''
    Save reference, realizations flags, settings to HDF5
    '''

    def __init__(self, verbose):

        PAA.__init__(self, verbose)

        self._save_h5_name = 'phsann.h5'

        self._save_verify_flag = True
        return

    def _get_flags(self):

        flags = [('_vb', self._vb)]
        for var in vars(self):
            if not fnmatch(var, '*_flag'):
                continue

            flag = getattr(self, var)

            if not isinstance(flag, bool):
                continue

            flags.append((var, flag))

        assert flags, 'No flags selected!'

        return flags

    def _write_flags(self, h5_hdl):

        flags = self._get_flags()

        flags_grp = h5_hdl.create_group('flags')

        for flag_lab, flag_val in flags:
            flags_grp.attrs[flag_lab] = flag_val

        h5_hdl.flush()
        return

    def _get_ref_data(self):

        datas = []
        for var in vars(self):
            if not fnmatch(var, '_data_*'):
                continue

            datas.append((var, getattr(self, var)))

        return datas

    def _write_ref_data(self, h5_hdl):

        datas = self._get_ref_data()

        datas_grp = h5_hdl.create_group('data_ref')

        for data_lab, data_val in datas:
            if isinstance(data_val, np.ndarray):
                datas_grp[data_lab] = data_val

            else:
                datas_grp.attrs[data_lab] = data_val

        h5_hdl.flush()
        return

    def _get_settings(self):
        setts = []
        for var in vars(self):
            if not fnmatch(var, '_sett_*'):
                continue

            setts.append((var, getattr(self, var)))

        return setts

    def _write_settings(self, h5_hdl):

        setts = self._get_settings()

        setts_grp = h5_hdl.create_group('settings')

        for sett_lab, sett_val in setts:
            if isinstance(sett_val, np.ndarray):
                setts_grp[sett_lab] = sett_val

            elif sett_val is None:
                setts_grp.attrs[sett_lab] = str(sett_val)

            elif isinstance(sett_val, Path):
                setts_grp.attrs[sett_lab] = str(sett_val)

            else:
                setts_grp.attrs[sett_lab] = sett_val

        h5_hdl.flush()
        return

    def _get_ref_rltzn_data(self):

        datas = []
        for var in vars(self):
            if not fnmatch(var, '_ref_*'):
                continue

            datas.append((var, getattr(self, var)))

        return datas

    def _write_ref_rltzn_data(self, h5_hdl):

        datas = self._get_ref_rltzn_data()

        datas_grp = h5_hdl.create_group('data_ref_rltzn')

        for data_lab, data_val in datas:
            if isinstance(data_val, np.ndarray):
                datas_grp[data_lab] = data_val

            else:
                datas_grp.attrs[data_lab] = data_val

        h5_hdl.flush()
        return

    def _get_sim_rltzns_data(self):

        sims = []
        for i in range(self._sett_misc_n_rltzns):
            sim = self._alg_rltzns[i]

            sims.append((i, sim))

        return sims

    def _write_sim_rltzns_data(self, h5_hdl):

        sims = self._get_sim_rltzns_data()

        sim_grp_main = h5_hdl.create_group('data_sim_rltzns')

        pad_zeros = len(str(self._sett_misc_n_rltzns))

        for i, sim in sims:
            sim_grp = sim_grp_main.create_group(f'{i:0{pad_zeros}d}')

            for sim_lab, sim_val in sim._asdict().items():
                if isinstance(sim_val, np.ndarray):
                    sim_grp[sim_lab] = sim_val

                else:
                    sim_grp.attrs[sim_lab] = sim_val

        h5_hdl.flush()
        return

    def _get_prep_data(self):

        datas = []
        for var in vars(self):
            if not fnmatch(var, '_prep_*'):
                continue

            datas.append((var, getattr(self, var)))

        return datas

    def _write_prep_data(self, h5_hdl):

        datas = self._get_prep_data()

        datas_grp = h5_hdl.create_group('prep')

        for data_lab, data_val in datas:
            if isinstance(data_val, np.ndarray):
                datas_grp[data_lab] = data_val

            else:
                datas_grp.attrs[data_lab] = data_val

        h5_hdl.flush()
        return

    def save_realizations(self):

        if self._vb:
            print_sl()

            print('Saving realizations...')

        assert self._alg_rltzns_gen_flag, 'Call generate_realizations first!'

        if not self._sett_misc_outs_dir.exists():
            self._sett_misc_outs_dir.mkdir(exist_ok=True)

        assert self._sett_misc_outs_dir.exists(), (
            'Could not create outputs_dir!')

        h5_hdl = h5py.File(
            self._sett_misc_outs_dir / self._save_h5_name,
            mode='w',
            driver=None)

        self._write_flags(h5_hdl)

        self._write_ref_data(h5_hdl)

        self._write_settings(h5_hdl)

        self._write_ref_rltzn_data(h5_hdl)

        self._write_sim_rltzns_data(h5_hdl)

        self._write_prep_data(h5_hdl)

        h5_hdl.close()

        if self._vb:
            print('Done saving.')

            print_el()

        return

    def verify(self):

        PAA._PhaseAnnealingAlgorithm__verify(self)
        assert self._alg_verify_flag, 'Algorithm in an unverified state!'

        self._save_verify_flag = True
        return

    __verify = verify
