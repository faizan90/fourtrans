'''
@author: Faizan-Uni-Stuttgart

'''
import os
import timeit
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fourtrans import SimultaneousExtremes, SimultaneousExtremesPlot


def main():

    main_dir = Path(r'P:\Synchronize\IWS\QGIS_Neckar\hydmod\input_hyd_data')
    os.chdir(main_dir)

    in_file = r'neckar_daily_discharge_1961_2015.csv'

    clusters_shp = r'P:\Synchronize\IWS\QGIS_Neckar\raster\taudem_out_spate_rockenau\watersheds.shp'
    clusters_shp_fld = 'DN'

    out_dir = 'test_simultexts_tfm_01'

#     stns = ['420', '427', '454']  # , '3470', '3465', '3421'
#     excd_probs = [0.001, 0.005]  # [0.001, 0.005, 0.0001, 0.0005, 0.00001]  #
#     time_windows = [1, 3, 0]  # [2, 10, 1, 3, 0]  #

#     stns = ['420', '427', '454', '3470', '3465', '3421']
#     excd_probs = [0.001, 0.005, 0.0001, 0.0005]  # , 0.00001]
#     time_windows = [2, 10, 1, 3, 0]

    stns = ['427', '454', '420']
    excd_probs = [0.0001, 0.0005]  # , 0.00001]
    time_windows = [2, 1, 0, ]
    tfm_type = 'prob'

    n_sims = 100

    n_cpus = 'auto'

    n_steps_extend = 0  # int(3e4)

    verbose_flag = False
    overwrite_flag = False
    cmpt_simultexts_flag = False
    save_sim_cdfs_flag = False
    save_sim_corrs_flag = False
    save_sim_ft_cumm_corrs_flag = False
    plot_freqs_flag = False
    plot_clusters_flag = False
    plot_sim_cdfs_flag = False
    plot_sim_auto_corrs_flag = False
    plot_sim_ft_corrs_flag = False
    plot_sim_ft_pair_corrs_dist_flag = False

    verbose_flag = True
    overwrite_flag = True
    cmpt_simultexts_flag = True
    save_sim_cdfs_flag = True
    save_sim_corrs_flag = True
    save_sim_ft_cumm_corrs_flag = True
    plot_freqs_flag = True
    plot_clusters_flag = True
    plot_sim_cdfs_flag = True
    plot_sim_auto_corrs_flag = True
    plot_sim_ft_corrs_flag = True
    plot_sim_ft_pair_corrs_dist_flag = True

    in_df = pd.read_csv(in_file, sep=';', index_col=0)

    in_df.index = pd.to_datetime(in_df.index, format='%Y-%m-%d')

    in_df = in_df.loc[:, stns]

    if cmpt_simultexts_flag:
        SE = SimultaneousExtremes(verbose_flag, overwrite_flag)

        SE.set_data(in_df)

        SE.set_outputs_directory(out_dir)

        SE.set_exceedance_probabilities(np.array(excd_probs))

        SE.set_time_windows(np.array(time_windows))

        SE.set_number_of_simulations(n_sims)

        SE.set_tfm_type(tfm_type)

        SE.set_additonal_analysis_flags(
            save_sim_cdfs_flag,
            save_sim_corrs_flag,
            save_sim_ft_cumm_corrs_flag)

        SE.set_misc_settings(n_cpus, n_steps_extend)

        SE.verify()

        SE.cmpt_simult_exts_freqs()

        SE = None

    if any([
        plot_freqs_flag,
        plot_clusters_flag,
        plot_sim_cdfs_flag,
        plot_sim_auto_corrs_flag,
        plot_sim_ft_corrs_flag,
        plot_sim_ft_pair_corrs_dist_flag]):

        SEP = SimultaneousExtremesPlot(verbose_flag)

        SEP.set_outputs_directory(out_dir)

        SEP.set_misc_settings(n_cpus)

        SEP.set_plot_type_flags(
            plot_freqs_flag,
            plot_sim_cdfs_flag,
            plot_sim_auto_corrs_flag,
            plot_sim_ft_corrs_flag,
            plot_sim_ft_pair_corrs_dist_flag)

        if plot_clusters_flag:
            SEP.set_clusters_shapefile_path(clusters_shp, clusters_shp_fld)

        SEP.verify()

        SEP.plot()

        SEP = None
    return


if __name__ == '__main__':

    _save_log_ = False
    if _save_log_:
        from datetime import datetime
        from std_logger import StdFileLoggerCtrl

        # save all console activity to out_log_file
        out_log_file = os.path.join(
            r'P:\Synchronize\python_script_logs\\%s_log_%s.log' % (
            os.path.basename(__file__),
            datetime.now().strftime('%Y%m%d%H%M%S')))

        log_link = StdFileLoggerCtrl(out_log_file)

    print('#### Started on %s ####\n' % time.asctime())
    START = timeit.default_timer()  # to get the runtime of the program

    main()

    STOP = timeit.default_timer()  # Ending time
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))

    if _save_log_:
        log_link.stop()
