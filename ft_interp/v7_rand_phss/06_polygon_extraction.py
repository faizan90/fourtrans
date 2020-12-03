'''
@author: Faizan-Uni-Stuttgart

Nov 12, 2020

7:00:48 PM

'''
import os
import time
import timeit
from pathlib import Path

from spinterps import Extract

DEBUG_FLAG = True


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Testings\fourtrans_practice\multisite_phs_spec_corr\5min\v7_rand_phss')
    os.chdir(main_dir)

    path_to_shp = r'P:\Synchronize\IWS\QGIS_Neckar\raster\taudem_out_spate_rockenau\watersheds_all.shp'

    label_field = r'DN'

    var = 'ifted'
    suff = '__RR1D_RTsum'

    path_to_ras = f'{var}/{var}{suff}.nc'

    nc_x_crds_label = 'X'
    nc_y_crds_label = 'Y'
    nc_variable_labels = ['OK']
    nc_time_label = 'time'

    overwrite_flag = True

    path_to_output = Path(f'{var}/{var}{suff}.h5')

    if overwrite_flag and path_to_output.exists():
        os.remove(path_to_output)

    Ext = Extract(True)

    res = Ext.extract_from_netCDF(
        path_to_shp,
        label_field,
        path_to_ras,
        path_to_output,
        nc_x_crds_label,
        nc_y_crds_label,
        nc_variable_labels,
        nc_time_label)

    print('res:', res)

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
    START = timeit.default_timer()

    #==========================================================================
    # When in post_mortem:
    # 1. "where" to show the stack
    # 2. "up" move the stack up to an older frame
    # 3. "down" move the stack down to a newer frame
    # 4. "interact" start an interactive interpreter
    #==========================================================================

    if DEBUG_FLAG:
        try:
            main()

        except:
            import pdb
            pdb.post_mortem()

    else:
        main()

    STOP = timeit.default_timer()
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))

    if _save_log_:
        log_link.stop()
