# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=False
# cython: embedsignature=True

from libc.math cimport cos, atan2


cdef extern from "intel_ft.h" nogil:
    cdef:
        void mkl_real_dft(
            DT_D *in_reals_arr,
            DT_DC *out_comps_arr,
            long n_pts,
            int ft_type)


cdef extern from "complex.h" nogil:
    cdef:
        DT_D creal(DT_DC)
        DT_D cimag(DT_DC)


cdef void cmpt_real_forw_ft_1d(FT1DReal *ft_struct) nogil except +:

    cdef:
        Py_ssize_t i

        DT_UL n_pts = ft_struct.n_pts

        DT_DC ft, ft_imag, ft_real

        DT_D *amps
        DT_D *angs

    amps = ft_struct.amps
    angs = ft_struct.angs

    mkl_real_dft(ft_struct.orig, ft_struct.ft, n_pts, 1)

    for i in range((n_pts // 2) - 1):
        ft = ft_struct.ft[i + 1]

        ft_imag = cimag(ft)
        ft_real = creal(ft)

        angs[i] = atan2(ft_imag, ft_real)

        amps[i] = ((ft_real**2) + (ft_imag**2))**0.5
    return


cpdef void cmpt_real_forw_ft_1d_cy(
        DT_D[::1] orig, 
        DT_DC[::1] ft,
        DT_D[::1] amps,
        DT_D[::1] angs) nogil except +:

    cdef:
        FT1DReal ft_struct

    ft_struct.orig = &orig[0]
    ft_struct.ft = &ft[0]
    ft_struct.amps = &amps[0]
    ft_struct.angs = &angs[0]
    ft_struct.n_pts = orig.shape[0]

    cmpt_real_fourtrans_1d(&ft_struct)
    return


# cdef void cmpt_cumm_freq_pcorrs(
#         const ForFourTrans1DReal *obs_for_four_trans_struct,
#         const ForFourTrans1DReal *sim_for_four_trans_struct,
#               DT_D *freq_corrs) nogil except +:
# 
#     cdef:
#         Py_ssize_t i
# 
#         DT_UL n_pts = obs_for_four_trans_struct.n_pts
# 
#         DT_D tot_cov = 0
#         DT_D obs_amps_sq_sum = 0, sim_amps_sq_sum = 0
#         DT_D pcorr, freq_cov_scale
# 
#         DT_D *obs_amps
#         DT_D *sim_amps
#         DT_D *obs_angs
#         DT_D *sim_angs
# 
#     obs_amps = obs_for_four_trans_struct.amps
#     obs_angs = obs_for_four_trans_struct.angs
# 
#     sim_amps = sim_for_four_trans_struct.amps
#     sim_angs = sim_for_four_trans_struct.angs
# 
#     for i in range((n_pts // 2) - 1):
#         obs_amps_sq_sum += obs_amps[i]**2
#         sim_amps_sq_sum += sim_amps[i]**2
# 
#         freq_corrs[i] = (obs_amps[i] * sim_amps[i]) * (
#             cos(obs_angs[i] - sim_angs[i]))
# 
#         tot_cov += freq_corrs[i]
# 
#     pcorr = tot_cov / (obs_amps_sq_sum * sim_amps_sq_sum)**0.5
#     freq_cov_scale = pcorr / tot_cov
# 
#     freq_corrs[0] = freq_cov_scale * freq_corrs[0]
#     for i in range(1, (n_pts // 2) - 1):
#         freq_corrs[i] = freq_corrs[i - 1] + (freq_cov_scale * freq_corrs[i])
#     return


# cpdef void cmpt_cumm_freq_pcorrs_cy(
#         DT_D[::1] obs_orig, 
#         DT_DC[::1] obs_ft,
#         DT_D[::1] obs_amps,
#         DT_D[::1] obs_angs,
#         DT_D[::1] sim_orig, 
#         DT_DC[::1] sim_ft,
#         DT_D[::1] sim_amps,
#         DT_D[::1] sim_angs,
#         DT_D[::1] cumm_corrs,
#         ) nogil except +:
# 
#     cdef:
#         ForFourTrans1DReal obs_for_four_trans_struct
#         ForFourTrans1DReal sim_for_four_trans_struct
# 
#     obs_for_four_trans_struct.orig = &obs_orig[0]
#     obs_for_four_trans_struct.ft = &obs_ft[0]
#     obs_for_four_trans_struct.amps = &obs_amps[0]
#     obs_for_four_trans_struct.angs = &obs_angs[0]
#     obs_for_four_trans_struct.n_pts = obs_orig.shape[0]
# 
#     sim_for_four_trans_struct.orig = &sim_orig[0]
#     sim_for_four_trans_struct.ft = &sim_ft[0]
#     sim_for_four_trans_struct.amps = &sim_amps[0]
#     sim_for_four_trans_struct.angs = &sim_angs[0]
#     sim_for_four_trans_struct.n_pts = sim_orig.shape[0]
# 
#     cmpt_real_fourtrans_1d(&obs_for_four_trans_struct)
#     cmpt_real_fourtrans_1d(&sim_for_four_trans_struct)
# 
#     cmpt_cumm_freq_pcorrs(
#         &obs_for_four_trans_struct, 
#         &sim_for_four_trans_struct,
#         &cumm_corrs[0])
#     return