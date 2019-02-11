ctypedef double DT_D
ctypedef unsigned long DT_UL
ctypedef unsigned long long DT_ULL
ctypedef double complex DT_DC

ctypedef struct FT1DReal:
    DT_D *orig      # The input array. N should be even.
    DT_DC *ft       # Real Fourier transform of orig with length of N//2 + 1
    DT_D *amps      # Amplitudes of ft. Starting from index 1 to N//2.
    DT_D *angs      # Angles of ft. Starting from index 1 to N//2.
    DT_D *pcorrs    # Cummulative pearson corrs for each frequency
    DT_UL n_pts     # number of values in orig


cdef void cmpt_real_forw_ft_1d(FT1DReal *ft_struct) nogil except +


# cdef void cmpt_cumm_freq_pcorrs(
#         const ForFourTrans1DReal *obs_for_four_trans_struct,
#         const ForFourTrans1DReal *sim_for_four_trans_struct,
#               DT_D *freq_corrs) nogil except +
