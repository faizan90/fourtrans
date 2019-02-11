#pragma once
#include <stdio.h>
#include <complex.h>
#include <mkl_dfti.h>

#if !defined(MKL_ILP64)
#define LI "%li"
#else
#define LI "%lli"
#endif


void mkl_real_dft(
		double *reals_arr,
		_Dcomplex *comps_arr,
		long n_pts,
		int ft_type) {

	DFTI_DESCRIPTOR_HANDLE desc_hdl;
	MKL_LONG status;

	if ((ft_type < 1) or (ft_type > 2)) {

		status = 999;
		goto failed;
	}

	status = DftiCreateDescriptor(
				&desc_hdl,
				DFTI_DOUBLE,
				DFTI_REAL,
				(MKL_LONG) 1,
				(MKL_LONG) n_pts);

	if (0 != status) goto failed;

	status = DftiSetValue(desc_hdl, DFTI_PLACEMENT, DFTI_NOT_INPLACE);

	if (0 != status) goto failed;

	status = DftiSetValue(
				desc_hdl,
				DFTI_CONJUGATE_EVEN_STORAGE,
				DFTI_COMPLEX_COMPLEX);

	if (0 != status) goto failed;

	status = DftiCommitDescriptor(desc_hdl);

	if (0 != status) goto failed;

	if (ft_type == 1) {
		status = DftiComputeForward(desc_hdl, reals_arr, comps_arr);
	}

	else if (ft_type == 2) {
		status = DftiComputeBackward(desc_hdl, comps_arr, reals_arr);
	}

	if (0 != status) goto failed;

	status = DftiFreeDescriptor(&desc_hdl);

	if (0 != status) goto failed;

	return status;

failed:
	printf("FT ERROR, status = "LI"\n", status);
	status = 1;
}


//void mkl_set_desc(DFTI_DESCRIPTOR_HANDLE &desc_hdl, long n_pts) {
//
//	MKL_LONG status;
//
//	status = DftiCreateDescriptor(
//				&desc_hdl,
//				DFTI_DOUBLE,
//				DFTI_REAL,
//				(MKL_LONG) 1,
//				(MKL_LONG) n_pts);
//
//	status = DftiSetValue(
//				desc_hdl,
//				DFTI_PLACEMENT,
//				DFTI_NOT_INPLACE);
//
//	status = DftiSetValue(
//				desc_hdl,
//				DFTI_CONJUGATE_EVEN_STORAGE,
//				DFTI_COMPLEX_COMPLEX);
//
//	status = DftiCommitDescriptor(desc_hdl);
//
//	return;
//}
//
//
//void mkl_dft_real_with_desc(
//		DFTI_DESCRIPTOR_HANDLE &desc_hdl,
//		double *in_reals_arr,
//		_Dcomplex *out_comps_arr) {
//
//	MKL_LONG status;
//
//	status = DftiComputeForward(
//				desc_hdl,
//				in_reals_arr,
//				out_comps_arr);
//
//	return;
//}
//
//
//void mkl_free_desc(DFTI_DESCRIPTOR_HANDLE &desc_hdl) {
//
//	MKL_LONG status;
//
//	status = DftiFreeDescriptor(&desc_hdl);
//
//	return;
//}

