/*
 * How to compile (assume cuda is installed at /usr/local/cuda/)
 *   nvcc -c -I/usr/local/cuda/include syevd_example.cpp 
 *   g++ -o a.out syevd_example.o -L/usr/local/cuda/lib64 -lcudart -lcusolver
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "defines.h"
#include "lib.h"
#include <cuda_runtime.h>
#include <cusolverDn.h>


int mil_svd_cuda(PRECISION *h, PRECISION *beta, PRECISION *delta){
	PRECISION epsilon;
	
	static PRECISION h1[NTERMS * NTERMS];
	
	static PRECISION v[NTERMS], w[NTERMS*NTERMS]; // w --> eigenvalues , v --> eigenvectors 
	int i, j;
	static PRECISION aux2[NTERMS];
	int aux_nf, aux_nc;
	
	epsilon = 1e-12;

	for (j = 0; j < NTERMS * NTERMS; j++)
	{
		h1[j] = h[j];
	}

    /********************* CALCULATE EIGEN VALUES AND EIGEN VECTORS *************************/
    cusolverDnHandle_t cusolverH = NULL;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat1 = cudaSuccess;
    cudaError_t cudaStat2 = cudaSuccess;
    cudaError_t cudaStat3 = cudaSuccess;
    double *d_A = NULL;
    double *d_W = NULL;
    int *devInfo = NULL;
    double *d_work = NULL;
    int  lwork = 0;
    int info_gpu = 0;
    // step 1: create cusolver/cublas handle
    cusolver_status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    // step 2: copy A and B to device
    cudaStat1 = cudaMalloc ((void**)&d_A, sizeof(double) * NTERMS * NTERMS);
    cudaStat2 = cudaMalloc ((void**)&d_W, sizeof(double) * NTERMS);
    cudaStat3 = cudaMalloc ((void**)&devInfo, sizeof(int));
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);

    cudaStat1 = cudaMemcpy(d_A, h1, sizeof(double) * NTERMS * NTERMS, cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat1);

    // step 3: query working space of syevd
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors.
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
    cusolver_status = cusolverDnDsyevd_bufferSize(cusolverH,jobz,uplo,NTERMS,d_A,NTERMS,d_W,&lwork);
    assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);
    cudaStat1 = cudaMalloc((void**)&d_work, sizeof(double)*lwork);
    assert(cudaSuccess == cudaStat1);

    printf("\n DEVICE MEMORY CALCULATE\n");
// step 4: compute spectrum
    cusolver_status = cusolverDnDsyevd(cusolverH,jobz,uplo,NTERMS,d_A,NTERMS,d_W,d_work,lwork,devInfo);
    cudaStat1 = cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    assert(cudaSuccess == cudaStat1);

    cudaStat1 = cudaMemcpy(w, d_W, sizeof(double)*NTERMS, cudaMemcpyDeviceToHost);
    cudaStat2 = cudaMemcpy(v, d_A, sizeof(double)*NTERMS*NTERMS, cudaMemcpyDeviceToHost);
    cudaStat3 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);


// free resources
    if (d_A    ) cudaFree(d_A);
    if (d_W    ) cudaFree(d_W);
    if (devInfo) cudaFree(devInfo);
    if (d_work ) cudaFree(d_work);

    if (cusolverH) cusolverDnDestroy(cusolverH);
    /****************************************************************************************/




	multmatrix(beta, 1, NTERMS, v, NTERMS, NTERMS, aux2, &aux_nf, &aux_nc);

	for (i = 0; i < NTERMS; i++)
	{
		aux2[i]= aux2[i]*((fabs(w[i]) > epsilon) ? (1/w[i]): 0.0);
	}

	multmatrix(v, NTERMS, NTERMS, aux2, NTERMS, 1, delta, &aux_nf, &aux_nc);
	
	return 1;
}