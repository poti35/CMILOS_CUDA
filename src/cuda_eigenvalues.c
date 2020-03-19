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
#include <gsl/gsl_spline.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_eigen.h>


extern gsl_vector *eval;
extern gsl_matrix *evec;
extern gsl_eigen_symmv_workspace * workspace;


int mil_svd_cuda(PRECISION *h, PRECISION *beta, PRECISION *delta){
	
    PRECISION epsilon;
	static PRECISION h1[NTERMS * NTERMS];
    static PRECISION h2[NTERMS * NTERMS];
    static PRECISION h3[NTERMS] [NTERMS];

	PRECISION v[NTERMS*NTERMS], w[NTERMS]; // w --> eigenvalues , v --> eigenvectors 
    PRECISION *v1, *w1;
	int i, j;
	static PRECISION aux2[NTERMS];
	int aux_nf, aux_nc;
	
	epsilon = 1e-12;

	for (j = 0; j < NTERMS * NTERMS; j++)
	{
		h1[j] = h[j];
        h2[j] = h[j];
	}

	gsl_matrix_view gsl_h1 = gsl_matrix_view_array (h1, NTERMS, NTERMS);
	gsl_eigen_symmv(&gsl_h1.matrix, eval, evec, workspace);
	w1 = gsl_vector_ptr(eval,0);
	v1 = gsl_matrix_ptr(evec,0,0);

    printf("\n AUTOVECTORES GSL V1\n");
    for(i=0;i<NTERMS;i++){
        for(j=0;j<NTERMS;j++){
            printf("%f\t",v1[j+ (NTERMS*i)]);
        }   
        printf("\n");
    }
    printf("\n");

    printf("\n AUTOVALORES GSL W1 \n");
    for(i=0;i<NTERMS;i++){
        printf("%f\n",w1[i]);
    }
    printf("\n");    


    /********************* CALCULATE EIGEN VALUES AND EIGEN VECTORS *************************/
    cusolverDnHandle_t cusolverH = NULL;
    cudaStream_t stream = NULL;
    syevjInfo_t syevj_params = NULL;
    cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat1 = cudaSuccess;
    cudaError_t cudaStat2 = cudaSuccess;
    cudaError_t cudaStat3 = cudaSuccess;
    double *d_A = NULL;
    double *d_W = NULL;
    int *devInfo = NULL;
    int *d_info = NULL; /* error info */
    double *d_work = NULL;
    int  lwork = 0;
    int info_gpu = 0;

    /* configuration of syevj  */
    const double tol = 1.e-14;
    const int max_sweeps = 15;
    const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvectors.
    const cublasFillMode_t  uplo = CUBLAS_FILL_MODE_LOWER;

    // step 1: create cusolver/cublas handle
    /*cusolver_status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);*/

    /* step 1: create cusolver handle, bind a stream  */
    status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == status);

    cudaStat1 = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    assert(cudaSuccess == cudaStat1);

    status = cusolverDnSetStream(cusolverH, stream);
    assert(CUSOLVER_STATUS_SUCCESS == status);

/* step 2: configuration of syevj */
    status = cusolverDnCreateSyevjInfo(&syevj_params);
    assert(CUSOLVER_STATUS_SUCCESS == status);

/* default value of tolerance is machine zero */
    status = cusolverDnXsyevjSetTolerance(
        syevj_params,
        tol);
    assert(CUSOLVER_STATUS_SUCCESS == status);

/* default value of max. sweeps is 100 */
    status = cusolverDnXsyevjSetMaxSweeps(
        syevj_params,
        max_sweeps);
    assert(CUSOLVER_STATUS_SUCCESS == status);


    // step 2: copy A and B to device
    cudaStat1 = cudaMalloc ((void**)&d_A, sizeof(double) * NTERMS * NTERMS);
    cudaStat2 = cudaMalloc ((void**)&d_W, sizeof(double) * NTERMS);
    cudaStat3 = cudaMalloc ((void**)&d_info, sizeof(int));
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);

    cudaStat1 = cudaMemcpy(d_A, h2, sizeof(double) * NTERMS * NTERMS, cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat1);

    // step 3: query working space of syevd

    /*cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors.
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;
    cusolver_status = cusolverDnDsyevd_bufferSize(cusolverH,jobz,uplo,NTERMS,d_A,NTERMS,d_W,&lwork);
    assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);
    cudaStat1 = cudaMalloc((void**)&d_work, sizeof(double)*lwork);
    assert(cudaSuccess == cudaStat1);*/

/* step 4: query working space of syevj */
    status = cusolverDnDsyevj_bufferSize(cusolverH,jobz,uplo,NTERMS,d_A,NTERMS,d_W,&lwork,syevj_params);
    assert(CUSOLVER_STATUS_SUCCESS == status);
 
    cudaStat1 = cudaMalloc((void**)&d_work, sizeof(double)*lwork);
    assert(cudaSuccess == cudaStat1);
    
// step 4: compute spectrum
    /*cusolver_status = cusolverDnDsyevd(cusolverH,jobz,uplo,NTERMS,d_A,NTERMS,d_W,d_work,lwork,devInfo);
    cudaStat1 = cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    assert(cudaSuccess == cudaStat1);*/
/* step 5: compute eigen-pair   */
    status = cusolverDnDsyevj(cusolverH,jobz,uplo, NTERMS,d_A,NTERMS,d_W,d_work,lwork,d_info,syevj_params);
    cudaStat1 = cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == status);
    assert(cudaSuccess == cudaStat1);    
    //printf("\n eigenvalues calculados\n");

    cudaStat1 = cudaMemcpy(w, d_W, sizeof(double)*NTERMS, cudaMemcpyDeviceToHost);
    cudaStat2 = cudaMemcpy(v, d_A, sizeof(double)*NTERMS*NTERMS, cudaMemcpyDeviceToHost);
    //cudaStat3 = cudaMemcpy(&info_gpu, d_info, sizeof(int), cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    //assert(cudaSuccess == cudaStat3);


// free resources
    if (d_A    ) cudaFree(d_A);
    if (d_W    ) cudaFree(d_W);
    if (d_info) cudaFree(d_info);
    if (d_work ) cudaFree(d_work);

    if (cusolverH) cusolverDnDestroy(cusolverH);
    /****************************************************************************************/


    printf("\n AUTOVECTORES CUDA V1\n");
    for(i=0;i<NTERMS;i++){
        for(j=0;j<NTERMS;j++){
            printf("%f\t",v[j+ (NTERMS*i)]);
        }
        printf("\n");
    }

    printf("\n\n");
    int fil,col;
    for(i=NTERMS-1,col=0;i>=0;i--,col++){
        for(j=0,fil=0;j<NTERMS;j++,fil++){
            h3[fil][col]=v[j+ (NTERMS*i)];
            //printf("%f(%d,%d)\t",h3[fil][col],fil,col);
        }
        //printf("\n");
    }
    printf("\n");
    printf("\n");

    for(fil=0;fil<NTERMS;fil++){
        for(col=0;col<NTERMS;col++){
            printf("%f(%d,%d)\t",h3[fil][col],fil,col);
        }
        printf("\n");
    }
    printf("\n");
    printf("\n AUTOVALORES CUDA W1 \n");
    for(i=NTERMS-1;i>=0;i--){
        printf("%f\n",w[i]);
    }
    printf("\n");    



	multmatrix(beta, 1, NTERMS, v, NTERMS, NTERMS, aux2, &aux_nf, &aux_nc);

	for (i = 0; i < NTERMS; i++)
	{
		aux2[i]= aux2[i]*((fabs(w[i]) > epsilon) ? (1/w[i]): 0.0);
	}

	multmatrix(v, NTERMS, NTERMS, aux2, NTERMS, 1, delta, &aux_nf, &aux_nc);
	
	return 1;
}