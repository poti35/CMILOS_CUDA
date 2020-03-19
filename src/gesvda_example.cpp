/*
 * How to compile (assume cuda is installed at /usr/local/cuda/)
 *   nvcc -c -I/usr/local/cuda/include gesvda_example.cpp 
 *   g++ -o gesvda_example gesvda_example.o -L/usr/local/cuda/lib64 -lcudart -lcusolver
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

void printMatrix(int m, int n, const float*A, int lda, const char* name)
{
    for(int row = 0 ; row < m ; row++){
        for(int col = 0 ; col < n ; col++){
            float Areg = A[row + col*lda];
            printf("%s(%d,%d) = %20.16E\n", name, row+1, col+1, Areg);
        }
    }
}

int main(int argc, char*argv[])
{
    cusolverDnHandle_t cusolverH = NULL;
    cudaStream_t stream = NULL;

    cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat1 = cudaSuccess;
    cudaError_t cudaStat2 = cudaSuccess;
    cudaError_t cudaStat3 = cudaSuccess;
    cudaError_t cudaStat4 = cudaSuccess;
    cudaError_t cudaStat5 = cudaSuccess;
    const int batchSize = 2;
    const int m = 3;
    const int n = 2;
    const int lda = m;
    const int ldu = m;
    const int ldv = n;
    const int rank = n;
    const long long int strideA = (long long int)lda*n;
    const long long int strideS = n;
    const long long int strideU = (long long int)ldu*n;
    const long long int strideV = (long long int)ldv*n;
/*        | 1 2  |       | 10 9 |
 *   A0 = | 4 5  |, A1 = |  8 7 |
 *        | 2 1  |       |  6 5 |
 */
    float A[strideA*batchSize] = { 1.0, 4.0, 2.0, 2.0, 5.0, 1.0, 10.0, 8.0, 6.0, 9.0, 7.0, 5.0};
    float U[strideU*batchSize]; /* left singular vectors  */
    float V[strideV*batchSize]; /* right singular vectors */
    float S[strideS*batchSize]; /* numerical singular value */

/* exact singular values */
    float S_exact[strideS*batchSize] = {7.065283497082729, 1.040081297712078, 18.839649186929730, 0.260035600289472};
    float *d_A = NULL;  /* device copy of A */
    float *d_S = NULL;  /* singular values */
    float *d_U = NULL;  /* left singular vectors */
    float *d_V = NULL;  /* right singular vectors */
    int *d_info = NULL;  /* error info */
    int lwork = 0;       /* size of workspace */
    float *d_work = NULL; /* devie workspace for gesvda */
    const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvectors.
    double RnrmF[batchSize]; /* residual norm */
    int info[batchSize];  /* host copy of error info */

    printf("example of gesvdaStridedBatched \n");
    printf("A = (matlab base-1)\n");
    printMatrix(m, n, A, lda, "A0");
    printf("=====\n");
    printMatrix(m, n, A + strideA, lda, "A1");
    printf("=====\n");

/* step 1: create cusolver handle, bind a stream */
    status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == status);
    cudaStat1 = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    assert(cudaSuccess == cudaStat1);
    status = cusolverDnSetStream(cusolverH, stream);
    assert(CUSOLVER_STATUS_SUCCESS == status);

/* step 2: copy A to device */
    cudaStat1 = cudaMalloc ((void**)&d_A   , sizeof(float)*strideA*batchSize);
    cudaStat2 = cudaMalloc ((void**)&d_S   , sizeof(float)*strideS*batchSize);
    cudaStat3 = cudaMalloc ((void**)&d_U   , sizeof(float)*strideU*batchSize);
    cudaStat4 = cudaMalloc ((void**)&d_V   , sizeof(float)*strideV*batchSize);
    cudaStat5 = cudaMalloc ((void**)&d_info, sizeof(int)*batchSize);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);
    assert(cudaSuccess == cudaStat4);
    assert(cudaSuccess == cudaStat5);

    cudaStat1 = cudaMemcpy(d_A, A, sizeof(float)*strideA*batchSize, cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat1);
    cudaDeviceSynchronize(); /* sync with null stream */

/* step 3: query workspace of SVD */
    status = cusolverDnSgesvdaStridedBatched_bufferSize(
        cusolverH,
        jobz, /* CUSOLVER_EIG_MODE_NOVECTOR: compute singular values only */
              /* CUSOLVER_EIG_MODE_VECTOR: compute singular value and singular vectors */
        rank, /* number of singular values */
        m,    /* nubmer of rows of Aj, 0 <= m */
        n,    /* number of columns of Aj, 0 <= n  */
        d_A,     /* Aj is m-by-n */
        lda,     /* leading dimension of Aj */
        strideA, /* >= lda*n */
        d_S,     /* Sj is rank-by-1, singular values in descending order */
        strideS, /* >= rank */
        d_U,     /* Uj is m-by-rank */
        ldu,     /* leading dimension of Uj, ldu >= max(1,m) */
        strideU, /* >= ldu*rank */
        d_V,     /* Vj is n-by-rank */
        ldv,     /* leading dimension of Vj, ldv >= max(1,n) */
        strideV, /* >= ldv*rank */
        &lwork,
        batchSize /* number of matrices */
    );
    assert(CUSOLVER_STATUS_SUCCESS == status);
    cudaStat1 = cudaMalloc((void**)&d_work , sizeof(float)*lwork);
    assert(cudaSuccess == cudaStat1);

/* step 4: compute SVD */
    status = cusolverDnSgesvdaStridedBatched(
        cusolverH,
        jobz, /* CUSOLVER_EIG_MODE_NOVECTOR: compute singular values only */
              /* CUSOLVER_EIG_MODE_VECTOR: compute singular value and singular vectors */
        rank, /* number of singular values */
        m,    /* nubmer of rows of Aj, 0 <= m */
        n,    /* number of columns of Aj, 0 <= n  */
        d_A,     /* Aj is m-by-n */
        lda,     /* leading dimension of Aj */
        strideA, /* >= lda*n */
        d_S,     /* Sj is rank-by-1 */
                 /* the singular values in descending order */
        strideS, /* >= rank */
        d_U,     /* Uj is m-by-rank */
        ldu,     /* leading dimension of Uj, ldu >= max(1,m) */
        strideU, /* >= ldu*rank */
        d_V,     /* Vj is n-by-rank */
        ldv,     /* leading dimension of Vj, ldv >= max(1,n) */
        strideV, /* >= ldv*rank */
        d_work,
        lwork,
        d_info,
        RnrmF,
        batchSize /* number of matrices */
    );
    cudaStat1 = cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == status);
    assert(cudaSuccess == cudaStat1);

    cudaStat1 = cudaMemcpy(U, d_U, sizeof(float)*strideU*batchSize, cudaMemcpyDeviceToHost);
    cudaStat2 = cudaMemcpy(V, d_V, sizeof(float)*strideV*batchSize, cudaMemcpyDeviceToHost);
    cudaStat3 = cudaMemcpy(S, d_S, sizeof(float)*strideS*batchSize, cudaMemcpyDeviceToHost);
    cudaStat4 = cudaMemcpy(info, d_info, sizeof(int)*batchSize, cudaMemcpyDeviceToHost);
    cudaStat5 = cudaDeviceSynchronize();
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);
    assert(cudaSuccess == cudaStat4);
    assert(cudaSuccess == cudaStat5);

    if ( 0 > info[0] ){
        printf("%d-th parameter is wrong \n", -info[0]);
        exit(1);
    }
    for(int idx = 0 ; idx < batchSize; idx++){
        if ( 0 == info[idx] ){
            printf("%d-th matrix, gesvda converges \n", idx );
        }else{
           printf("WARNING: info[%d] = %d : gesvda does not converge \n", idx, info[idx] );
        }
    }

    printf("S = singular values (matlab base-1)\n");
    printf("U = left singular vectors (matlab base-1)\n");
    printf("V = right singular vectors (matlab base-1)\n\n");
    printMatrix(rank, 1, S, n, "S0");
    printf("=====\n");

    printMatrix(m, rank, U, ldu, "U0");
    printf("=====\n");

    printMatrix(n, rank, V, ldv, "V0");
    printf("=====\n");

    float ds_sup = 0;
    for(int j = 0; j < n; j++){
        float err = fabs( S[j] - S_exact[j] );
        ds_sup = (ds_sup > err)? ds_sup : err;
    }
    printf("|S0 - S0_exact|_sup = %E \n", ds_sup);

    printf("residual |A0 - U0*S0*V0**H|_F = %E \n", RnrmF[0] );

    printMatrix(rank, 1, S + strideS, n, "S1");
    printf("=====\n");

    printMatrix(m, rank, U + strideU, ldu, "U1");
    printf("=====\n");

    printMatrix(n, rank, V + strideV, ldv, "V1");
    printf("=====\n");

    ds_sup = 0;
    for(int j = 0; j < n; j++){
        float err = fabs( S[strideS + j] - S_exact[strideS + j] );
        ds_sup = (ds_sup > err)? ds_sup : err;
    }
    printf("|S1 - S1_exact|_sup = %E \n", ds_sup);

    printf("residual |A1 - U1*S1*V1**H|_F = %E \n", RnrmF[1] );

/*  free resources  */
    if (d_A    ) cudaFree(d_A);
    if (d_S    ) cudaFree(d_S);
    if (d_U    ) cudaFree(d_U);
    if (d_V    ) cudaFree(d_V);
    if (d_info ) cudaFree(d_info);
    if (d_work ) cudaFree(d_work);

    if (cusolverH) cusolverDnDestroy(cusolverH);
    if (stream   ) cudaStreamDestroy(stream);

    cudaDeviceReset();

    return 0;
}    