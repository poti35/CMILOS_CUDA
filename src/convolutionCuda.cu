
#define MAX_MASK_WIDTH 100
__constant__ double M[MAX_MASK_WIDTH];

/**

    Convolution 1D with kernel M stored in constant memory
    @param *N --> input array signal
    @param *P --> output array 
    @param mask_Width --> width of mask (2width-1)
    @param Width --> size of all arrays 
*/

/*
    for copy array to constant memory 
*/  
    //cudaMemcpyToSymbol(M, M_h, Mask_Width*sizeof(float));

__global__ void convolution_1D_basic_kernel(double *N, double *P, int Mask_Width, int Width) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    __shared__ float N_ds[TILE_SIZE];
    N_ds[threadIdx.x] = N[i];
    __syncthreads();
    int This_tile_start_point = blockIdx.x * blockDim.x;
    int Next_tile_start_point = (blockIdx.x + 1) * blockDim.x;
    int N_start_point = i - (Mask_Width/2);
    float Pvalue = 0;
    for (int j = 0; j < Mask_Width; j ++) {
        int N_index = N_start_point + j;
        if (N_index >= 0 && N_index < Width) {
            if ((N_index >= This_tile_start_point) && (N_index < Next_tile_start_point)) {
                Pvalue += N_ds[threadIdx.x+j-(Mask_Width/2)]*M[j];
            } else {
                Pvalue += N[N_index] * M[j];
            }
        }
    }
    P[i] = Pvalue;
}



#include <stdio.h>
#include <stdlib.h>

#include "TimingGPU.cuh"
#include "Utilities.cuh"

#define RG          10
#define BLOCKSIZE   8

/****************/
/* CPU FUNCTION */
/****************/
void h_convolution_1D(const float * __restrict__ h_Signal, const float * __restrict__ h_ConvKernel, float * __restrict__ h_Result_CPU, 
                      const int N, const int K) {

    for (int i = 0; i < N; i++) {

        float temp = 0.f;

        int N_start_point = i - (K / 2);

        for (int j = 0; j < K; j++) if (N_start_point + j >= 0 && N_start_point + j < N) {
            temp += h_Signal[N_start_point+ j] * h_ConvKernel[j];
        }

        h_Result_CPU[i] = temp;
    }
}

/********************/
/* BASIC GPU KERNEL */
/********************/
__global__ void d_convolution_1D_basic(const float * __restrict__ d_Signal, const float * __restrict__ d_ConvKernel, float * __restrict__ d_Result_GPU, 
                                       const int N, const int K) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float temp = 0.f;

    int N_start_point = i - (K / 2);

    for (int j = 0; j < K; j++) if (N_start_point + j >= 0 && N_start_point + j < N) {
        temp += d_Signal[N_start_point+ j] * d_ConvKernel[j];
    }

    d_Result_GPU[i] = temp;
}

/***************************/
/* GPU KERNEL WITH CACHING */
/***************************/
__global__ void d_convolution_1D_caching(const float * __restrict__ d_Signal, const float * __restrict__ d_ConvKernel, float * __restrict__ d_Result_GPU, 
                                         const int N, const int K) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float d_Tile[BLOCKSIZE];

    d_Tile[threadIdx.x] = d_Signal[i];
    __syncthreads();

    float temp = 0.f;

    int N_start_point = i - (K / 2);

    for (int j = 0; j < K; j++){
        if (N_start_point + j >= 0 && N_start_point + j < N) {

            if ((N_start_point + j >= blockIdx.x * blockDim.x) && (N_start_point + j < (blockIdx.x + 1) * blockDim.x))
                // --- The signal element is in the tile loaded in the shared memory
                temp += d_Tile[threadIdx.x + j - (K / 2)] * d_ConvKernel[j]; 
            else
                // --- The signal element is not in the tile loaded in the shared memory
                temp += d_Signal[N_start_point + j] * d_ConvKernel[j];
        }   
    }

    d_Result_GPU[i] = temp;
}


#define O_Tile_Width 3
#define Mask_width 3
#define width 5
#define Block_width (O_Tile_Width+(Mask_width-1))
#define Mask_radius (Mask_width/2)


__global__ void convolution_1D_tiled(float *N,float *M,float *P)
{
    int index_out_x=blockIdx.x*O_Tile_Width+threadIdx.x;
    int index_in_x=index_out_x-Mask_radius;
    __shared__ float N_shared[Block_width];
    float Pvalue=0.0;

    //Load Data into shared Memory (into TILE)
    if((index_in_x>=0)&&(index_in_x<width))
    {
        N_shared[threadIdx.x]=N[index_in_x];
    }
    else
    {
        N_shared[threadIdx.x]=0.0f;
    }
    __syncthreads();

    //Calculate Convolution (Multiply TILE and Mask Arrays)
    if(threadIdx.x<O_Tile_Width)
    {
        //Pvalue=0.0f;
        for(int j=0;j<Mask_width;j++)
        {
            Pvalue+=M[j]*N_shared[j+threadIdx.x];
        }
    P[index_out_x]=Pvalue;
    }
}


__global__
/**
 * This is the kernel that convolutes the two given float arrays.
 * The result is saved in the third array.
 */
void cudaConvolve(float *a, float *b, float *res, int aLen, int bLen)
{
    // Calculate the length of the result
    int abMax = max(aLen, bLen);
    int convLen = aLen + bLen - 1;

    // Find the starting point and the step of the loop
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;
    for (int n = index; n < convLen; n += step)
    {
        float prod = 0;

        // Find the minimum amount of iterations needed
        int kMax = min(abMax, n);
        for (int k = 0; k <= kMax; ++k)
        {
            // Make sure we're in bounds for both arrays,
            // otherwise there's no overlap between the two.
            if (k < aLen && n - k < bLen)
            {
                prod += a[k] * b[n - k];
            }
        }
        res[n] = prod;
    }
}

/**
 * This method calls the CUDA kernel for the convolution, after
 * calculating the proper amount of blocks and threads needed.
 */
void myConvolve(float *a, float *b, float *res, int aLen, int bLen)
{
    int blockSize = 256;
    int numBlocks = ((aLen + bLen - 1) + blockSize - 1) / blockSize;
    cudaConvolve<<<numBlocks, blockSize>>>(a, b, res, aLen, bLen);
    cudaDeviceSynchronize(); // Wait for all CUDA cores to finish
}

/********/
/* MAIN */
/********/
int main(){

    const int N = 15;           // --- Signal length
    const int K = 5;            // --- Convolution kernel length

    float *h_Signal         = (float *)malloc(N * sizeof(float));
    float *h_Result_CPU     = (float *)malloc(N * sizeof(float));
    float *h_Result_GPU     = (float *)malloc(N * sizeof(float));
    float *h_ConvKernel     = (float *)malloc(K * sizeof(float));

    float *d_Signal;        gpuErrchk(cudaMalloc(&d_Signal,     N * sizeof(float)));
    float *d_Result_GPU;    gpuErrchk(cudaMalloc(&d_Result_GPU, N * sizeof(float)));
    float *d_ConvKernel;    gpuErrchk(cudaMalloc(&d_ConvKernel, K * sizeof(float)));

    for (int i=0; i < N; i++) { h_Signal[i] = (float)(rand() % RG); }

    for (int i=0; i < K; i++) { h_ConvKernel[i] = (float)(rand() % RG); }

    gpuErrchk(cudaMemcpy(d_Signal,      h_Signal,       N * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_ConvKernel,  h_ConvKernel,   K * sizeof(float), cudaMemcpyHostToDevice));

    h_convolution_1D(h_Signal, h_ConvKernel, h_Result_CPU, N, K);

    d_convolution_1D_basic<<<iDivUp(N, BLOCKSIZE), BLOCKSIZE>>>(d_Signal, d_ConvKernel, d_Result_GPU, N, K);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(h_Result_GPU, d_Result_GPU, N * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; i++) if (h_Result_CPU[i] != h_Result_GPU[i]) {printf("mismatch2 at %d, cpu: %d, gpu %d\n", i, h_Result_CPU[i], h_Result_GPU[i]); return 1;}

    printf("Test basic passed\n");

    d_convolution_1D_caching<<<iDivUp(N, BLOCKSIZE), BLOCKSIZE>>>(d_Signal, d_ConvKernel, d_Result_GPU, N, K);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(h_Result_GPU, d_Result_GPU, N * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; i++) if (h_Result_CPU[i] != h_Result_GPU[i]) {printf("mismatch2 at %d, cpu: %d, gpu %d\n", i, h_Result_CPU[i], h_Result_GPU[i]); return 1;}

    printf("Test caching passed\n");

    return 0;
}