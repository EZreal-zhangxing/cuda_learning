#include<cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<sys/time.h>

#define dimx 32
#define dimy 32

long seconds(){
    timeval t;
    gettimeofday(&t,NULL);
    return t.tv_sec * 1e6 + t.tv_usec;
}

__shared__ int array[dimy][dimx];

/**
 * 按行写入，按列读取
*/
__global__ void copyRow(int * out){
    int val = threadIdx.y * blockDim.x + threadIdx.x;
    array[threadIdx.y][threadIdx.x] = val;
    out[val] = array[threadIdx.x][threadIdx.y];
}
/**
 * 按列写入，按行读取
*/
__global__ void copyCol(int * out){
    int val = threadIdx.x * blockDim.y + threadIdx.y;
    array[threadIdx.x][threadIdx.y] = val;
    out[val] = array[threadIdx.y][threadIdx.x];
}
int main(int argc,char * argv[]){
    int * d_out,bytes = dimx * dimy * sizeof(int);
    cudaMalloc((void **)&d_out,bytes);
    
    dim3 block(dimy,dimx);
    dim3 grid(1,1);

    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);

    copyRow<<<grid,block>>>(d_out);
    cudaDeviceSynchronize();
    copyCol<<<grid,block>>>(d_out);
    cudaDeviceSynchronize();

    cudaFree(d_out);
    cudaDeviceReset();
    return 0;
}