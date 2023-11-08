#include<cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<sys/time.h>

#define STREAM 4

long seconds(){
    timeval t;
    gettimeofday(&t,NULL);
    return t.tv_sec * 1e6 + t.tv_usec;
}

__global__ void kernal(int * d_matrix,int bias){
    d_matrix[bias + threadIdx.x + blockIdx.x * blockDim.x] = __fsqrt_rn(__expf(d_matrix[bias + threadIdx.x + blockIdx.x * blockDim.x]));
}

void create_matrix(int * array,int matrix_len){
    // time_t t;
    // srand((unsigned)time(&t));
    for(int i=0;i<matrix_len;i++){
        // array[i] = (rand() & 0xFF) / 100.0;
        array[i] = i;
    }
}

int main(int argv, char * argc[]){
    int nums = 1<<20,bytes = sizeof(int) * nums;

    int * h_matrix,* d_matrix;

    // h_matrix = (int *)malloc(bytes);
    cudaMallocHost((void **)&h_matrix,bytes);
    create_matrix(h_matrix,nums);

    cudaStream_t streams[STREAM];

    for(int i=0;i<STREAM;i++){
        cudaStreamCreate(streams+i);
    }

    int per_nums = nums/STREAM;

    cudaMalloc((void **)&d_matrix,bytes);
    long start = seconds();
    dim3 block(32);
    dim3 grid((per_nums + block.x - 1) / block.x);
    for(int i=0;i<STREAM;i++){
        cudaMemcpyAsync(d_matrix + i * per_nums,h_matrix + i * per_nums,bytes / STREAM,cudaMemcpyHostToDevice,streams[i]);
        kernal<<<grid,block,0,streams[i]>>>(d_matrix,i * per_nums);
        cudaMemcpyAsync(h_matrix + i * per_nums,d_matrix + i * per_nums,bytes / STREAM,cudaMemcpyDeviceToHost,streams[i]);
        // cudaStreamSynchronize(streams[i]);
    }

    for(int i=0;i<STREAM;i++){
        cudaStreamSynchronize(streams[i]);
    }
    // for(int x = 0;x<nums;x++){
    //     printf("%d ",h_matrix[x]);
    // }
    // printf("\n");
    printf("using %f ms \n",(seconds() - start)/1000.0);
    for(int i=0;i<STREAM;i++){
        cudaStreamDestroy(streams[i]);
    }

    
    // cudaFree(d_matrix);
    // free(h_matrix);
    return 0;
}