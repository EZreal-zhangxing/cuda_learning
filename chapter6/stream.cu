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

__global__ void kernal(float * d_matrix,int bias){
    d_matrix[bias + threadIdx.x + blockIdx.x * blockDim.x] = __fsqrt_rn(__expf(d_matrix[bias + threadIdx.x + blockIdx.x * blockDim.x]));
}

void create_matrix(float * array,int matrix_len){
    // time_t t;
    // srand((unsigned)time(&t));
    for(int i=0;i<matrix_len;i++){
        // array[i] = (rand() & 0xFF) / 100.0;
        array[i] = i;
    }
}

int main(int argv, char * argc[]){
    int nums = 1<<20,bytes = sizeof(float) * nums;

    float * h_matrix,* d_matrix;

    // h_matrix = (int *)malloc(bytes);
    cudaMallocHost((void **)&h_matrix,bytes);
    create_matrix(h_matrix,nums);
    for(int x = 0;x<10;x++){
        printf("%5.2f ",h_matrix[x]);
    }
    printf("\n");
    cudaStream_t streams[STREAM];

    cudaEvent_t start[STREAM];
    cudaEvent_t end[STREAM];

    for(int i=0;i<STREAM;i++){
        cudaStreamCreate(streams+i);
        // cudaStreamCreateWithFlags(streams+i,cudaStreamNonBlocking);
        cudaEventCreate(start+i);
        cudaEventCreate(end+i);
    }

    int per_nums = nums/STREAM;

    cudaMalloc((void **)&d_matrix,bytes);
    long start_time = seconds();
    dim3 block(32);
    dim3 grid((per_nums + block.x - 1) / block.x);
    for(int i=0;i<STREAM;i++){
        cudaEventRecord(start[i],streams[i]);
        float * d_matrix_start = d_matrix + i *per_nums;
        float * h_matrix_start = h_matrix + i *per_nums;
        cudaMemcpyAsync(d_matrix_start,h_matrix_start,bytes / STREAM,cudaMemcpyHostToDevice,streams[i]);
        // cudaMemcpyAsync(d_matrix + i * per_nums,h_matrix + i * per_nums,bytes / STREAM,cudaMemcpyHostToDevice,streams[i]);
        kernal<<<grid,block,0,streams[i]>>>(d_matrix,i * per_nums);
        // kernal<<<grid,block,0,streams[i]>>>(d_matrix_start,0);
        // kernal<<<grid,block,0,streams[i]>>>(h_matrix + i * per_nums,0);
        cudaMemcpyAsync(h_matrix + i * per_nums,d_matrix + i * per_nums,bytes / STREAM,cudaMemcpyDeviceToHost,streams[i]);
        cudaEventRecord(end[i],streams[i]);
        // cudaStreamSynchronize(streams[i]);
    }

    for(int i=0;i<STREAM;i++){
        cudaStreamSynchronize(streams[i]);
    }
    float times = 0;
    for(int i=0;i<STREAM;i++){
        cudaEventElapsedTime(&times,start[i],end[i]);
        printf("stream %d elapse time :%f \n",i,times);
    }
    for(int x = 0;x<10;x++){
        printf("%5.2f ",h_matrix[x]);
    }
    printf("\n");
    printf("using %f ms \n",(seconds() - start_time)/1000.0);
    for(int i=0;i<STREAM;i++){
        cudaStreamDestroy(streams[i]);
    }

    cudaDeviceReset();
    // cudaFree(d_matrix);
    // free(h_matrix);
    return 0;
}