#include<cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<sys/time.h>

#define DIMX 16

long seconds(){
    timeval t;
    gettimeofday(&t,NULL);
    return t.tv_sec * 1e6 + t.tv_usec;
}


__global__ void xor_trans(int * d_matrix,int * d_o_matrix,int laneMask){
    int value = d_matrix[threadIdx.x];
    value = __shfl_xor(value,laneMask,DIMX);
    d_o_matrix[threadIdx.x] = value;
}

__global__ void xor_trans_2(int * d_matrix,int * d_o_matrix,int laneMask){
    int value[2];
    value[0] = d_matrix[threadIdx.x * 2];
    value[1] = d_matrix[threadIdx.x * 2 + 1];

    value[0] = __shfl_xor(value[0],laneMask,DIMX);
    value[1] = __shfl_xor(value[1],laneMask,DIMX);

    d_o_matrix[threadIdx.x * 2] = value[0];
    d_o_matrix[threadIdx.x * 2 + 1] = value[1];
}

void init_matrix(int * matrix,int row,int col){
    for(int i=0;i<row;i++){
        for(int j=0;j<col;j++){
            matrix[j + i* col] = j + i*col;
        }
    }
}

int main(int argv,char * argc[]){
    int nums = DIMX,bytes = sizeof(int) * nums;

    int * h_matrix,*h_o_matrix;
    h_matrix = (int * )malloc(bytes);
    h_o_matrix = (int * )malloc(bytes);

    init_matrix(h_matrix,1,nums);

    int * d_matrix,* d_o_matrix;
    cudaMalloc((void **)&d_matrix,bytes);
    cudaMalloc((void **)&d_o_matrix,bytes);

    cudaMemcpy(d_matrix,h_matrix,bytes,cudaMemcpyHostToDevice);

    dim3 block(DIMX);
    dim3 grid(1);

    xor_trans_2<<<grid,block.x/2>>>(d_matrix,d_o_matrix,1);
    cudaMemcpy(h_o_matrix,d_o_matrix,bytes,cudaMemcpyDeviceToHost);

    for(int i=0;i<DIMX;i++){
        printf("%d ",h_matrix[i]);
    }
    printf("\n");
    for(int i=0;i<DIMX;i++){
        printf("%d ",h_o_matrix[i]);
    }
    printf("\n");

    xor_trans<<<grid,block>>>(d_matrix,d_o_matrix,1);
    cudaMemcpy(h_o_matrix,d_o_matrix,bytes,cudaMemcpyDeviceToHost);

    for(int i=0;i<DIMX;i++){
        printf("%d ",h_matrix[i]);
    }
    printf("\n");
    for(int i=0;i<DIMX;i++){
        printf("%d ",h_o_matrix[i]);
    }
    printf("\n");
    return 0;
}
