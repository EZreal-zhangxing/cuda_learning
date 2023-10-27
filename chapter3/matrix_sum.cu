#include<cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>
#include<sys/time.h>

long seconds(){
    timeval t;
    gettimeofday(&t,NULL);
    return t.tv_sec * 1e6 + t.tv_usec;
}

__global__ void matrixSum(int *martix1,int * matrix2,int * matrix3,int nx,int ny){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    if(idx <nx && idy < ny){
        matrix3[idx + ny*idy] = martix1[idx + ny*idy] + matrix2[idx + ny*idy];
    }
}

void create_matrix(int * array,int matrix_len){
    time_t t;
    srand((unsigned)time(&t));
    for(int i=0;i<matrix_len;i++){
        array[i] = (rand() & 0xFF) / 100.0;
    }
}

int main(int argc,char * argv[]){
    int nx=1<<14,ny=1<<14;
    int nums = nx*ny,bytes = sizeof(int) * nums;
    int dimx=0,dimy=0;
    if(argc > 2){
        dimx = atoi(argv[1]);
        dimy = atoi(argv[2]);
    }

    int *h_matrix1,* h_matrix2,* h_matrix3;
    h_matrix1 = (int *)malloc(bytes);
    h_matrix2 = (int *)malloc(bytes);
    h_matrix3 = (int *)malloc(bytes);

    memset(h_matrix1,0,bytes);
    memset(h_matrix2,0,bytes);
    memset(h_matrix3,0,bytes);

    create_matrix(h_matrix1,nums);
    create_matrix(h_matrix2,nums);

    int * d_matrix1,*d_matrix2,*d_matrix3;
    cudaMalloc((void **)&d_matrix1,bytes);
    cudaMalloc((void **)&d_matrix2,bytes);
    cudaMalloc((void **)&d_matrix3,bytes);

    cudaMemset(d_matrix3,0,bytes);

    cudaMemcpy(d_matrix1,h_matrix1,bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix2,h_matrix2,bytes,cudaMemcpyHostToDevice);
    
    dim3 block(dimx,dimy);
    dim3 grid((nx + block.x - 1)/block.x,(ny + block.y - 1)/block.y);
    long start = seconds();
    matrixSum<<<grid,block>>>(d_matrix1,d_matrix2,d_matrix3,nx,ny);
    cudaDeviceSynchronize();

    printf(" <<<(%d,%d,%d),(%d,%d,%d) elapsed %f ms\n",grid.x,grid.y,grid.z,block.x,block.y,block.z,(seconds()-start) / 1000.0);
    // cudaMemcpy(h_matrix3,d_matrix3,bytes,cudaMemcpyDeviceToHost);

    // for(int i=0;i<ny;i++){
    //     for(int j=0;j<nx;j++){
    //         if(h_matrix3[j + i * nx] != h_matrix1[j + i * nx] + h_matrix2[j + i * nx]){
    //             printf("calculate failed!\n");
    //             break;
    //         }
    //     }
    // }
    // printf("calculate success!\n");

    free(h_matrix1);
    free(h_matrix2);
    free(h_matrix3);

    cudaFree(d_matrix1);
    cudaFree(d_matrix2);
    cudaFree(d_matrix3);
    cudaDeviceReset();
    return 0;
}