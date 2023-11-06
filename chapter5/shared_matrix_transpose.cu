#include<cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<sys/time.h>

#define DIMX 32
#define DIMY 16

long seconds(){
    timeval t;
    gettimeofday(&t,NULL);
    return t.tv_sec * 1e6 + t.tv_usec;
}

/**
 * 按行拷贝
*/
__global__ void copyRow(int * d_matrix,int * d_o_matrix,int row,int col){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if(idx < col && idy < row){
        d_o_matrix[idy * col + idx] = d_matrix[idy * col + idx];
    }
}

/**
 * 按行读取，按列写入
*/
__global__ void naiveGmem(int * d_matrix,int * d_o_matrix,int row,int col){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if(idx < col && idy < row){
        d_o_matrix[idx * row + idy] = d_matrix[idy * col + idx];
    }
}

/**
 * 按行读取写入共享内存，共享内存按行读取写入输出
*/
__global__ void naiveSmem(int * d_matrix,int * d_o_matrix,int row,int col){

    __shared__ int tile[DIMY][DIMX];
    
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if(idx < col && idy < row){
        tile[threadIdx.y][threadIdx.x] = d_matrix[idx + idy * col];

        __syncthreads();

        d_o_matrix[idy + idx * row] = tile[threadIdx.y][threadIdx.x];
    }
}

/**
 * 按行读取写入共享内存，共享内存按列读取写入输出
*/
__global__ void naiveSmemCol(int * d_matrix,int * d_o_matrix,int row,int col){

    __shared__ int tile[DIMY][DIMX];
    
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    int tid_in_block = threadIdx.x + threadIdx.y * blockDim.x;
    if(idx < col && idy < row){
        tile[threadIdx.y][threadIdx.x] = d_matrix[idx + idy * col];

        __syncthreads();
        int blockCol = tid_in_block % blockDim.y; 
        int blockRow = tid_in_block / blockDim.y;
        // tile 按列读取 输出矩阵按行写入
        int toi = (blockRow + blockIdx.x * blockDim.x) * row + (blockCol + blockIdx.y * blockDim.y);
        d_o_matrix[toi] = tile[blockCol][blockRow];
    }
}


int check_transpose(int *matrix1,int * matrix2,int row,int col){
    for(int i=0;i<row;i++){
        for(int j=0;j<col;j++){
            if(matrix1[j + i* col] != matrix2[i + j*row]){
                return -1;
            }
        }
    }
    return 0;
}

void init_matrix(int * matrix,int row,int col){
    for(int i=0;i<row;i++){
        for(int j=0;j<col;j++){
            matrix[j + i* col] = j + i*col;
        }
    }
}

void print_matrix(int * matrix,int row,int col){
    printf("\n");
    for(int i=0;i<row;i++){
        for(int j=0;j<col;j++){
            printf("%4d ",matrix[j + i * col]);
        }
        printf("\n");
    }
}

int main(int argc,char *argv[]){
    int dimx = DIMX,dimy = DIMY;
    if(argc > 2){
        dimx = atoi(argv[1]);
        dimy = atoi(argv[2]);
    }
    int row = 1<<12,col = 1<<12;
    // int row = DIMY,col = DIMX;
    int nums = row * col,bytes = nums * sizeof(int);

    int * h_matrix1,*h_matrix2;
    h_matrix1 = (int *)malloc(bytes);
    h_matrix2 = (int *)malloc(bytes);

    init_matrix(h_matrix1,row,col);

    int * d_matrix, *d_o_matrix;

    cudaMalloc((void **)&d_matrix,bytes);
    cudaMalloc((void **)&d_o_matrix,bytes);

    dim3 block(dimx,dimy);
    dim3 grid((col + block.x - 1)/block.x,(row + block.y - 1)/block.y);
    cudaMemcpy(d_matrix,h_matrix1,bytes,cudaMemcpyHostToDevice);

    printf("execute info <<<(%d,%d),(%d,%d)>>> \n",grid.x,grid.y,block.x,block.y);

    long start = seconds();
    copyRow<<<grid,block>>>(d_matrix,d_o_matrix,row,col);
    cudaDeviceSynchronize();
    printf("row read and row write using %f ms \n",(seconds() - start)/1000.0);

    start = seconds();
    naiveGmem<<<grid,block>>>(d_matrix,d_o_matrix,row,col);
    cudaDeviceSynchronize();
    printf("row read and col write using %f ms ",(seconds() - start)/1000.0);
    cudaMemcpy(h_matrix2,d_o_matrix,bytes,cudaMemcpyDeviceToHost);
    if(check_transpose(h_matrix1,h_matrix2,row,col) < 0){
        printf("\t -- failed !\n");
    }else{
        printf("\t -- success !\n");
    }
    
    memset(h_matrix2,0,bytes);

    start = seconds();
    naiveSmem<<<grid,block>>>(d_matrix,d_o_matrix,row,col);
    cudaDeviceSynchronize();
    printf("read a row write to shared mem and trans to output with read from shared memory by row , using %f ms ",(seconds() - start)/1000.0);
    cudaMemcpy(h_matrix2,d_o_matrix,bytes,cudaMemcpyDeviceToHost);
    if(check_transpose(h_matrix1,h_matrix2,row,col) < 0){
        printf("\t -- failed !\n");
    }else{
        printf("\t -- success !\n");
    }

    memset(h_matrix2,0,bytes);

    start = seconds();
    naiveSmemCol<<<grid,block>>>(d_matrix,d_o_matrix,row,col);
    cudaDeviceSynchronize();
    printf("read a row write to shared mem and trans to output with read from shared memory by col , using %f ms ",(seconds() - start)/1000.0);
    cudaMemcpy(h_matrix2,d_o_matrix,bytes,cudaMemcpyDeviceToHost);
    if(check_transpose(h_matrix1,h_matrix2,row,col) < 0){
        printf("\t -- failed !\n");
    }else{
        printf("\t -- success !\n");
    }
    // print_matrix(h_matrix1,row,col);
    // print_matrix(h_matrix2,col,row);

    cudaFree(d_matrix);
    cudaFree(d_o_matrix);
    free(h_matrix1);
    free(h_matrix2);
    cudaDeviceReset();
    return 0;
}

