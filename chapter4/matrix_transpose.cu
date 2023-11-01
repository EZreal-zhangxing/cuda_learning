#include<cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<sys/time.h>

long seconds(){
    timeval t;
    gettimeofday(&t,NULL);
    return t.tv_sec * 1e6 + t.tv_usec;
}

/**
 * 按行拷贝
*/
__global__ void copyRow(int * d_matrix,int * d_o_matrix,int row,int col){
    int thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int thread_idy = threadIdx.y + blockIdx.y * blockDim.y;

    if(thread_idx < col && thread_idy < row){
        d_o_matrix[thread_idx + thread_idy * col] = d_matrix[thread_idx + thread_idy * col];
    }
}

/**
 * 按列拷贝
*/
__global__ void copyCol(int * d_matrix,int * d_o_matrix,int row,int col){
    int thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int thread_idy = threadIdx.y + blockIdx.y * blockDim.y;

    if(thread_idx < col && thread_idy < row){
        d_o_matrix[thread_idy + thread_idx * row] = d_matrix[thread_idy + thread_idx * row];
    }
}

/**
 * 按行转置
*/
__global__ void transposeNaiveRow(int * d_matrix,int * d_o_matrix,int row,int col){
    int thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int thread_idy = threadIdx.y + blockIdx.y * blockDim.y;

    if(thread_idx < col && thread_idy < row){
        d_o_matrix[thread_idy + thread_idx * row] = d_matrix[thread_idx + thread_idy * col];
    }
}

/**
 * 按列转置
*/
__global__ void transposeNaiveCol(int * d_matrix,int * d_o_matrix,int row,int col){
    int thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int thread_idy = threadIdx.y + blockIdx.y * blockDim.y;

    if(thread_idx < col && thread_idy < row){
        d_o_matrix[thread_idx + thread_idy * row] = d_matrix[thread_idy + thread_idx * col];
    }
}


__global__ void transposeNaiveRow_unroll4(int * d_matrix,int * d_o_matrix,int row,int col){
    int thread_idx = threadIdx.x + blockIdx.x * blockDim.x * 4;
    int thread_idy = threadIdx.y + blockIdx.y * blockDim.y;

    if((thread_idx + 3 * blockDim.x) < col && thread_idy < row){
        d_o_matrix[thread_idy + thread_idx * row] = d_matrix[thread_idx + thread_idy * col];
        d_o_matrix[thread_idy + (thread_idx + blockDim.x) * row] = d_matrix[thread_idx + thread_idy * col + blockDim.x];
        d_o_matrix[thread_idy + (thread_idx + blockDim.x * 2) * row] = d_matrix[thread_idx + thread_idy * col + blockDim.x * 2];
        d_o_matrix[thread_idy + (thread_idx + blockDim.x * 3) * row] = d_matrix[thread_idx + thread_idy * col + blockDim.x * 3];
    }
}

__global__ void transposeNaiveCol_unroll4(int * d_matrix,int * d_o_matrix,int row,int col){
    int thread_idx = threadIdx.x + blockIdx.x * blockDim.x * 4;
    int thread_idy = threadIdx.y + blockIdx.y * blockDim.y;

    if((thread_idx + 3 * blockDim.x) < col && thread_idy < row){
        d_o_matrix[thread_idx + thread_idy * row] = d_matrix[thread_idy + thread_idx * col];
        d_o_matrix[thread_idx + blockDim.x + thread_idy * row] = d_matrix[thread_idy + (thread_idx + blockDim.x) * col];
        d_o_matrix[thread_idx + blockDim.x * 2 + thread_idy * row] = d_matrix[thread_idy + (thread_idx + blockDim.x * 2) * col];
        d_o_matrix[thread_idx + blockDim.x * 3 + thread_idy * row] = d_matrix[thread_idy + (thread_idx + blockDim.x * 3) * col];
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
    int dimx = 1,dimy = 1;
    if(argc > 2){
        dimx = atoi(argv[1]);
        dimy = atoi(argv[2]);
    }
    int row = 1<<11,col = 1<<11;
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
    copyCol<<<grid,block>>>(d_matrix,d_o_matrix,row,col);
    cudaDeviceSynchronize();
    printf("copyCol using %f ms \n",(seconds() - start)/1000.0);

    start = seconds();
    copyRow<<<grid,block>>>(d_matrix,d_o_matrix,row,col);
    cudaDeviceSynchronize();
    printf("copyRow using %f ms \n",(seconds() - start)/1000.0);

    start = seconds();
    transposeNaiveCol<<<grid,block>>>(d_matrix,d_o_matrix,row,col);
    cudaDeviceSynchronize();
    printf("transposeNaiveCol using %f ms \n",(seconds() - start)/1000.0);
    cudaMemcpy(h_matrix2,d_o_matrix,bytes,cudaMemcpyDeviceToHost);
    if(check_transpose(h_matrix1,h_matrix2,row,col) < 0){
        printf("transposeNaiveCol failed !\n");
    }else{
        printf("transposeNaiveCol success !\n");
    }
    // cudaMemset(d_o_matrix,0,bytes);
    // print_matrix(h_matrix1,row,col);
    // print_matrix(h_matrix2,col,row);

    start = seconds();
    transposeNaiveRow<<<grid,block>>>(d_matrix,d_o_matrix,row,col);
    cudaDeviceSynchronize();
    printf("transposeNaiveRow using %f ms \n",(seconds() - start)/1000.0);
    cudaMemcpy(h_matrix2,d_o_matrix,bytes,cudaMemcpyDeviceToHost);
    if(check_transpose(h_matrix1,h_matrix2,row,col) < 0){
        printf("transposeNaiveRow failed !\n");
    }else{
        printf("transposeNaiveRow success !\n");
    }
    // cudaMemset(d_o_matrix,0,bytes);

    dim3 block_4(dimx,dimy);
    dim3 grid_4((col + 4 * block_4.x - 1)/(4*block_4.x),(row + block_4.y - 1)/block_4.y);

    start = seconds();
    transposeNaiveRow_unroll4<<<grid_4,block_4>>>(d_matrix,d_o_matrix,row,col);
    cudaDeviceSynchronize();
    printf("transposeNaiveRow_unroll4 using %f ms \n",(seconds() - start)/1000.0);
    cudaMemcpy(h_matrix2,d_o_matrix,bytes,cudaMemcpyDeviceToHost);
    if(check_transpose(h_matrix1,h_matrix2,row,col) < 0){
        printf("transposeNaiveRow_unroll4 failed !\n");
    }else{
        printf("transposeNaiveRow_unroll4 success !\n");
    }
    // cudaMemset(d_o_matrix,0,bytes);

    start = seconds();
    transposeNaiveCol_unroll4<<<grid_4,block_4>>>(d_matrix,d_o_matrix,row,col);
    cudaDeviceSynchronize();
    printf("transposeNaiveCol_unroll4 using %f ms \n",(seconds() - start)/1000.0);
    cudaMemcpy(h_matrix2,d_o_matrix,bytes,cudaMemcpyDeviceToHost);
    if(check_transpose(h_matrix1,h_matrix2,row,col) < 0){
        printf("transposeNaiveCol_unroll4 failed !\n");
    }else{
        printf("transposeNaiveCol_unroll4 success !\n");
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

