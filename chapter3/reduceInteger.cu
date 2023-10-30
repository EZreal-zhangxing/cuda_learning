#include<cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>
#include<sys/time.h>

/**
 * 并行归约问题
*/

long seconds(){
    timeval t;
    gettimeofday(&t,NULL);
    return t.tv_sec * 1e6 + t.tv_usec;
}

long sum_on_cpu(int * matrix1,int stride,int size){
    if(size == 1){
        return matrix1[0];
    }
    for(int i=0;i<stride;i++){
        matrix1[i] += matrix1[i + stride];
    }
    return sum_on_cpu(matrix1,stride /2,size/2);
}

/**
 * threadIdx
 * 间隔一个线程去执行
 * -------------------------------------
 * |  0  |     |  3  |     |  5  |     |
 * -------------------------------------
 *    |     |     |     |     |     |   
 *    -------     -------     -------  
*/
__global__ void matrixSum(int *martix1){
    int thread_idx = threadIdx.x;
    int *matrix_start = martix1 + blockIdx.x * blockDim.x;
    for(int i=1;i<blockDim.x;i*=2){
        if(thread_idx % (2 * i) == 0){
            matrix_start[thread_idx] += matrix_start[thread_idx + i];
        }
        __syncthreads();
    }
}


/**
 * threadIdx
 * 相邻线程执行
 * -------------------------------------
 * |  0  |     |  1  |     |  2  |     |
 * -------------------------------------
 *    |     |     |     |     |     |   
 *    -------     -------     -------  
 *    |           |           |
 * -------------------------------------
 * |  0  |     |  1  |     |  2  |     |
 * -------------------------------------
*/
__global__ void matrixSumNeighbored(int * matrix1){
    int * matrix_start = matrix1 + blockIdx.x * blockDim.x;
    for(int i=1;i<blockDim.x;i*=2){
        if(threadIdx.x * (2 * i) < blockDim.x){
            matrix_start[threadIdx.x * (2 * i)] += matrix_start[threadIdx.x * (2 * i) + i];
        }
        __syncthreads();
    }
}


/**
 * threadIdx
 * 相邻线程执行，交错求和
*/
__global__ void matrixSumInterleaved(int * matrix1){
    int * matrix_start = matrix1 + blockIdx.x * blockDim.x;

    for(int stride=blockDim.x/2;stride >= 1;stride/=2){
        if(threadIdx.x < stride){
            matrix_start[threadIdx.x] += matrix_start[threadIdx.x + stride];
        }
        __syncthreads();
    }

}

/**
 * 交错求和+展开归约2
*/
__global__ void matrixSumInterleavedUnrolling2(int * matrix1){
    // 每个线程块处理两块数据
    int * matrix_start = matrix1 + blockIdx.x * blockDim.x * 2;
    matrix_start[threadIdx.x] += matrix_start[threadIdx.x + blockDim.x];
    __syncthreads();
    for(int stride=blockDim.x/2;stride >= 1;stride/=2){
        if(threadIdx.x < stride){
            matrix_start[threadIdx.x] += matrix_start[threadIdx.x + stride];
        }
        __syncthreads();
    }
}

/**
 * 交错求和+展开归约2+块内展开归约4
*/
__global__ void matrixSumInterleavedUnrolling2_4(int * matrix1){
    // 每个线程块处理两块数据
    int * matrix_start = matrix1 + blockIdx.x * blockDim.x * 2;
    matrix_start[threadIdx.x] += matrix_start[threadIdx.x + blockDim.x];
    __syncthreads();
    for(int stride=blockDim.x/2;stride >= 16;stride /= 2){
        if(threadIdx.x < stride){
            matrix_start[threadIdx.x] += matrix_start[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if(threadIdx.x < 16){
        volatile int * vol_matrix = matrix_start; //内存可见 让前面的数据操作立马更新到内存
        vol_matrix[threadIdx.x] += vol_matrix[threadIdx.x + 8];
        vol_matrix[threadIdx.x] += vol_matrix[threadIdx.x + 4];
        vol_matrix[threadIdx.x] += vol_matrix[threadIdx.x + 2];
        vol_matrix[threadIdx.x] += vol_matrix[threadIdx.x + 1];
    }
}

void create_matrix(int * array,int matrix_len){
    time_t t;
    srand((unsigned)time(&t));
    for(int i=0;i<matrix_len;i++){
        array[i] = (rand() & 0xFF) / 10.0;
    }
}

int main(int argc,char * argv[]){
    int nx=1<<24;
    int nums = nx,bytes = sizeof(int) * nums;
    int dimx=0,dimy=1;
    if(argc > 1){
        dimx = atoi(argv[1]);
    }

    int *h_matrix1,*h_matrix2;
    h_matrix1 = (int *)malloc(bytes);
    h_matrix2 = (int *)malloc(bytes);

    memset(h_matrix1,0,bytes);
    memset(h_matrix2,0,bytes);

    create_matrix(h_matrix1,nums);

    int * d_matrix1;
    cudaMalloc((void **)&d_matrix1,bytes);

    cudaMemcpy(d_matrix1,h_matrix1,bytes,cudaMemcpyHostToDevice);
    

    dim3 block(dimx,dimy);
    dim3 grid((nx + block.x - 1)/block.x,1);
    long start = seconds();
    matrixSum<<<grid,block>>>(d_matrix1);
    cudaDeviceSynchronize();
    printf(" matrixSum <<<(%d,%d,%d),(%d,%d,%d)>>> elapsed %f ms \n",grid.x,grid.y,grid.z,block.x,block.y,block.z,
        (seconds()-start) / 1000.0);
    cudaMemcpy(h_matrix2,d_matrix1,bytes,cudaMemcpyDeviceToHost);

    long sum = 0;
    for(int i=0;i<grid.x;i++){
        sum += h_matrix2[i * block.x];
    }
    printf(" matrixSum <<<(%d,%d,%d),(%d,%d,%d)>>> elapsed all %f ms sum %ld\n",grid.x,grid.y,grid.z,block.x,block.y,block.z,
        (seconds()-start) / 1000.0,sum);

    // reset data
    cudaMemcpy(d_matrix1,h_matrix1,bytes,cudaMemcpyHostToDevice);
    start = seconds();
    matrixSumNeighbored<<<grid,block>>>(d_matrix1);
    cudaDeviceSynchronize();
    printf(" matrixSumNeighbored <<<(%d,%d,%d),(%d,%d,%d)>>> elapsed %f ms \n",grid.x,grid.y,grid.z,block.x,block.y,block.z,
        (seconds()-start) / 1000.0);
    cudaMemcpy(h_matrix2,d_matrix1,bytes,cudaMemcpyDeviceToHost);

    sum = 0;
    for(int i=0;i<grid.x;i++){
        sum += h_matrix2[i * block.x];
    }
    printf(" matrixSumNeighbored <<<(%d,%d,%d),(%d,%d,%d)>>> elapsed all %f ms sum %ld\n",grid.x,grid.y,grid.z,block.x,block.y,block.z,
        (seconds()-start) / 1000.0,sum);

    // reset data
    cudaMemcpy(d_matrix1,h_matrix1,bytes,cudaMemcpyHostToDevice);
    start = seconds();
    matrixSumInterleaved<<<grid,block>>>(d_matrix1);
    cudaDeviceSynchronize();
    printf(" matrixSumInterleaved <<<(%d,%d,%d),(%d,%d,%d)>>> elapsed %f ms \n",grid.x,grid.y,grid.z,block.x,block.y,block.z,
        (seconds()-start) / 1000.0);
    cudaMemcpy(h_matrix2,d_matrix1,bytes,cudaMemcpyDeviceToHost);

    sum = 0;
    for(int i=0;i<grid.x;i++){
        sum += h_matrix2[i * block.x];
    }
    printf(" matrixSumInterleaved <<<(%d,%d,%d),(%d,%d,%d)>>> elapsed all %f ms sum %ld\n",grid.x,grid.y,grid.z,block.x,block.y,block.z,
        (seconds()-start) / 1000.0,sum);


    // reset data
    cudaMemcpy(d_matrix1,h_matrix1,bytes,cudaMemcpyHostToDevice);
    start = seconds();
    matrixSumInterleavedUnrolling2<<<grid.x/2,block>>>(d_matrix1);
    cudaDeviceSynchronize();
    printf(" matrixSumInterleavedUnrolling2 <<<(%d,%d,%d),(%d,%d,%d)>>> elapsed %f ms \n",grid.x/2,grid.y,grid.z,block.x,block.y,block.z,
        (seconds()-start) / 1000.0);
    cudaMemcpy(h_matrix2,d_matrix1,bytes,cudaMemcpyDeviceToHost);

    sum = 0;
    for(int i=0;i<grid.x/2;i++){
        sum += h_matrix2[i * block.x * 2];
    }
    printf(" matrixSumInterleavedUnrolling2 <<<(%d,%d,%d),(%d,%d,%d)>>> elapsed all %f ms sum %ld\n",grid.x/2,grid.y,grid.z,block.x,block.y,block.z,
        (seconds()-start) / 1000.0,sum);


    // reset data
    cudaMemcpy(d_matrix1,h_matrix1,bytes,cudaMemcpyHostToDevice);
    start = seconds();
    matrixSumInterleavedUnrolling2_4<<<grid.x/2,block>>>(d_matrix1);
    cudaDeviceSynchronize();
    printf(" matrixSumInterleavedUnrolling2_4 <<<(%d,%d,%d),(%d,%d,%d)>>> elapsed %f ms \n",grid.x/2,grid.y,grid.z,block.x,block.y,block.z,
        (seconds()-start) / 1000.0);
    cudaMemcpy(h_matrix2,d_matrix1,bytes,cudaMemcpyDeviceToHost);

    sum = 0;
    for(int i=0;i<grid.x/2;i++){
        sum += h_matrix2[i * block.x * 2];
    }
    printf(" matrixSumInterleavedUnrolling2_4 <<<(%d,%d,%d),(%d,%d,%d)>>> elapsed all %f ms sum %ld\n",grid.x/2,grid.y,grid.z,block.x,block.y,block.z,
        (seconds()-start) / 1000.0,sum);

    // cudaMemcpy(h_matrix3,d_matrix3,bytes,cudaMemcpyDeviceToHost);
    start = seconds();
    long sum_c = sum_on_cpu(h_matrix1,nx/2,nx);

    printf(" cpu elapsed %f ms sum %ld \n",(seconds()-start) / 1000.0,sum_c);
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

    cudaFree(d_matrix1);
    cudaDeviceReset();
    return 0;
}