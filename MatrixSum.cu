#include<stdio.h>
#include<time.h>

unsigned int matrix_len = 1 << 6;
unsigned int size = sizeof(double) * matrix_len;

void create_matrix(double * array){
    time_t t;
    srand((unsigned)time(&t));
    for(int i=0;i<matrix_len;i++){
        array[i] = (rand() & 0xFF) / 100.0;
    }
}


__global__ void matrix_sum_on_gpu(double * mat1,double * mat2,double *mat3){
    unsigned int idx =  blockIdx.x * blockDim.x + threadIdx.x; 
    printf(" block (%d,%d,%d) ,thread (%d,%d,%d) idx %u \n",blockIdx.x,blockIdx.y,blockIdx.z,
        threadIdx.x,threadIdx.y,threadIdx.z,idx);
    mat3[idx] = mat1[idx] + mat2[idx];
}

void matrix_sum(double * mat1,double *mat2,double *mat3){
    int dev = 0; // 驱动索引
    cudaSetDevice(dev);

    dim3 block = (16); // 16 * 1 * 1
    dim3 grid = ((matrix_len + block.x - 1) / block.x); // 64 * 1 * 1
    double * hmat1,* hmat2,* hmat3;
    cudaMalloc(&hmat1,size);
    cudaMalloc(&hmat2,size);
    cudaMalloc(&hmat3,size);

    cudaMemcpy(hmat1,mat1,size,cudaMemcpyHostToDevice);
    cudaMemcpy(hmat2,mat2,size,cudaMemcpyHostToDevice);

    matrix_sum_on_gpu<<<grid,block>>>(hmat1,hmat2,hmat3);

    cudaMemcpy(mat3,hmat3,size,cudaMemcpyDeviceToHost);

    cudaDeviceReset();
}

int main(){
    double *matrix1,*matrix2,*matrix3_sum;
    
    matrix1 = (double *)malloc(size);
    matrix2 = (double *)malloc(size);
    matrix3_sum = (double *)malloc(size);

    memset(matrix1,0,size);
    memset(matrix2,0,size);
    memset(matrix3_sum,0,size);

    create_matrix(matrix1);
    create_matrix(matrix2);

    matrix_sum(matrix1,matrix2,matrix3_sum);

    for(int i=0;i<matrix_len;i++){
        printf(" %f + %f = %f \n",matrix1[i],matrix2[i],matrix3_sum[i]);
    }
    return 0;
}
