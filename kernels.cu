#include "kernels.h"

template <typename T>
__global__ void matAdd(T *A, T *B, T *C, int N) {
  int t_x = blockIdx.x * blockDim.x + threadIdx.x;

  if(t_x < N) {
    C[t_x] = A[t_x] + B[t_x]; 
  }
}

template <typename T>
__global__ void scalarAdd(T *A, T *B, T S, int N) {
  int t_x = blockIdx.x * blockDim.x + threadIdx.x;

  if(t_x < N) {
    B[t_x] = A[t_x] + S;
  }

}

template <typename T>
__global__ void biasAdd(T* A, T* B, T* C, int J, int K) {
  int t_x = blockIdx.x * blockDim.x + threadIdx.x;
  int t_y = blockIdx.y * blockDim.y + threadIdx.y;

  if (t_x < K && t_y < J) {
    C[t_y * K + t_x] = A[t_y * K + t_x] + B[t_x];
  }

}

template <bool aTrans, bool bTrans, typename T>
__global__ void gemm2(T *A, T *B, T *C, int J, int K, int M, int N) {

  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  __shared__ T a[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ T b[BLOCK_SIZE][BLOCK_SIZE];

  T sum = 0;

  for(int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; t++) {
    int aTile = (t * BLOCK_SIZE) + threadIdx.x;
    int aRow = aTrans ? aTile : row;
    int aCol = aTrans ? row : aTile;

    if(aRow < J && aCol < K){
      a[threadIdx.y][threadIdx.x] = A[(aRow * K) + aCol];
    }
    else{
      a[threadIdx.y][threadIdx.x] = 0;
    }

    int bTile = (t * BLOCK_SIZE) + threadIdx.y;
    int bRow = bTrans ? col : bTile;
    int bCol = bTrans ? bTile : col;

    if(bRow < K && bCol < N) {
      b[threadIdx.y][threadIdx.x] = B[(bRow * N) + bCol];
    }
    else {
      b[threadIdx.y][threadIdx.x] = 0;
    }

    __syncthreads();
    
    #pragma unroll
    for(int k = 0; k < BLOCK_SIZE; k++) {
      sum += a[threadIdx.y][k] * b[k][threadIdx.x];
    }

    __syncthreads();
  }

  if(col < N && row < J) {
    C[row * N + col] = sum;
  }

}

template <typename T>
__global__ void gemm(T *A, T *B, T *C, int J, int K, int M, int N) {

  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  __shared__ T a[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ T b[BLOCK_SIZE][BLOCK_SIZE];

  T sum = 0;

  for(int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; t++) {
    int a_i = (t * BLOCK_SIZE) + threadIdx.x;

    if(a_i < K && row < J){
      a[threadIdx.y][threadIdx.x] = A[(row * K) + a_i];
    }
    else{
      a[threadIdx.y][threadIdx.x] = 0;
    }

    int b_i = (t * BLOCK_SIZE) + threadIdx.y;
    if(b_i < K && col < N) {
      b[threadIdx.y][threadIdx.x] = B[(b_i * N) + col];
    }
    else {
      b[threadIdx.y][threadIdx.x] = 0;
    }

    __syncthreads();
    
    for(int k = 0; k < BLOCK_SIZE; k++) {
      sum += a[threadIdx.y][k] * b[k][threadIdx.x];
    }

    __syncthreads();
  }

  if(col < N && row < J) {
    C[row * N + col] = sum;
  }

}

//compute softmax then subtract indices encoded label
template <typename T>
__global__ void gradient(T *A, uint32_t *Y, T *B, int J, int K) {
  int col = threadIdx.x;
  int row = blockIdx.x;

  __shared__ T s_data[BLOCK_SIZE];
  T local_max = (T) -INFINITY;

  for(int i = col; i < K; i += BLOCK_SIZE) {
    local_max = local_max > A[(row * K) + i] ? local_max : A[(row * K) + i];
  }

  s_data[col] = local_max;

  __syncthreads();

  for(int i = BLOCK_SIZE / 2; i > 0; i /= 2) {
    if(col < i) {
      s_data[col] = s_data[col] > s_data[col + i] ? s_data[col] : s_data[col + i];
    }
    
    __syncthreads();
  }

  
  for(int i = col; i < K; i += BLOCK_SIZE) {
    B[(row * K) + i] = (T) expf(A[(row * K) + i] - s_data[0]);
  }

  T local_sum = 0;
  for(int i = col; i < K; i += BLOCK_SIZE) {
    local_sum += B[(row * K) + i];
  }

  s_data[col] = local_sum;

  __syncthreads();

  for(int i = BLOCK_SIZE / 2; i > 0; i /= 2) {
    if(col < i) {
      s_data[col] = s_data[col] + s_data[col + i];
    }
    
    __syncthreads();
  }

  __syncthreads();

  for(int i = col; i < K; i += BLOCK_SIZE) {
    B[(row * K) + i] = B[(row * K) + i] / s_data[0];
  }

  for(int i = col; i < K; i += BLOCK_SIZE) {
    if(i == Y[row]) {
      B[(row *K) + i] -= (T) 1.0f;
    }
  }

}

template <typename T>
__global__ void softmax(T *A, T *B, int J, int K) {
  int col = threadIdx.x;
  int row = blockIdx.x;

  __shared__ T s_data[BLOCK_SIZE];
  T local_max = (T) -INFINITY;

  for(int i = col; i < K; i += BLOCK_SIZE) {
    local_max = local_max > A[(row * K) + i] ? local_max : A[(row * K) + i];
  }

  s_data[col] = local_max;

  __syncthreads();

  for(int i = BLOCK_SIZE / 2; i > 0; i /= 2) {
    if(col < i) {
      s_data[col] = s_data[col] > s_data[col + i] ? s_data[col] : s_data[col + i];
    }
    
    __syncthreads();
  }

  __syncthreads();

  for(int i = col; i < K; i += BLOCK_SIZE) {
    B[(row * K) + i] = (T) expf(A[(row * K) + i] - s_data[0]);
  }

  T local_sum = 0;
  for(int i = col; i < K; i += BLOCK_SIZE) {
    local_sum += B[(row * K) + i];
  }

  s_data[col] = local_sum;

  __syncthreads();

  for(int i = BLOCK_SIZE / 2; i > 0; i /= 2) {
    if(col < i) {
      s_data[col] = s_data[col] + s_data[col + i];
    }
    
    __syncthreads();
  }

  __syncthreads();

  for(int i = col; i < K; i += BLOCK_SIZE) {
    B[(row * K) + i] = B[(row * K) + i] / s_data[0];
  }

}

template <typename T>
__global__ void relu(T *A, T *B, int N) {
  int t_x = blockIdx.x * blockDim.x + threadIdx.x;

  if(t_x < N) {
    B[t_x] = A[t_x] > 0 ? A[t_x] : 0;
  }

}


template __global__ void matAdd<float>(float *A, float *B, float *C, int N);
template __global__ void scalarAdd<float>(float *A, float *B, float S, int N);
template __global__ void biasAdd<float>(float *A, float *B, float *C, int J, int K);
template __global__ void gemm<float>(float *A, float *B, float *C, int J, int K, int M, int N);
template __global__ void gemm2<bool, bool, float>(float *A, float *B, float *C, int J, int K, int M, int N);
template __global__ void gemm<int>(int *A, int *B, int *C, int J, int K, int M, int N);
template __global__ void softmax<float>(float *A, float *B, int J, int K);
template __global__ void gradient<float>(float *A, uint32_t *Y, float *B, int J, int K);
template __global__ void relu<float>(float *A, float *B, int N);