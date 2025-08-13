#include <stdio.h>
#include <cuda.h>
#include "op.h"
#include "kernels.h"
#include "tensor.h"

extern "C" void launchSoftmax(float *A, float *B, int J, int K) {
  Tensor<float> *t_A = new Tensor<float>({J, K}, A);
  Tensor<float> *t_B = new Tensor<float>({J, K}, B);

  std::vector<Tensor<float>*> tensors = {t_A, t_B};

  Softmax<float> op = Softmax<float>(tensors);

  op.forward();

  float* result = t_B->to_host();
  memcpy(B, result, t_B->n_bytes());

  free(result);

  delete t_A;
  delete t_B;
}

extern "C" void launchGradient(float *A, uint32_t *Y, float *B, int J, int K) {
  float *d_A, *d_B;
  uint32_t *d_Y;

  size_t sz_a = J * K * sizeof(float);
  size_t sz_y = J * sizeof(uint32_t);

  cudaMalloc((void**)&d_A, sz_a);
  cudaMalloc((void**)&d_B, sz_a);
  cudaMalloc((void**)&d_Y, sz_y);

  cudaMemcpy(d_A, A, sz_a, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, sz_a, cudaMemcpyHostToDevice);
  cudaMemcpy(d_Y, Y, sz_y, cudaMemcpyHostToDevice);

  dim3 blockDim(BLOCK_SIZE);
  dim3 gridDim(J);

  gradient<float> <<<gridDim, blockDim>>>(d_A, d_Y, d_B, J, K);

  cudaMemcpy(B, d_B, sz_a, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_Y);
}

extern "C" void launchBiasAdd(float *A, float *B, float *C, int J, int K) {
  Tensor<float> *t_A = new Tensor<float>({J, K}, A);
  Tensor<float> *t_B = new Tensor<float>({K}, B);
  Tensor<float> *t_C = new Tensor<float>({J, K}, C);

  std::vector<Tensor<float>*> tensors = {t_A, t_B, t_C};

  BiasAdd<float> op = BiasAdd<float>(tensors);

  op.forward();

  float* result = t_C->to_host();

  memcpy(C, result, t_C->n_bytes());

  free(result);

  delete t_A;
  delete t_B;
  delete t_C;
}

extern "C" void launchScalarAdd(float *A, float *B, float S, int N) {
  float *d_A, *d_B;

  size_t sz_a = N * sizeof(float);
  size_t sz_b = N * sizeof(float);

  cudaMalloc((void**)&d_A, sz_a);
  cudaMalloc((void**)&d_B, sz_b);

  cudaMemcpy(d_A, A, sz_a, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, sz_b, cudaMemcpyHostToDevice);

  dim3 blockDim(BLOCK_SIZE);
  dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

  scalarAdd<float> <<<gridDim, blockDim>>>(d_A, d_B, S, N);

  cudaMemcpy(B, d_B, sz_b, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  cudaFree(d_A);
  cudaFree(d_B);
}

extern "C" void launchRelu(float *A, float *B, int N) {
  Tensor<float> *t_A = new Tensor<float>({N}, A);
  Tensor<float> *t_B = new Tensor<float>({N}, B);

  std::vector<Tensor<float>*> tensors = {t_A, t_B};

  Relu<float> op = Relu<float>(tensors);
  op.forward();

  float* result = t_B->to_host();
  memcpy(B, result, t_B->n_bytes());
  free(result);

  delete t_A;
  delete t_B;
}

extern "C" void launchSumCols(float *A, float *B, int J, int K) {
  float *d_A, *d_B;

  size_t sz_a = J * K * sizeof(float);
  size_t sz_b = K * sizeof(float);

  cudaMalloc((void**)&d_A, sz_a);
  cudaMalloc((void**)&d_B, sz_b);

  cudaMemcpy(d_A, A, sz_a, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, sz_b, cudaMemcpyHostToDevice);

  dim3 blockDim(BLOCK_SIZE);
  dim3 gridDim(K);

  sumCols<float> <<<gridDim, blockDim>>>(d_A, d_B, J, K);

  cudaMemcpy(B, d_B, sz_b, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  cudaFree(d_A);
  cudaFree(d_B);
}

extern "C" void launchMultInt(int *A, int *B, int *C, int J, int K, int M, int N) {

  int *d_A, *d_B, *d_C;

  size_t sz_a = J * K * sizeof(int);
  size_t sz_b = M * N * sizeof(int);
  size_t sz_c = J * N * sizeof(int);

  cudaMalloc((void**)&d_A, sz_a);
  cudaMalloc((void**)&d_B, sz_b);
  cudaMalloc((void**)&d_C, sz_c);

  cudaMemcpy(d_A, A, sz_a, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, sz_b, cudaMemcpyHostToDevice);

  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (J + BLOCK_SIZE - 1) / BLOCK_SIZE);

  gemm<int><<<gridDim, blockDim>>>(d_A, d_B, d_C, J, K, M, N);

  cudaMemcpy(C, d_C, sz_c, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

extern "C" void launchMult(float *A, float *B, float *C, int J, int K, int M, int N) {

  float *d_A, *d_B, *d_C;

  size_t sz_a = J * K * sizeof(float);
  size_t sz_b = M * N * sizeof(float);
  size_t sz_c = J * N * sizeof(float);

  cudaMalloc((void**)&d_A, sz_a);
  cudaMalloc((void**)&d_B, sz_b);
  cudaMalloc((void**)&d_C, sz_c);

  cudaMemcpy(d_A, A, sz_a, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, sz_b, cudaMemcpyHostToDevice);

  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (J + BLOCK_SIZE - 1) / BLOCK_SIZE);

  gemm<float><<<gridDim, blockDim>>>(d_A, d_B, d_C, J, K, M, N);

  cudaMemcpy(C, d_C, sz_c, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

extern "C" void launchTranspose2(float *A, float *B, float *C, int J, int K, int M, int N) {
  float *d_A, *d_B, *d_C;

  size_t sz_a = J * K * sizeof(float);
  size_t sz_b = M * N * sizeof(float);
  size_t sz_c = K * N * sizeof(float);

  cudaMalloc((void**)&d_A, sz_a);
  cudaMalloc((void**)&d_B, sz_b);
  cudaMalloc((void**)&d_C, sz_c);

  cudaMemcpy(d_A, A, sz_a, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, sz_b, cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, C, sz_c, cudaMemcpyHostToDevice);

  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (K + BLOCK_SIZE - 1) / BLOCK_SIZE);

  gemm2<true, false, float><<<gridDim, blockDim>>>(d_A, d_B, d_C, J, K, M, N);


  cudaMemcpy(C, d_C, sz_c, cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

extern "C" void launchTranspose(float *A, float *B, float *C, int J, int K, int M, int N) {
  float *d_A, *d_B, *d_C;

  size_t sz_a = J * K * sizeof(float);
  size_t sz_b = M * N * sizeof(float);
  size_t sz_c = J * M * sizeof(float);

  cudaMalloc((void**)&d_A, sz_a);
  cudaMalloc((void**)&d_B, sz_b);
  cudaMalloc((void**)&d_C, sz_c);

  cudaMemcpy(d_A, A, sz_a, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, sz_b, cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, C, sz_c, cudaMemcpyHostToDevice);

  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridDim((M + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (J + BLOCK_SIZE - 1) / BLOCK_SIZE);

  gemm2<false, true, float><<<gridDim, blockDim>>>(d_A, d_B, d_C, J, K, M, N);


  cudaMemcpy(C, d_C, sz_c, cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

extern "C" void launchMult2(float *A, float *B, float *C, int J, int K, int M, int N, bool backward) {
  Tensor<float> *t_A = new Tensor<float>({J, K}, A);
  Tensor<float> *t_B = new Tensor<float>({M, N}, B);

  Tensor<float> *t_C = new Tensor<float>({J, N}, C);
  std::vector<Tensor<float>*> tensors = {t_A, t_B, t_C};

  Gemm<float> op = Gemm<float>(tensors);

  op.forward();
  float *result1 = t_C->to_host();
  memcpy(C, result1, t_C->n_bytes());
  free(result1);

  if(backward) {
    op.backward();
    float *result = t_A->grad_to_host();
    memcpy(A, result, t_A->n_bytes());
    free(result);
    
    float *result2 = t_B->grad_to_host();
    memcpy(B, result2, t_B->n_bytes());
    free(result2);
  }


  delete t_A;
  delete t_B;
  delete t_C;
}

extern "C" void launchAdd(float* a, float* b, float* c, int J, int K) {
  Tensor<float> *t_a = new Tensor<float>({J, K}, a);
  Tensor<float> *t_b = new Tensor<float>({J, K}, b);
  Tensor<float> *t_c = new Tensor<float>({J, K}, c);

  std::vector<Tensor<float>*> tensors = {t_a, t_b, t_c};

  MatAdd<float> op = MatAdd<float>(tensors);

  op.forward();

  float* result = t_c->to_host();

  memcpy(c, result, t_c->n_bytes());

  free(result);
}