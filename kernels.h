#include<cuda.h>

#pragma once
inline constexpr int BLOCK_SIZE = 16;

template <typename T>
__global__ void matAdd(T *A, T *B, T *C, int N);

template <typename T>
__global__ void scalarAdd(T *A, T *B, T S, int N);

template <typename T>
__global__ void biasAdd(T *A, T *B, T *C, int J, int K);

template <typename T>
__global__ void gemm(T *A, T *B, T *C, int J, int K, int M, int N);

template <typename T>
__global__ void gemm2(T *A, T *B, T *C, int J, int K, int M, int N);


template <typename T>
__global__ void softmax(T *A, T *B, int J, int K);

template <typename T>
__global__ void gradient(T *A, uint32_t *Y, T *B, int J, int K);

template <typename T>
__global__ void relu(T *A, T *B, int N);