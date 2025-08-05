#pragma once
#include <vector>
#include <stdexcept>
#include "kernels.h"
#include "tensor.h"

template<typename T>
class Op {
  protected:
    std::vector<Tensor<T>*> _tensors;

  public:
    Op(std::vector<Tensor<T>*> tensors) {
      _tensors = tensors;
    }

    virtual void forward() = 0;
};

template<typename T>
class Gradient : public Op<T> {
  public:
    Gradient(std::vector<Tensor<T>*> tensors) : Op<T>(tensors) {
      if(tensors.size() != 3) {
        throw std::invalid_argument("Gradient requires exactly 3 tensors");
      }
    }
  
    void forward() {
      Tensor<T> *a = this->_tensors[0];
      Tensor<uint32_t> *y = this->_tensors[1];
      Tensor<T> *b = this->_tensors[2];

      int J = a->shape()[0];
      int K = a->shape()[1];

      dim3 blockDim(BLOCK_SIZE);
      dim3 gridDim(J);

      gradient<T> <<<gridDim, blockDim>>>(a->data(), y->data(), b->data(), J, K);

      cudaDeviceSynchronize();
      
    }
};

template<typename T>
class Softmax : public Op<T> {
  public:
    Softmax(std::vector<Tensor<T>*> tensors) : Op<T>(tensors) {
      if (tensors.size() != 2) {
        throw std::invalid_argument("Softmax requires exactly 2 tensors");
      }
    }

    void forward() {
      Tensor<T> *a = this->_tensors[0];
      Tensor<T> *b = this->_tensors[1];

      int J = a->shape()[0];
      int K = a->shape()[1];

      dim3 blockDim(BLOCK_SIZE);
      dim3 gridDim(J);

      softmax<T> <<<gridDim, blockDim>>>(a->data(), b->data(), J, K);

      cudaDeviceSynchronize();

    }
    
};

template<typename T>
class Relu : public Op<T> {
  public:
    Relu(std::vector<Tensor<T>*> tensors) : Op<T>(tensors) {
      if (tensors.size() != 2) {
        throw std::invalid_argument("Relu requires exactly 2 tensors");
      }
    }

    void forward() {
      Tensor<T> *a = this->_tensors[0];
      Tensor<T> *b = this->_tensors[1];

      int N = a->n_elem();
      dim3 blockDim(BLOCK_SIZE);
      dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

      relu<T> <<<gridDim, blockDim>>>(a->data(), b->data(), N);

      cudaDeviceSynchronize();
    }
};


template<typename T>
class MatAdd: public Op<T> {
  public:
    MatAdd(std::vector<Tensor<T>*> tensors) : Op<T>(tensors) {
      if (tensors.size() != 3) {
        throw std::invalid_argument("MatAdd requires exactly 3 tensors");
      }
    }

    void forward() {
        Tensor<T> *t_a = this->_tensors[0];
        Tensor<T> *t_b = this->_tensors[1];
        Tensor<T> *t_c = this->_tensors[2];

        int J = t_a->shape()[0];
        int K = t_a->shape()[1];

        int N = J * K;

        dim3 blockDim(BLOCK_SIZE);
        dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

        matAdd<float> <<<gridDim, blockDim>>>(t_a->data(), t_b->data(), t_c->data(), N);

        cudaDeviceSynchronize();

    }
};

template<typename T>
class BiasAdd: public Op<T> {
  public:
    BiasAdd(std::vector<Tensor<T>*> tensors) : Op<T>(tensors) {
      if(tensors.size() != 3) {
        throw std::invalid_argument("BiasAdd requires exactly 3 tensors");
      }
    }


    void forward() {

      Tensor<T> *a = this->_tensors[0];
      Tensor<T> *b = this->_tensors[1];
      Tensor<T> *c = this->_tensors[2];

      int J = a->shape()[0];
      int K = a->shape()[1];

      dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
      dim3 gridDim((K + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (J + BLOCK_SIZE - 1) / BLOCK_SIZE);

      biasAdd<T><<<gridDim, blockDim>>>(a->data(), b->data(), c->data(), J , K);

      cudaDeviceSynchronize();
    }
};

template<typename T>
class Gemm: public Op<T> {
  public:
    Gemm(std::vector<Tensor<T>*> tensors) : Op<T>(tensors) {
      if (tensors.size() != 3) {
        throw std::invalid_argument("Gemm requires exactly 3 tensors");
      }
    }

    void forward() {
      Tensor<T> *a = this->_tensors[0];
      Tensor<T> *b = this->_tensors[1];
      Tensor<T> *c = this->_tensors[2];

      int J = a->shape()[0];
      int K = a->shape()[1];
      int M = b->shape()[0];
      int N = b->shape()[1];

      dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
      dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                  (J + BLOCK_SIZE - 1) / BLOCK_SIZE);

      // gemm2<false, false, T><<<gridDim, blockDim>>>(a->data(), b->data(), c->data(), J, K, M, N);
      gemm<T><<<gridDim, blockDim>>>(a->data(), b->data(), c->data(), J, K, M, N);

      cudaDeviceSynchronize();
    }

    void backward() {
      Tensor<T> *a = this->_tensors[0];
      Tensor<T> *b = this->_tensors[1];
      Tensor<T> *c = this->_tensors[2];

      int J1 = c->shape()[0];
      int K1 = c->shape()[1];
      int M1 = b->shape()[0];
      int N1 = b->shape()[1];

      dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
      dim3 gridDim((M1 + BLOCK_SIZE - 1) / BLOCK_SIZE,
                  (J1 + BLOCK_SIZE - 1) / BLOCK_SIZE);

      gemm2<false, true, T><<<gridDim, blockDim>>>(c->data(), b->data(), a->grad(), J1, K1, M1, N1);

      cudaDeviceSynchronize();
    }
};