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
    virtual void backward() {}
};

template<typename T>
class Linear : public Op<T> {
  public:
    Linear(std::vector<Tensor<T>*> tensors) : Op<T>(tensors) {
      if (tensors.size() != 4) {
        throw std::invalid_argument("Linear requires exactly 4 tensors");
      }
    }
    
    /*
      y = wx + b
    */
    void forward() {
      Tensor<T> *x = this->_tensors[0];
      Tensor<T> *w = this->_tensors[1];
      Tensor<T> *b = this->_tensors[2];
      Tensor<T> *y = this->_tensors[3];

      int J = x->shape()[0];
      int K = x->shape()[1];
      int M = w->shape()[0];
      int N = w->shape()[1];

      dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
      dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                  (J + BLOCK_SIZE - 1) / BLOCK_SIZE);

      linear<T><<<gridDim, blockDim>>>(x->data(), w->data(), b->data(), y->data(), J, K, M, N);
      
      cudaDeviceSynchronize();
    }

    void backward() {
      Tensor<T> *x = this->_tensors[0];
      Tensor<T> *w = this->_tensors[1];
      Tensor<T> *b = this->_tensors[2];
      Tensor<T> *y = this->_tensors[3];
      
      int J1 = x->shape()[0];
      int K1 = x->shape()[1];
      int M1 = y->shape()[0];
      int N1 = y->shape()[1];

      dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
      dim3 aGridDim((N1 + BLOCK_SIZE - 1) / BLOCK_SIZE,
                  (K1 + BLOCK_SIZE - 1) / BLOCK_SIZE);

      gemm2<true, false, T><<<aGridDim, blockDim>>>(x->data(), y->grad(), w->grad(), J1, K1, M1, N1);

      int J2 = y->shape()[0];
      int K2 = y->shape()[1];
      int M2 = w->shape()[0];
      int N2 = w->shape()[1];

      dim3 bGridDim((M2 + BLOCK_SIZE - 1) / BLOCK_SIZE,
                  (J2 + BLOCK_SIZE - 1) / BLOCK_SIZE);

      gemm2<false, true, T><<<bGridDim, blockDim>>>(y->grad(), w->data(), x->grad(), J2, K2, M2, N2);

      dim3 sumGridDim(N1);
      dim3 sumBlockDim(BLOCK_SIZE);
      
      sumCols<T><<<sumGridDim, sumBlockDim>>>(y->grad(), b->grad(), M1, N1);

      cudaDeviceSynchronize();
    }
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
  private:
    Tensor<uint32_t> *_labels;
  public:
    Softmax(std::vector<Tensor<T>*> tensors, Tensor<uint32_t> *labels = nullptr) : Op<T>(tensors) {
      if (tensors.size() != 2) {
        throw std::invalid_argument("Softmax requires exactly 2 tensors");
      }

      _labels = labels;
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

    void backward() {
      Tensor<T> *a = this->_tensors[0];

      int J = a->shape()[0];
      int K = a->shape()[1];

      dim3 blockDim(BLOCK_SIZE);
      dim3 gridDim(J);

      if(_labels == nullptr) {
        throw std::runtime_error("Labels tensor is not set for softmax backward pass");
      }

      gradient<T> <<<gridDim, blockDim>>>(a->data(), _labels->data(), a->grad(), J, K);

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

    void backward() {
      Tensor<T> *a = this->_tensors[0];
      Tensor<T> *b = this->_tensors[1];

      int N = a->n_elem();
      dim3 blockDim(BLOCK_SIZE);
      dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
      
      // USE b->grad() FOR BACKWARD PASS
      relu_backward<T> <<<gridDim, blockDim>>>(a->data(), b->data(), a->grad(), N);

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
        Tensor<T> *a = this->_tensors[0];
        Tensor<T> *b = this->_tensors[1];
        Tensor<T> *c = this->_tensors[2];

        int J = a->shape()[0];
        int K = a->shape()[1];

        int N = J * K;

        dim3 blockDim(BLOCK_SIZE);
        dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

        matAdd<float> <<<gridDim, blockDim>>>(a->data(), b->data(), c->data(), N);

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

      gemm2<false, false, T><<<gridDim, blockDim>>>(a->data(), b->data(), c->data(), J, K, M, N);

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
      dim3 aGridDim((M1 + BLOCK_SIZE - 1) / BLOCK_SIZE,
                  (J1 + BLOCK_SIZE - 1) / BLOCK_SIZE);

      gemm2<false, true, T><<<aGridDim, blockDim>>>(c->data(), b->data(), a->grad(), J1, K1, M1, N1);

      int J2 = a->shape()[0];
      int K2 = a->shape()[1];
      int M2 = c->shape()[0];
      int N2 = c->shape()[1];

      dim3 bGridDim((N2 + BLOCK_SIZE - 1) / BLOCK_SIZE,
                  (K2 + BLOCK_SIZE - 1) / BLOCK_SIZE);

      gemm2<true, false, T><<<bGridDim, blockDim>>>(a->data(), c->data(), b->grad(), J2, K2, M2, N2);

      cudaDeviceSynchronize();
    }
};