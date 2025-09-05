#pragma once
#include <vector>
#include "kernels.h"
#include <random>


template <typename T>
class Tensor {
  private:
    T* _data;
    T* _grad;
    std::vector<int> _shape;
    int _n_elem;
    size_t _size;
    bool _weight_decay;

  public:
    Tensor(std::vector<int> shape, T* data, bool weight_decay = false) {
      _shape = shape;
      
      _n_elem = 1;
      for(int i : shape) _n_elem *= i;
      _size = _n_elem * sizeof(T);

      cudaMalloc((void**)&_data, _size);
      if(data == nullptr) {
        cudaMemset(_data, 0, _size);
      } else {
        cudaMemcpy(_data, data, _size, cudaMemcpyHostToDevice);
      }

      cudaMalloc((void**)&_grad, _size);
      cudaMemset(_grad, 0, _size);

      _weight_decay = weight_decay;
    }
    Tensor(std::vector<int> shape, bool weight_decay = false, bool random_init = false) {
      _shape = shape;
      
      _n_elem = 1;
      for(int i : shape) _n_elem *= i;
      _size = _n_elem * sizeof(T);

      cudaMalloc((void**)&_data, _size);

      cudaMalloc((void**)&_grad, _size);
      cudaMemset(_grad, 0, _size);

      if(random_init) {
        //TODO: He initialization
        //placeholder code
        T* w = (T*)malloc(_size);

        static thread_local std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<T> dist((T)-0.1, (T)0.1);
        for(int i = 0; i < _n_elem; i++) w[i] = dist(rng);

        cudaMemcpy(_data, w, _size, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        free(w);

      } else {
        cudaMemset(_data, 0, _size);
      }

      _weight_decay = weight_decay;
    }

    T* to_host() {
      T* host_data = (T*)malloc(_size);
      cudaMemcpy(host_data, _data, _size, cudaMemcpyDeviceToHost);
      cudaDeviceSynchronize();
      return host_data;
    }

    T* grad_to_host() {
      T* host_grad = (T*)malloc(_size);
      cudaMemcpy(host_grad, _grad, _size, cudaMemcpyDeviceToHost);
      cudaDeviceSynchronize();
      return host_grad;
    }

    bool weight_decay() {
      return _weight_decay;
    }

    size_t n_bytes() {
      return _size;
    }

    T* grad() {
      return _grad;
    }

    T* data() {
      return _data;
    }

    std::vector<int> shape() {
      return _shape;
    }

    int n_elem() {
      return _n_elem;
    }

    void zero_grad() {
      cudaMemset(_grad, 0, _size);
      cudaDeviceSynchronize();
    }

    void set_data(T* data) {
      cudaMemcpy(_data, data, _size, cudaMemcpyHostToDevice);
      cudaDeviceSynchronize();
    }

    ~Tensor() {
      cudaFree(_data);
      cudaFree(_grad);
    }
};