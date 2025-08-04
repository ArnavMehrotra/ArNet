#pragma once
#include <vector>
#include "kernels.h"


template <typename T>
class Tensor {
  private:
    T* _data;
    std::vector<int> _shape;
    int _n_elem;
    size_t _size;

  public:
    Tensor(T* data, std::vector<int> shape){
      _shape = shape;
      
      _n_elem = 1;
      for(int i : shape) _n_elem *= i;
      _size = _n_elem * sizeof(T);
      cudaMalloc((void**)&_data, _size);
      cudaMemcpy(_data, data, _size, cudaMemcpyHostToDevice);
    }

    T* to_host() {
      T* host_data = (T*)malloc(_size);
      cudaMemcpy(host_data, _data, _size, cudaMemcpyDeviceToHost);
      cudaDeviceSynchronize();
      return host_data;
    }

    size_t n_bytes() {
      return _size;
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

    ~Tensor() {
      cudaFree(_data);
    }
};