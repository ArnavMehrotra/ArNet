#pragma once
#include <vector>
#include <cassert>
#include "op.h"
#include "tensor.h"

#define BATCH_SIZE 1024

template<typename T>
class Net {
    private:
        std::vector<Op<T>*> _ops;

        Linear<T>* create_linear(Tensor <T>* input, int in_dim, int out_dim) {
            int n = input->shape()[0];
            assert(input->shape()[1] == in_dim);
            Tensor<T> *w = new Tensor<T>({in_dim, out_dim}, true, true);
            Tensor<T> *b = new Tensor<T>({out_dim}, true, true);
            Tensor<T> *z = new Tensor<T>({n, out_dim});

            return new Linear<T>({input, w, b, z});
        }


    public:
        Net(std::vector<Op<T>*> ops) {
            _ops = ops;
        }
        Net(int n, int in_dim, int out_dim, int hidden_dim = 0, int hidden_layers = 0) {
            if ((hidden_dim == 0 ) != (hidden_layers == 0)) {
                throw std::invalid_argument("hidden dims > 0 requires hidden layers > 0 and vice versa"); 
            }
            
            Tensor<T> *input_tensor = new Tensor<T>({n, in_dim});
            _ops.push_back(create_linear(input_tensor, in_dim, hidden_layers > 0 ? hidden_dim : out_dim));
            
            if(hidden_layers > 0) {
                //add first activation for hidden layers
                Tensor<T> *a = _ops.back()->tensors().back();
                Tensor<T> *relu_out = new Tensor<T>({n, hidden_dim});
                _ops.push_back(new Relu<T>({a, relu_out}));
                a = relu_out;
                
                //add remaining hidden layers and activations
                for(int i = 0; i < hidden_layers - 1; i++) {
                    _ops.push_back(create_linear(a, hidden_dim, hidden_dim));
                    Tensor<T> *z = _ops.back()->tensors().back();
                    Tensor<T> *a_next = new Tensor<T>({n, hidden_dim});
                    _ops.push_back(new Relu<T>({z, a_next}));
                    a = a_next;
                }
                
                //final hidden linear layer before softmax
                _ops.push_back(create_linear(a, hidden_dim, out_dim));
            }
            
            return;
            //softmax for logits
            Tensor<T> *y = _ops.back()->tensors().back();
            Tensor<T> *logits = new Tensor<T>({n, out_dim});
            _ops.push_back(new Softmax<T>({y, logits}));
        }

        ~Net() {
            for (Op<T>* op : _ops) {
                for(Tensor<T>* tensor : op->tensors()) {
                    delete tensor;
                }
                delete op;
            }
            _ops.clear();
        }

        void load_data(T* data, uint32_t *labels, int n_samples, int in_dim) {
            Tensor<T> *in_tensor = _ops[0]->tensors()[0];
            assert(in_tensor->shape()[0] == in_dim && in_tensor->shape()[1] == n_samples);
            in_tensor->set_data(data);
            ((Softmax<T>*) _ops.back())->set_labels(labels);
        } 

        void zero_grad() {
            for (Op<T>* op : _ops) {
                op->zero_grad();
            }
        }

        void update(T lr) {
            for (Op<T>* op : _ops) {
                op->update(lr);
            }
        }

        T* forward(T* data, uint32_t *labels, int n_samples, int in_dim) {
            load_data(data, labels, n_samples, in_dim);
            for (Op<T>* op : _ops) {
                op->forward();
            }

            return _ops.back()->tensors().back()->to_host();
        }

        void forward() {
            for (Op<T>* op : _ops) {
                op->forward();
            }
        }

        void backward() {
            for (auto it = _ops.rbegin(); it != _ops.rend(); ++it) {
                (*it)->backward();
            }
        }
};