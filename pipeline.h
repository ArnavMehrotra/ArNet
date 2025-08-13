#pragma once
#include <vector>
#include "op.h"
#include "tensor.h"


template<typename T>
class Net {
    private:
        std::vector<Op<T>*> _ops;
    public:
        Net(std::vector<Op<T>*> ops) {
            _ops = ops;
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