#include "pipeline.h"
#include "op.h"
#include "tensor.h"

// template<typename T>
// void classify_nn_forward(T* data, std::vector<int> shape, uint32_t* labels, int num_classes) {
//     Tensor<T> input_tensor(data, shape);
//     Tensor<uint32_t> label_tensor(labels, {shape[0]});
//     new Tensor<T>(nullptr, {shape[0], num_classes});

//     std::vector<Ops<T>*> ops;
//     ops.push_back(new Gemm<T>({&input_tensor, &label_tensor, }));
// }


void forward_pass(float* X, float* W1, float* b1, float* W2, float* b2, float* out,
    int J, int K, int M, int N) {
    Tensor<float>* x = new Tensor<float>(X, {J, K});
    Tensor<float>* w1 = new Tensor<float>(W1, {K, M});
    Tensor<float>* b1 = new Tensor<float>(b1, {M});
    Tensor<float>* w2 = new Tensor<float>(W2, {M, N});
    Tensor<float>* b2 = new Tensor<float>(b2, {N});
    Tensor<float>* h = new Tensor<float>({J, M});
    Tensor<float>* h_relu = new Tensor<float>({J, M});
    Tensor<float>* y = new Tensor<float>({J, N});
    Tensor<float>* y_softmax = new Tensor<float>({J, N});

    // layer 1: h = relu(x @ w1 + b1)
    Gemm<float> gemm1({x, w1, h});
    gemm1.forward();
    BiasAdd<float> add1({h, b1});
    add1.forward();
    Relu<float> relu({h, h_relu});
    relu.forward();

    // layer 2: y = softmax(h_relu @ w2 + b2)
    Gemm<float> gemm2({h_relu, w2, y});
    gemm2.forward();
    BiasAdd<float> add2({y, b2});
    add2.forward();
    Softmax<float> sm({y, y_softmax});
    sm.forward();

    float* result = y_softmax->to_host();
    memcpy(out, result, J * N * sizeof(float));
    free(result);

    delete x; delete w1; delete b1; delete w2; delete b2;
    delete h; delete h_relu; delete y; delete y_softmax;
}