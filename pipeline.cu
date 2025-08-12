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

extern "C" void test_layers(float* X, float* W1, float* B1, float* W2, float* B2, float* out,
    int J, int K, int M, int N) {
    
    Tensor<float> *t_x = new Tensor<float>(X, {J, K});
    Tensor<float> *t_w1 = new Tensor<float>(W1, {K, M});
    Tensor<float> *t_b1 = new Tensor<float>(B1, {M});
    Tensor<float> *t_z = new Tensor<float>({J, M});
    Tensor<float> *t_z_relu = new Tensor<float>({J, M});

    Tensor<float> *t_w2 = new Tensor<float>(W2, {M, N});
    Tensor<float> *t_b2 = new Tensor<float>(B2, {N});
    Tensor<float> *t_y = new Tensor<float>({J, N});
    Tensor<float> *t_y_softmax = new Tensor<float>({J, N});
    
    std::vector<Op<float>*> ops = {
        new Linear<float>({t_x, t_w1, t_b1, t_z}),
        new Relu<float> ({t_z, t_z_relu}),
        new Linear<float>({t_z_relu, t_w2, t_b2, t_y}),
        new Softmax<float> ({t_y, t_y_softmax})
    };

    Net nn = Net(ops);
    nn.forward();
    
    // float *result = t_z->to_host();
    // memcpy(out, result, J * M * sizeof(float));
    // free(result);

    float *result = t_y_softmax->to_host();
    memcpy(out, result, J * N * sizeof(float));
    free(result);

    for (Op<float>* op : ops) {
        delete op;
    }

    delete t_x; delete t_w1; delete t_b1; delete t_z; delete t_z_relu;
    delete t_w2; delete t_b2; delete t_y; delete t_y_softmax;


}

extern "C" void forward_pass(float* X, float* W1, float* B1, float* W2, float* B2, float* out,
    int J, int K, int M, int N) {

    Tensor<float>* t_x = new Tensor<float>(X, {J, K});
    Tensor<float>* t_w1 = new Tensor<float>(W1, {K, M});
    Tensor<float>* t_b1 = new Tensor<float>(B1, {M});

    Tensor<float>* t_h = new Tensor<float>({J, M});
    Tensor<float>* t_z = new Tensor<float>({J, M});
    Tensor<float>* t_z_relu = new Tensor<float>({J, M});

    Tensor<float>* t_w2 = new Tensor<float>(W2, {M, N});
    Tensor<float>* t_b2 = new Tensor<float>(B2, {N});
    Tensor<float>* t_r = new Tensor<float>({J, N});
    Tensor<float>* t_y = new Tensor<float>({J, N});
    Tensor<float>* t_y_softmax = new Tensor<float>({J, N});
    
    std::vector<Op<float>*> ops = {
        new Gemm<float>({t_x, t_w1, t_h}),
        new BiasAdd<float>({t_h, t_b1, t_z}),
        new Relu<float>({t_z, t_z_relu}),
        new Gemm<float>({t_z_relu, t_w2, t_r}),
        new BiasAdd<float>({t_r, t_b2, t_y}),
        new Softmax<float> ({t_y, t_y_softmax}),
    };

    Net nn = Net(ops);
    nn.forward();

    float *result = t_y_softmax->to_host();
    memcpy(out, result, J * N * sizeof(float));
    free(result);

    delete t_x; delete t_w1; delete t_b1; delete t_h; delete t_z;
    delete t_z_relu; delete t_w2; delete t_r; delete t_b2; delete t_y;
    delete t_y_softmax;

    for (Op<float>* op : ops) {
        delete op;
    }
}