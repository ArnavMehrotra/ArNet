#include "pipeline.h"
#include "op.h"
#include "tensor.h"


extern "C" void test_layers(float* X, float* W1, float* B1, float* W2, float* B2, uint32_t* labels, float* out,
    int J, int K, int M, int N, float lr) {
    
    Tensor<float> *t_x = new Tensor<float>({J, K}, X);
    Tensor<float> *t_w1 = new Tensor<float>({K, M}, W1, true);
    Tensor<float> *t_b1 = new Tensor<float>({M}, B1, true);
    Tensor<float> *t_z = new Tensor<float>({J, M});
    Tensor<float> *t_z_relu = new Tensor<float>({J, M});

    Tensor<float> *t_w2 = new Tensor<float>({M, N}, W2, true);
    Tensor<float> *t_b2 = new Tensor<float>({N}, B2, true);
    Tensor<float> *t_y = new Tensor<float>({J, N});
    Tensor<float> *t_y_softmax = new Tensor<float>({J, N});
    
    std::vector<Op<float>*> ops = {
        new Linear<float>({t_x, t_w1, t_b1, t_z}),
        new Relu<float> ({t_z, t_z_relu}),
        new Linear<float>({t_z_relu, t_w2, t_b2, t_y}),
        new Softmax<float> ({t_y, t_y_softmax}, new Tensor<uint32_t>({J}, labels))
    };

    Net nn = Net(ops);

    //"training loop"
    nn.forward(); 
    nn.backward();
    nn.update(lr);


    float *result = t_y->grad_to_host();
    memcpy(out, result, J * N * sizeof(float));
    free(result);

    float* dW2 = t_w2->to_host();
    memcpy(W2, dW2, M * N * sizeof(float));
    free(dW2);

    float *db2 = t_b2->to_host();
    memcpy(B2, db2, N * sizeof(float));
    free(db2);

    float *dW1 = t_w1->to_host();
    memcpy(W1, dW1, K * M * sizeof(float));
    free(dW1);

    float *db1 = t_b1->to_host();
    memcpy(B1, db1, M * sizeof(float));
    free(db1);

    nn.zero_grad();

    for (Op<float> *op : ops) {
        delete op;
    }

    delete t_x; delete t_w1; delete t_b1; delete t_z; delete t_z_relu;
    delete t_w2; delete t_b2; delete t_y; delete t_y_softmax;

}