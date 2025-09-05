#include "pipeline.h"
#include "op.h"
#include "tensor.h"


extern "C" void big_test(float *X, int n, int in_dim, int out_dim, int hidden_dim, int hidden_layers, 
    uint32_t *labels, float *out, float lr, int epochs) {
    
    Net<float> nn = Net<float>(n, in_dim, out_dim, hidden_dim, hidden_layers);

    //"training loop"
    for(int i = 0; i < epochs; i ++) {
        nn.forward(X, labels, n, in_dim); 
        nn.backward();
        nn.update(lr);
        nn.zero_grad();
    }

    float *result = nn.forward(X, labels, n, in_dim);
    memcpy(out, result, n * out_dim * sizeof(float));
    free(result);
}