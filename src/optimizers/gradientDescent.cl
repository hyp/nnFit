
typedef float Scalar;

kernel void gradientDescent(global Scalar *weights, global Scalar *gradients, const float learningRate) {
    size_t i = get_global_id(0);
    weights[i] = weights[i] - learningRate*gradients[i];
}