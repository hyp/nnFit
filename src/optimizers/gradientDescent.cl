
typedef float Scalar;

kernel void gradientDescent(global Scalar *weights, global Scalar *gradients, const float learningRate) {
    size_t i = get_global_id(0);
    weights[i] = weights[i] - learningRate*gradients[i];
}

kernel void momentumGradientDescent(global Scalar *weights, global Scalar *gradients, global Scalar *velocity, const float learningRate, const float momentumDecay) {
    size_t i = get_global_id(0);
    Scalar v = velocity[i]*momentumDecay - learningRate*gradients[i];
    weights[i] = weights[i] + v;
    velocity[i] = v;
}