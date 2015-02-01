#include "dropout.h"

using namespace nnFit;

DropoutLayer::DropoutLayer(Device &device, size_t size, float activationProbability, size_t parallelisationFactor) : gen(device, size * parallelisationFactor), activationProbability(activationProbability) {
}

const Vector &DropoutLayer::predict(NNContext &ctx, const Vector &input) {
    return input;
}

const Vector &DropoutLayer::feedforward(NNContext &ctx, const Vector &input) {
    gen.invertedDropout(input, activationProbability);
    return input;
}
