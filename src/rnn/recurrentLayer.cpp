#include "RecurrentLayer.h"

using namespace nnFit;

RecurrentLayer::RecurrentLayer(Device &device, size_t neuronCount, size_t inputCount, TransferFunction transferFunction) : layer(device, neuronCount, inputCount + neuronCount, transferFunction), initialActivations(device, neuronCount), currentSequenceLength(0) { }

void RecurrentLayer::init(uint32_t seed) {
    layer.init(seed);
}

void RecurrentLayer::tune() {
    layer.tune();
}

void RecurrentLayer::reset() {
    currentSequenceLength = 0;
}

void RecurrentLayer::unroll(size_t length) {
    auto &device = initialActivations.device();
    for (size_t i = 0; i < length; ++i) {
        unrolledState.push_back(UnrolledState(device, layer.neuronCount(), layer.inputCount()));
    }
}

const Vector &RecurrentLayer::mergeInput(const Vector &input) {
    assert(input.size() == (layer.inputCount() - layer.neuronCount()));
    
    const auto &dest = unrolledState[currentSequenceLength].input;
    input.copy(dest.slice(0, input.size()));
    if (currentSequenceLength == 0)
        initialActivations.copy(dest.slice(input.size()));
    else
        layer.activation().copy(dest.slice(input.size()));
    ++currentSequenceLength;
    return dest;
}

const Vector &RecurrentLayer::predict(NNContext &ctx, const Vector &input) {
    return layer.predict(ctx, mergeInput(input));
}

const Vector &RecurrentLayer::feedforward(NNContext &ctx, const Vector &input) {
    const auto &derivatives = unrolledState[currentSequenceLength].derivatives;
    // activation = f(Wx + b)
    // derivative = f'(Wx + b)
    return layer.transferFunction().apply(ctx, layer.predictLinear(ctx, mergeInput(input)), derivatives);
}

RecurrentLayer::UnrolledState::UnrolledState(Device &device, size_t neuronCount, size_t inputSize) : input(device, inputSize), derivatives(device, neuronCount) {
}

RecurrentLayer::UnrolledState::UnrolledState(UnrolledState &&other) : input(std::move(other.input)), derivatives(std::move(other.derivatives)) {
}

