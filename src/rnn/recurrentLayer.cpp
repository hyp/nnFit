#include "RecurrentLayer.h"

using namespace nnFit;

RecurrentLayer::RecurrentLayer(Device &device, size_t neuronCount, size_t inputCount, TransferFunction transferFunction) : layer(device, neuronCount, inputCount + neuronCount, transferFunction), inputAndPreviousActivation(device, inputCount + neuronCount), initialActivations(device, neuronCount), currentSequenceLength(0) { }

void RecurrentLayer::init(uint32_t seed) {
    layer.init(seed);
}

void RecurrentLayer::tune() {
    layer.tune();
}

void RecurrentLayer::reset() {
    currentSequenceLength = 0;
}

const Vector &RecurrentLayer::mergeInput(const Vector &input) {
    assert(input.size() == (layer.inputCount() - layer.neuronCount()));
    
    input.copy(inputAndPreviousActivation.slice(0, input.size()));
    if (currentSequenceLength == 0)
        initialActivations.copy(inputAndPreviousActivation.slice(input.size()));
    else
        layer.activation().copy(inputAndPreviousActivation.slice(input.size()));
    ++currentSequenceLength;
    return inputAndPreviousActivation;
}

const Vector &RecurrentLayer::predict(NNContext &ctx, const Vector &input) {
    return layer.predict(ctx, mergeInput(input));
}

