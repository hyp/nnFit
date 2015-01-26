#pragma once

#include "nn/layer.h"

namespace nnFit {
    
class RecurrentLayer {
public:
    RecurrentLayer(Device &device, size_t neuronCount, size_t inputCount, TransferFunction transferFunction);
    
    Matrix &neuronWeights() {
        return layer.neuronWeights();
    }
    Vector &neuronBiases() {
        return layer.neuronBiases();
    }
    Vector &initalActivation() {
        return initialActivations;
    }
    
    const Vector &mergeInput(const Vector &input);
    
    void init(uint32_t seed);
    void tune();
    
    void reset();
    const Vector &predict(NNContext &ctx, const Vector &input);
private:
    RecurrentLayer(const RecurrentLayer&) = delete;
    Layer layer;
    Vector inputAndPreviousActivation;
    Vector initialActivations;
    size_t currentSequenceLength;
};

} // namespace nnFit
