#pragma once

#include "nn/layer.h"

namespace nnFit {
    
class RecurrentLayer {
public:
    RecurrentLayer(Device &device, size_t neuronCount, size_t inputCount, TransferFunction transferFunction);
    
    const Matrix &neuronWeights() const {
        return layer.neuronWeights();
    }
    const Vector &neuronBiases() const {
        return layer.neuronBiases();
    }
    const Vector &initalActivation() const {
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
