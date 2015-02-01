#pragma once 

#include "core/random.h"

namespace nnFit {
    
class NNContext;
    
class DropoutLayer {
public:
    DropoutLayer(Device &device, size_t size, float activationProbability, size_t parallelisationFactor = 1);
    
    const Vector &predict(NNContext &ctx, const Vector &input);
    const Vector &feedforward(NNContext &ctx, const Vector &input);
private:
    RandomGenerator gen;
    float activationProbability;
};
    
} // namespace nnFit