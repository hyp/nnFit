#pragma once 

#include "core/random.h"
#include "abstractLayer.h"

namespace nnFit {
    
class NNContext;
    
class DropoutLayer: public AbstractLayer {
public:
    DropoutLayer(Device &device, size_t size, float activationProbability, size_t parallelisationFactor = 1);
    
    bool backpropagates() const override {
        return false;
    }
    
    const Vector &predict(NNContext &ctx, const Vector &input) override;
    
    const Vector &feedforward(NNContext &ctx, const Vector &input) override;
    
    const Vector &backpropagate(NNContext &ctx, const Vector &expectedOutput, const ErrorCriterion &criterion, bool backpropagateDown) override;
    
    const Vector &backpropagate(NNContext &ctx, const Vector &errorInput, bool backpropagateDown) override;
private:
    RandomGenerator gen;
    float activationProbability;
};
    
} // namespace nnFit