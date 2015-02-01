#pragma once

#include "core/vector.h"

namespace nnFit {
    
class NNContext;
class ErrorCriterion;
    
class AbstractLayer {
public:
    
    virtual ~AbstractLayer() { }
    
    virtual void init(uint32_t seed) { }
    virtual void dump() { }
    virtual void tune() { }
    
    // Return true when a layer needs error backpropagated from the layers that follow it.
    virtual bool backpropagates() const {
        return true;
    }
    
    virtual const Vector &predict(NNContext &ctx, const Vector &input) = 0;
    virtual const Vector &feedforward(NNContext &ctx, const Vector &input) = 0;
    virtual const Vector &backpropagate(NNContext &ctx, const Vector &expectedOutput, const ErrorCriterion &criterion, bool backpropagateDown = true) = 0;
    virtual const Vector &backpropagate(NNContext &ctx, const Vector &errorInput, bool backpropagateDown = true) = 0;
    
    virtual void collectWeightsAndGradients(std::vector<std::pair<const Vector*, const Vector*>> &weightsAndGradients) { }
    virtual void accumulateGradients(NNContext &ctx) { }
};

} // namespace nnFit