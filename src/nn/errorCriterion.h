#pragma once

#include <memory>
#include "core/vector.h"

namespace nnFit {

class NNContext;
class Layer;

class ErrorCriterion {
public:
    virtual const Vector &computeError(NNContext &ctx, const Vector &prediction, const Vector &expectedOutput, Vector &accumulatedErrors) = 0;
    
    virtual void computeLayerError(NNContext &ctx, const Vector &prediction, const Vector &expectedOutput, const Vector &derivative, const Vector &errorTerm) = 0;
};

// Mean squared error criterion.
// error(y,p) = sum((y - p) .^ 2)
class MSECriterion : public ErrorCriterion {
public:
  
    const Vector &computeError(NNContext &ctx, const Vector &prediction, const Vector &expectedOutput, Vector &accumulatedErrors) override;

    void computeLayerError(NNContext &ctx, const Vector &prediction, const Vector &expectedOutput, const Vector &derivative, const Vector &errorTerm) override;
};

// Cross entropy error criterion.
// error(y, p) = sum(- y .* log(p) - (1 - y).*log(1 - p))
class CrossEntropyCriterion : public ErrorCriterion {
public:
    const Vector &computeError(NNContext &ctx, const Vector &prediction, const Vector &expectedOutput, Vector &accumulatedErrors) override;
    
    void computeLayerError(NNContext &ctx, const Vector &prediction, const Vector &expectedOutput, const Vector &derivative, const Vector &errorTerm) override;
};

} // namespace nnFit