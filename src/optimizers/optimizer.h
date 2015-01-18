#pragma once

#include "core/vector.h"

namespace nnFit {

class Optimizer {
public:
    virtual void optimize(float error, const std::vector<std::pair<Vector*, Vector*>> &weightsAndGradients) = 0;
};
    
} // namespace nnFit
