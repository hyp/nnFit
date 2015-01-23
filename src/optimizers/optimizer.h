#pragma once

#include "core/vector.h"

namespace nnFit {

class Optimizer {
public:
    virtual void optimize(const std::vector<std::pair<Vector*, Vector*>> &weightsAndGradients, size_t trainingExamples) = 0;
};
    
} // namespace nnFit
