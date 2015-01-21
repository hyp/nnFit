#pragma once

#include "optimizer.h"

namespace nnFit {

class GradientDescent: public Optimizer {
public:
    GradientDescent(Device &device, float learningRate);
    
    void optimize(const std::vector<std::pair<Vector*, Vector*>> &weightsAndGradients) override;
    
private:
    Device &device;
    Kernel kernel;
    float learningRate;
};

} // namespace nnFit