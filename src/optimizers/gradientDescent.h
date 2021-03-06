#pragma once

#include "optimizer.h"

namespace nnFit {

class GradientDescent: public Optimizer {
public:
    GradientDescent(Device &device, float learningRate);
    
    void optimize(const std::vector<std::pair<const Vector*, const Vector*>> &weightsAndGradients, size_t trainingExamples) override;
    
private:
    Device &device;
    Kernel kernel;
    float learningRate;
};
    
class MomentumGradientDescent: public Optimizer {
public:
    MomentumGradientDescent(Device &device, float learningRate, float momentumDecay);
    
    void optimize(const std::vector<std::pair<const Vector*, const Vector*>> &weightsAndGradients, size_t trainingExamples) override;
private:
    Device &device;
    Kernel kernel;
    std::vector<Vector> velocities;
    float learningRate, momentumDecay;
};

} // namespace nnFit