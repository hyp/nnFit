#pragma once

#include <functional>
#include "network.h"
#include "core/dataset.h"

namespace nnFit {

class Optimizer;
class ErrorCriterion;

class Trainer {
public:
    std::function<void (size_t, size_t, float)> afterIteration;
    bool reshuffleIndices;
    
    Trainer(Network &network, ErrorCriterion &criterion, Dataset &data);
    
    void gradientDescent(Optimizer &opt, size_t iterations);
    void miniBatchGradientDescent(Optimizer &opt, size_t iterations, size_t miniBatchSize);
    
    size_t numberOfTrainingExamples() const {
        return trainingExampleCount;
    }
    
private:
    void train(Optimizer &opt, size_t iterations, size_t miniBatchSize);
    Network &network;
    ErrorCriterion &criterion;
    Dataset &data;
    size_t trainingExampleCount;
    Vector input;
    Vector output;
};

} // namespace nnFit