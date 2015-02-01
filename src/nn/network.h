#pragma once

#include "layer.h"

namespace nnFit {
    
class ErrorCriterion;

class NNContext {
public:
    struct Specialization {
        Kernel sigmoidPredict;
        Kernel sigmoidFeedforward;
        Kernel reluPredict;
        Kernel reluFeedforward;
        Kernel meanSquaredError;
        Kernel crossEntropyError;
        Kernel computeMSELayerError;
        Kernel computeCrossEntropyLayerError;
        Kernel computeError;
        Kernel computeWeightGradients;
        Kernel computeWeightGradients4;
        Kernel computeWeightGradientsParallel;
        Kernel computeWeightGradients4Parallel;
        Kernel computeBiasGradients;
        Kernel evaluateClassification;
        
        Specialization(Device &device, Program &program);
    };
    Specialization floatKernels;
    
    NNContext(Device &device);
    
    CommandQueue &queue() const {
        return queue_;
    }
private:
    CommandQueue &queue_;
};

class Network {
public:
    Network(Device &device);
    
    NNContext &context() {
        return ctx;
    }
    
    Device &device() const {
        return dev;
    }
    
    Network &add(std::unique_ptr<AbstractLayer> layer);
    
    std::vector<std::pair<const Vector*, const Vector*>> weightsAndGradients();
    
    void init(uint32_t seed);
    void init();
    void dump();
    void tune();
    const Vector &predict(const Vector &input);
    const Vector &feedforward(const Vector &input);
    void backpropagate(const Vector &expectedOutput, const ErrorCriterion &criterion);
private:
    Network(const Network&) = delete;
    Device &dev;
    NNContext ctx;
    std::vector<std::unique_ptr<AbstractLayer>> layers;
};

} // namespace nnFit