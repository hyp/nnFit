#pragma once

#include "layer.h"

namespace nnFit {

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
        Kernel evaluateClassification;
        
        Specialization(Device &device, Program &program);
    };
    Specialization floatKernels;
    
    NNContext(Device &device);
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
    
    Network &inputLayer(size_t size);
    
    Network &add(std::unique_ptr<Layer> layer) {
        layers.push_back(std::move(layer));
        return *this;
    }
    
    std::vector<std::pair<Vector*, Vector*>> weightsAndGradients();
    
    Layer &lastLayer() const {
        return *layers.back();
    }
    
    void init(uint32_t seed);
    void init();
    void dump();
    void tune();
    const Vector &predict(const Vector &input);
    const Vector &feedforward(const Vector &input);
    void backpropagate();
private:
    Network(const Network&) = delete;
    Device &dev;
    NNContext ctx;
    size_t inputLayerSize;
    const Vector *networkInput;
    std::vector<std::unique_ptr<Layer>> layers;
};

} // namespace nnFit