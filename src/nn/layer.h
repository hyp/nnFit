#pragma once

#include "core/vector.h"
#include "transferFunction.h"

namespace nnFit {

class NNContext;

class Layer {
public:
    
    Layer(Device &device, size_t neuronCount, size_t inputCount, TransferFunction transferFunction = TransferFunction::Linear, size_t parallelisationFactor = 1);
    
    size_t neuronCount() const {
        return weights.rows();
    }
    size_t inputCount() const {
        return weights.columns();
    }
    const TransferFunction &transferFunction() const {
        return function;
    }
    Matrix &neuronWeights() {
        return weights;
    }
    Matrix &neuronWeightGradients() {
        return weightGradients;
    }
    Vector &neuronBiases() {
        return biases;
    }
    Vector &neuronBiasGradients() {
        return biasGradients;
    }
    Vector &activation() {
        return activations;
    }
    Vector &derivative() {
        return derivatives;
    }
    const Vector &errorTerm() const {
        return derivatives;
    }
    Vector &errorTerm() {
        return derivatives;
    }
    
    void init(uint32_t seed);
    void dump();
    void tune();
    const Vector &predict(NNContext &ctx, const Vector &input);
    const Vector &feedforward(NNContext &ctx, const Vector &input);
    const Vector &backpropagate(NNContext &ctx);
    
    void computeErrorTerm(NNContext &ctx, const Vector &errorInput);
    void computeGradients(NNContext &ctx, const Vector &input);
private:
    Layer(const Layer&) = delete;
    const Vector &predictLinear(NNContext &ctx, const Vector &input);
    Matrix weights;
    Vector biases;
    Matrix weightGradients;
    Vector biasGradients;
    Vector activations;
    Vector derivatives;
    Vector errorOutput;
    Range2D weightInputMulWorkgroupSize;
    TransferFunction function;
    size_t parallelisationFactor;
};

} // namespace nnFit
