#pragma once

#include "core/vector.h"

namespace nnFit {

class NNContext;

class Layer {
public:
    enum NeuronType {
        Linear,
        Sigmoid,
        RectifiedLinearUnit
    };
    
    Layer(Device &device, size_t neuronCount, size_t inputCount, NeuronType type);
    
    size_t neuronCount() const {
        return weights.rows();
    }
    size_t inputCount() const {
        return weights.columns();
    }
    NeuronType type() const {
        return neuronType;
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
    Vector &predict(NNContext &ctx, const Vector &input);
    Vector &feedforward(NNContext &ctx, const Vector &input);
    
    void computeErrorTerm(NNContext &ctx, const Layer &next);
    void computeGradients(NNContext &ctx, const Vector &input);
private:
    Layer(const Layer&) = delete;
    void predictLinear(NNContext &ctx, const Vector &input);
    Matrix weights;
    Vector biases;
    Matrix weightGradients;
    Vector biasGradients;
    Vector activations;
    Vector derivatives;
    NeuronType neuronType;
};

} // namespace nnFit
