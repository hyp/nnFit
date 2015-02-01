#pragma once

#include "core/vector.h"
#include "transferFunction.h"
#include "abstractLayer.h"

namespace nnFit {

class NNContext;
class ErrorCriterion;
    
class Layer: public AbstractLayer {
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
    const Matrix &neuronWeights() const {
        return weights;
    }
    const Matrix &neuronWeightGradients() const {
        return weightGradients;
    }
    const Vector &neuronBiases() const {
        return biases;
    }
    const Vector &neuronBiasGradients() const {
        return biasGradients;
    }
    const Vector &activation() const {
        return activations;
    }
    const Vector &derivative() const {
        return errorTerms;
    }
    const Vector &errorTerm() const {
        return errorTerms;
    }
    const Vector &errorOutput() const {
        return errorOutputs;
    }
    
    void init(uint32_t seed) override;
    void dump() override;
    void tune() override;
    
    const Vector &predict(NNContext &ctx, const Vector &input) override;
    const Vector &feedforward(NNContext &ctx, const Vector &input) override;
    const Vector &backpropagate(NNContext &ctx, const Vector &expectedOutput, const ErrorCriterion &criterion, bool backpropagateDown = true) override;
    const Vector &backpropagate(NNContext &ctx, const Vector &errorInput, bool backpropagateDown = true) override;
    
    const Vector &predictLinear(NNContext &ctx, const Vector &input);
    const Vector &backpropagate(NNContext &ctx);
    void updatePreviousInput(const Vector &input);
    void accumulateGradients(NNContext &ctx) override;
    void collectWeightsAndGradients(std::vector<std::pair<const Vector*, const Vector*>> &weightsAndGradients) override;
private:
    Layer(const Layer&) = delete;
    Matrix weights;
    Vector biases;
    Matrix weightGradients;
    Vector biasGradients;
    Vector activations;
    Vector errorTerms;
    Vector errorOutputs;
    const Vector *previousInput;
    Range2D weightInputMulWorkgroupSize;
    TransferFunction function;
    size_t parallelisationFactor;
};

} // namespace nnFit
