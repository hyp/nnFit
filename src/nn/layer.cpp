#include <iostream>
#include <random>
#include "layer.h"
#include "network.h"

using namespace nnFit;

Layer::Layer(Device &device, size_t neuronCount, size_t inputCount, NeuronType type)
: weights(device, neuronCount, inputCount), biases(device, neuronCount), weightGradients(device, neuronCount, inputCount), biasGradients(device, neuronCount), activations(device, neuronCount), derivatives(device, neuronCount), neuronType(type) {
}

void Layer::init(uint32_t seed) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> d(0.0, 1.0/std::sqrt(float(inputCount())));
    std::vector<float> init(weights.size());
    for (auto &v : init)
        v = d(gen);
    weights.write(init);
    init.resize(biases.size());
    for (auto &v : init)
        v = d(gen);
    biases.write(init);
}

void Layer::dump() {
    std::vector<float> w(weights.size());
    weights.copy(w);
    std::vector<float> b(biases.size());
    biases.copy(b);
    std::cout << "Neurons :\n";
    for (size_t i = 0; i < neuronCount(); ++i) {
        std::cout << "#" << i << ": " << b[i] << ", ";
        for (size_t j = 0; j < inputCount(); ++j) {
            std::cout << w[i*inputCount()+j] << ", ";
        }
        std::cout<< "\n";
    }
}

void Layer::predictLinear(NNContext &ctx, const Vector &input) {
    biases.copy(activations);
    auto &feedforward = ctx.floatKernels.feedforward;
    feedforward.setArg(0, weights).setArg(1, input).setArg(2, input.size()).setArg(3, activations);
    input.device().queue().enqueue1Dim(feedforward, activations.size());
}

static Kernel &predictFunction(NNContext &ctx, Layer::NeuronType type) {
    switch (type) {
        case Layer::Sigmoid:
            return ctx.floatKernels.sigmoidPredict;
            break;
        case Layer::RectifiedLinearUnit:
            return ctx.floatKernels.reluPredict;
            break;
        default: break;
    }
    assert(false && "Invalid type");
}

Vector &Layer::predict(NNContext &ctx, const Vector &input) {
    assert(input.size() == inputCount());
    
    // activation = f(Wx + b)
    auto &queue = input.device().queue();
    predictLinear(ctx, input);
    
    if (neuronType == Linear)
        return activations;
    auto &k = predictFunction(ctx, neuronType);
    k.setArg(0, activations);
    queue.enqueue1Dim(k, activations.size());
    return activations;
}

static Kernel &feedforwardFunction(NNContext &ctx, Layer::NeuronType type) {
    switch (type) {
        case Layer::Sigmoid:
            return ctx.floatKernels.sigmoidFeedforward;
            break;
        case Layer::RectifiedLinearUnit:
            return ctx.floatKernels.reluFeedforward;
            break;
        default: break;
    }
    assert(false && "Invalid type");
}

Vector &Layer::feedforward(NNContext &ctx, const Vector &input) {
    assert(input.size() == inputCount());
    
    // activation = f(Wx + b)
    // derivative = f'(Wx + b)
    auto &queue = input.device().queue();
    predictLinear(ctx, input);
    
    if (neuronType == Linear) {
        derivatives.ones();
        return activations;
    }
    auto &k = feedforwardFunction(ctx, neuronType);
    k.setArg(0, activations).setArg(1, derivatives);
    queue.enqueue1Dim(k, activations.size());
    return activations;
}

void Layer::computeErrorTerm(NNContext &ctx, const Layer &next) {
    // error = (nextLayerWeights' * nextLayerError) .* derivative
    auto &queue = weights.device().queue();
    auto &k = ctx.floatKernels.computeError;
    k.setArg(0, next.weights).setArg(1, next.errorTerm()).setArg(2, next.neuronCount()).setArg(3, next.weights.columns()).setArg(4, derivatives);
    queue.enqueue1Dim(k, neuronCount());
}

void Layer::computeGradients(NNContext &ctx, const Vector &input) {
    // weightGradient += error * input'
    // biasGradient += error
    auto &queue = weights.device().queue();
    auto &k = ctx.floatKernels.computeWeightGradients;
    k.setArg(0, errorTerm()).setArg(1, input).setArg(2, weightGradients);
    queue.enqueue2Dim(k, weightGradients.rows(), weightGradients.columns());
    add(biasGradients, errorTerm());
}
