#include <random>
#include "network.h"

using namespace nnFit;

NNContext::Specialization::Specialization(Device &device, Program &program) {
    sigmoidPredict = Kernel(program, "sigmoidPredict");
    sigmoidFeedforward = Kernel(program, "sigmoidFeedforward");
    tanhPredict = Kernel(program, "tanhPredict");
    tanhFeedforward = Kernel(program, "tanhFeedforward");
    reluPredict = Kernel(program, "reluPredict");
    reluFeedforward = Kernel(program, "reluFeedforward");
    meanSquaredError = Kernel(program, "meanSquaredError");
    crossEntropyError = Kernel(program, "crossEntropyError");
    computeMSELayerError = Kernel(program, "computeMSELayerError");
    computeCrossEntropyLayerError = Kernel(program, "computeCrossEntropyLayerError");
    computeWeightGradients = Kernel(program, "computeWeightGradient");
    computeWeightGradients4 = Kernel(program, "computeWeightGradient4");
    computeWeightGradientsParallel = Kernel(program, "computeWeightGradientParallel");
    computeWeightGradients4Parallel = Kernel(program, "computeWeightGradient4Parallel");
    computeBiasGradients = Kernel(program, "computeBiasGradient");
    evaluateClassification = Kernel(program, "evaluateClassification");
}

NNContext::NNContext(Device &device) : floatKernels(device, device.getProgram("nn.cl")), queue_(device.queue()) {
}

Network::Network(Device &device) : dev(device), ctx(device), backpropagateUntil(0) {
}

Network &Network::add(std::unique_ptr<AbstractLayer> layer) {
    if (!layer->backpropagates() && backpropagateUntil == layers.size()) {
        backpropagateUntil++;
    }
    layers.push_back(std::move(layer));
    return *this;
}

std::vector<std::pair<const Vector*, const Vector*>> Network::weightsAndGradients() {
    std::vector<std::pair<const Vector*, const Vector*>> result;
    for (const auto &layer: layers) {
        layer->collectWeightsAndGradients(result);
    }
    return result;
}

void Network::init(uint32_t seed) {
    for (const auto &layer : layers) {
        layer->init(seed);
    }
}

void Network::init() {
    std::random_device seed;
    init(seed());
}

void Network::dump() {
    for (const auto &layer : layers) {
        layer->dump();
    }
}

void Network::tune() {
    for (const auto &layer : layers) {
        layer->tune();
    }
}

const Vector &Network::predict(const Vector &input) {
    const auto *x = &input;
    for (const auto &layer : layers) {
        x = &layer->predict(ctx, *x);
    }
    return *x;
}

const Vector &Network::feedforward(const Vector &input) {
    const auto *x = &input;
    for (const auto &layer : layers) {
        x = &layer->feedforward(ctx, *x);
    }
    return *x;
}

void Network::backpropagate(const Vector &expectedOutput, const ErrorCriterion &criterion) {
    size_t i = layers.size() - 1;
    const auto *error = &layers[i]->backpropagate(ctx, expectedOutput, criterion, i != backpropagateUntil);
    layers[i]->accumulateGradients(ctx);
    
    for (; i != backpropagateUntil; ) {
        --i;
        error = &layers[i]->backpropagate(ctx, *error, i != backpropagateUntil);
        layers[i]->accumulateGradients(ctx);
    }
}
