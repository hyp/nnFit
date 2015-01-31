#include <random>
#include "network.h"

using namespace nnFit;

NNContext::Specialization::Specialization(Device &device, Program &program) {
    sigmoidPredict = Kernel(program, "sigmoidPredict");
    sigmoidFeedforward = Kernel(program, "sigmoidFeedforward");
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

Network::Network(Device &device) : dev(device), ctx(device) {
}

std::vector<std::pair<const Vector*, const Vector*>> Network::weightsAndGradients() {
    std::vector<std::pair<const Vector*, const Vector*>> result;
    for (const auto &layer: layers) {
        result.push_back(std::make_pair(&layer->neuronWeights(), &layer->neuronWeightGradients()));
        result.push_back(std::make_pair(&layer->neuronBiases(), &layer->neuronBiasGradients()));
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
    const auto *error = &layers[i]->backpropagate(ctx, expectedOutput, criterion, i != 0);
    layers[i]->computeGradients(ctx);
    
    for (; i != 0; ) {
        --i;
        error = &layers[i]->backpropagate(ctx, *error, i != 0);
        layers[i]->computeGradients(ctx);
    }
}
