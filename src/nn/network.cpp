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
    computeError = Kernel(program, "computeError");
    computeWeightGradients = Kernel(program, "computeWeightGradient");
    computeWeightGradients4 = Kernel(program, "computeWeightGradient4");
    evaluateClassification = Kernel(program, "evaluateClassification");
}

NNContext::NNContext(Device &device) : floatKernels(device, device.getProgram("nn.cl")) {
}

Network::Network(Device &device) : dev(device), ctx(device) {
}

Network &Network::inputLayer(size_t size) {
    inputLayerSize = size;
    return *this;
}

std::vector<std::pair<Vector*, Vector*>> Network::weightsAndGradients() {
    std::vector<std::pair<Vector*, Vector*>> result;
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
    networkInput = &input;
    auto *x = &input;
    for (const auto &layer : layers) {
        x = &layer->predict(ctx, *x);
    }
    return *x;
}

const Vector &Network::feedforward(const Vector &input) {
    networkInput = &input;
    auto *x = &input;
    for (const auto &layer : layers) {
        x = &layer->feedforward(ctx, *x);
    }
    return *x;
}

void Network::backpropagate() {
    size_t i = layers.size() - 1;
    layers[i]->computeGradients(ctx, i == 0? *networkInput : layers[i - 1]->activation());
    
    for (; i != 0; ) {
        --i;
        layers[i]->computeErrorTerm(ctx, *layers[i + 1]);
        layers[i]->computeGradients(ctx, i == 0? *networkInput : layers[i - 1]->activation());
    }
}
