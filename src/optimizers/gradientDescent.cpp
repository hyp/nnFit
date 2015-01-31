#include "gradientDescent.h"

using namespace nnFit;

GradientDescent::GradientDescent(Device &device, float learningRate)
: device(device), learningRate(learningRate) {
    kernel = Kernel(device.getProgram("gradientDescent.cl"), "gradientDescent");
}

void GradientDescent::optimize(const std::vector<std::pair<const Vector*, const Vector*>> &weightsAndGradients, size_t trainingExamples) {
    auto &queue = device.queue();
    float k = learningRate/float(trainingExamples);
    for (auto &i : weightsAndGradients) {
        const auto &weights = *i.first;
        const auto &gradients = *i.second;
        assert(weights.size() == gradients.size());

        queue.enqueue1Dim(kernel(weights, gradients, k), weights.size());
    }
}

MomentumGradientDescent::MomentumGradientDescent(Device &device, float learningRate, float momentumDecay) : device(device), learningRate(learningRate), momentumDecay(momentumDecay) {
    kernel = Kernel(device.getProgram("gradientDescent.cl"), "momentumGradientDescent");
}

void MomentumGradientDescent::optimize(const std::vector<std::pair<const Vector*, const Vector*>> &weightsAndGradients, size_t trainingExamples) {
    if (velocities.empty()) {
        for (const auto &i : weightsAndGradients) {
            velocities.push_back(Vector(device, i.first->size()));
            velocities.back().zeros();
        }
    }
    assert(velocities.size() == weightsAndGradients.size());
    auto &queue = device.queue();
    float k = learningRate/float(trainingExamples);
    for (size_t i = 0; i < weightsAndGradients.size(); ++i) {
        const auto &weights = *weightsAndGradients[i].first;
        const auto &gradients = *weightsAndGradients[i].second;
        assert(weights.size() == gradients.size());
        
        queue.enqueue1Dim(kernel(weights, gradients, velocities[i], k, momentumDecay), weights.size());
    }
}