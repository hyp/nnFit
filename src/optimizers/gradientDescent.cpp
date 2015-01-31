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

        queue.enqueue1Dim(kernel(weights, gradients, k), weights.size());
    }
}