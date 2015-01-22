#include "gradientDescent.h"

using namespace nnFit;

GradientDescent::GradientDescent(Device &device, float learningRate)
: device(device), learningRate(learningRate) {
    kernel = Kernel(device.getProgram("gradientDescent.cl"), "gradientDescent");
}

void GradientDescent::optimize(const std::vector<std::pair<Vector*, Vector*>> &weightsAndGradients) {
    auto &queue = device.queue();
    for (auto &i : weightsAndGradients) {
        auto &weights = *i.first;
        auto &gradients = *i.second;
        
        kernel.setArg(0, weights).setArg(1, gradients).setArg(2, learningRate);
        queue.enqueue1Dim(kernel, weights.size());
    }
}