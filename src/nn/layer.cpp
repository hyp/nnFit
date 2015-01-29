#include <iostream>
#include <random>
#include <array>
#include "layer.h"
#include "network.h"

using namespace nnFit;

Layer::Layer(Device &device, size_t neuronCount, size_t inputCount, TransferFunction transferFunction, size_t parallelisationFactor)
: weights(device, neuronCount, inputCount), biases(device, neuronCount), weightGradients(device, neuronCount, inputCount), biasGradients(device, neuronCount), activations(device, neuronCount*parallelisationFactor), derivatives(device, neuronCount*parallelisationFactor), function(transferFunction), parallelisationFactor(parallelisationFactor) {
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

void Layer::tune() {
    // Tune weights by input multiplication
    const std::array<size_t, 7> workgroupColumns = {1,2,4,8,16,32,64};
    const std::array<size_t, 10> workgroupRows = {1,2,3,4,5,7,8,10,16,32};
    auto &device = weights.device();
    const size_t iterations = 100;
    
    Vector output(device, weights.rows());
    Vector input(device, weights.columns());
    input.ones();
    double bestTime = 0;
    bool first = true;
    for (auto rows: workgroupRows) {
        if ((weights.rows() % rows) != 0)
            continue;
        for (auto columns: workgroupColumns) {
            if ((weights.columns() % columns) != 0)
                continue;
            auto time = device.profile([&, this] () {
                for (size_t i = 0; i < iterations; ++i)
                    parallelMul(output, weights, input, Range2D(rows, columns));
            });
            if (first || time < bestTime) {
                weightInputMulWorkgroupSize = Range2D(rows, columns);
                first = false;
                bestTime = time;
            }
        }
    }
    std::cout << "Best workgroup size for "<<weights.rows() << " by " << weights.columns() << " matrix vector multiplication: " << weightInputMulWorkgroupSize[0] << ", " << weightInputMulWorkgroupSize[1] << "\n";
}

const Vector &Layer::predictLinear(NNContext &ctx, const Vector &input) {
    parallelMul(activations, weights, input, weightInputMulWorkgroupSize);
    parallelAdd(activations, biases, activations);
    return activations;
}

const Vector &Layer::predict(NNContext &ctx, const Vector &input) {
    assert(input.size() == inputCount()*parallelisationFactor);
    
    // activation = f(Wx + b)
    return function.apply(ctx, predictLinear(ctx, input));
}

const Vector &Layer::feedforward(NNContext &ctx, const Vector &input) {
    assert(input.size() == inputCount()*parallelisationFactor);
    
    // activation = f(Wx + b)
    // derivative = f'(Wx + b)
    return function.apply(ctx, predictLinear(ctx, input), derivatives);
}

void Layer::computeErrorTerm(NNContext &ctx, const Layer &next) {
    // error = (nextLayerWeights' * nextLayerError) .* derivative
    auto task = ctx.floatKernels.computeError(next.weights, next.errorTerm(), next.neuronCount(), next.weights.columns(), derivatives);
    ctx.queue().enqueue2Dim(task, Range2D(parallelisationFactor, neuronCount()));
}

static const Kernel &chooseWeightGradientKernel(NNContext &ctx, size_t parallelisationFactor, bool use4wide) {
    if (parallelisationFactor == 1) {
        return use4wide? ctx.floatKernels.computeWeightGradients4 : ctx.floatKernels.computeWeightGradients;
    }
    return use4wide? ctx.floatKernels.computeWeightGradients4Parallel : ctx.floatKernels.computeWeightGradientsParallel;
}

void Layer::computeGradients(NNContext &ctx, const Vector &input) {
    auto &queue = ctx.queue();
    // weightGradient += error * input'
    bool use4wide = weightGradients.columns() % 4 == 0;
    const auto &kernel = chooseWeightGradientKernel(ctx, parallelisationFactor, use4wide);
    queue.enqueue2Dim(parallelisationFactor == 1? kernel(errorTerm(), input, weightGradients) : kernel(errorTerm(), input, weightGradients, parallelisationFactor), Range2D(weightGradients.rows(), use4wide? weightGradients.columns()/4 : weightGradients.columns()));
    
    // biasGradient += error
    if (parallelisationFactor == 1) {
        add(biasGradients, errorTerm());
        return;
    }
    assert(errorTerm().size() == biasGradients.size()*parallelisationFactor);
    queue.enqueue1Dim(ctx.floatKernels.computeBiasGradients(errorTerm(), parallelisationFactor, biasGradients), biasGradients.size());
}
