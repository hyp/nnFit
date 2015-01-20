#include <iostream>
#include <chrono>
#include "trainer.h"
#include "errorCriterion.h"
#include "optimizers/optimizer.h"

using namespace nnFit;

Trainer::Trainer(Network &network, ErrorCriterion &criterion, Dataset &data) : network(network), criterion(criterion), data(data), trainingExampleCount(data.size()), input(network.device(), data.inputSize()), output(network.device(), data.outputSize()) {
    reshuffleIndices = false;
    profile = false;
}

void Trainer::gradientDescent(Optimizer &opt, size_t iterations) {
    train(opt, iterations, trainingExampleCount);
}

void Trainer::miniBatchGradientDescent(Optimizer &opt, size_t iterations, size_t miniBatchSize) {
    train(opt, iterations, miniBatchSize);
}

void Trainer::train(Optimizer &opt, size_t iterations, size_t miniBatchSize) {
    Vector errors(network.device(), data.outputSize());
    auto weightsAndGradients = network.weightsAndGradients();
    size_t batchCount = trainingExampleCount/miniBatchSize + (trainingExampleCount % miniBatchSize == 0? 0 : 1);
    
    std::vector<size_t> indices(trainingExampleCount);
    for (size_t i = 0; i < indices.size(); ++i)
        indices[i] = i;
    
    std::chrono::high_resolution_clock::time_point iterationStart;
    for (size_t iteration = 0; iteration < iterations; ++iteration) {
        float iterationError = 0.0f;
        if (profile)
            iterationStart = std::chrono::high_resolution_clock::now();
        // Shuffle indices if needed
        if (reshuffleIndices)
            std::random_shuffle(indices.begin(), indices.end());
        
        for (size_t batch = 0; batch < batchCount; ++batch) {
            size_t count = std::min(batch*miniBatchSize+miniBatchSize, trainingExampleCount) - batch*miniBatchSize;
            
            // Reset gradients and errors
            for (auto &i : weightsAndGradients) {
                i.second->zeros();
            }
            errors.zeros();
            
            // Train
            for (size_t i = batch*miniBatchSize, end = i + count; i < end; ++i) {
                data.get(indices[i], input, output);
                
                const auto &prediction = network.feedforward(input);
                criterion.computeError(network.context(), prediction, output, errors);
                criterion.computeLastLayerError(network.context(), network.lastLayer(), output);
                network.backpropagate();
            }
            
            // gradients = gradients / numberOfTrainingExamples
            for (auto &i : weightsAndGradients) {
                div(*i.second, float(count));
            }
            
            // Sum the error
            Vector errorSum(network.device(), 1);
            partialSum(errorSum, errors);

            std::vector<float> errs;
            errorSum.copy(errs);
            float err = errs[0];
            iterationError += err;
            err/=float(count);
            
            opt.optimize(weightsAndGradients);
        }
        if (profile) {
            network.device().queue().finish();
            auto now = std::chrono::high_resolution_clock::now();
            auto seconds = std::chrono::duration_cast<std::chrono::seconds>(now - iterationStart).count();
            std::cout << "One training iteration ran for " << seconds << "s\n";
        }
        if (afterIteration) {
            afterIteration(iteration, iterationError/trainingExampleCount);
        }
    }
}