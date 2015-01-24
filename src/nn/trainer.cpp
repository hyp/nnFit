#include <iostream>
#include <chrono>
#include "trainer.h"
#include "errorCriterion.h"
#include "optimizers/optimizer.h"

using namespace nnFit;

Trainer::Trainer(Network &network, ErrorCriterion &criterion, Dataset &data, size_t parallelisationFactor) : network(network), criterion(criterion), data(data), trainingExampleCount(data.size()), parallelisationFactor(parallelisationFactor) {
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
    Vector input(network.device(), data.inputSize() * parallelisationFactor);
    Vector output(network.device(), data.outputSize() * parallelisationFactor);
    Vector errors(network.device(), data.outputSize() * parallelisationFactor);
    Vector errorSum(network.device(), 1);
    std::vector<float> errs;
    
    auto weightsAndGradients = network.weightsAndGradients();
    
    assert((trainingExampleCount % parallelisationFactor) == 0);
    assert((trainingExampleCount % miniBatchSize) == 0);
    assert((miniBatchSize % parallelisationFactor) == 0);
    assert(miniBatchSize >= parallelisationFactor);
    size_t batchCount = trainingExampleCount / miniBatchSize;
    size_t passCount = trainingExampleCount / parallelisationFactor;
    size_t passPerBatchCount = miniBatchSize / parallelisationFactor;
    std::vector<size_t> indices(passCount);
    for (size_t i = 0; i < indices.size(); ++i)
        indices[i] = i;
    
    std::chrono::high_resolution_clock::time_point iterationStart;
    for (size_t iteration = 0; iteration < iterations; ++iteration) {
        // Reset errors
        errors.zeros();
        if (profile)
            iterationStart = std::chrono::high_resolution_clock::now();
        // Shuffle indices if needed
        if (reshuffleIndices)
            std::random_shuffle(indices.begin(), indices.end());
        
        for (size_t batch = 0; batch < batchCount; ++batch) {
            
            // Reset gradients
            for (auto &i : weightsAndGradients) {
                i.second->zeros();
            }
            
            // Train
            for (size_t i = 0; i < passPerBatchCount; ++i) {
                data.get(indices[batch*passPerBatchCount + i]*parallelisationFactor, parallelisationFactor, input, output);
                
                const auto &prediction = network.feedforward(input);
                criterion.computeError(network.context(), prediction, output, errors);
                criterion.computeLastLayerError(network.context(), network.lastLayer(), output);
                network.backpropagate();
            }
            
            // gradients = gradients / numberOfTrainingExamples
            // Scale the gradients while optimizing to avoid redundant division step.
            opt.optimize(weightsAndGradients, miniBatchSize);
        }
        
        // Compute the iteration error.
        partialSum(errorSum, errors);
        errorSum.copy(errs);
        float iterationError = errs[0] / float(trainingExampleCount);
        
        if (profile) {
            network.device().queue().finish();
            auto now = std::chrono::high_resolution_clock::now();
            auto seconds = std::chrono::duration_cast<std::chrono::seconds>(now - iterationStart).count();
            std::cout << "One training iteration ran for " << seconds << "s\n";
        }
        if (afterIteration) {
            afterIteration(iteration, iterationError);
        }
    }
}