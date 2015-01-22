#include <iostream>
#include "trainer.h"
#include "errorCriterion.h"
#include "optimizers/optimizer.h"

using namespace nnFit;

Trainer::Trainer(Network &network, ErrorCriterion &criterion, Dataset &data) : network(network), criterion(criterion), data(data), trainingExampleCount(data.size()), input(network.device(), data.inputSize()), output(network.device(), data.outputSize()) {
    reshuffleIndices = false;
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
    
    for (size_t iteration = 0; iteration < iterations; ++iteration) {
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
            float err = errs[0]/float(count);
            
            opt.optimize(weightsAndGradients);
            
            if (afterIteration) {
                afterIteration(iteration, batch, err);
            }
        }
    }
}