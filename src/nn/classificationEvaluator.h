#pragma once

#include "network.h"
#include "core/dataset.h"

namespace nnFit {

class ClassificationEvaluator {
public:
    struct Result {
        size_t count;
        size_t correctPredictions;

        float percentageOfCorrectPredictions() const {
            return float(correctPredictions)/float(count)*100.0f;
        }
    };
    
    ClassificationEvaluator(Dataset &data);
    
    Result evaluate(Network &net, size_t parallelisationFactor = 1);
private:
    Dataset &data;
    size_t correctPredictions;
};

} // namespace nnFit
