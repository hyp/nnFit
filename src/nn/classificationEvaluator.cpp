#include "classificationEvaluator.h"

using namespace nnFit;

ClassificationEvaluator::ClassificationEvaluator(Dataset &data) : data(data) {
    assert(data.hasClassificationLabels());
}

ClassificationEvaluator::Result ClassificationEvaluator::evaluate(Network &net, size_t parallelisationFactor) {
    auto classCount = data.outputSize();
    auto size = data.size();
    assert((size % parallelisationFactor) == 0);
    auto &device = net.device();
    Vector input(device, data.inputSize() * parallelisationFactor);
    Vector output(device, data.outputSize() * parallelisationFactor);
    assert(data.hasClassificationLabels());
    const auto &labels = *data.classificationLabels();
    Vector classificationResult(device, labels.size(), ValueType(ValueType::Uint8));

    auto &kernel = net.context().floatKernels.evaluateClassification;
    
    for (size_t i = 0; i < size; i += parallelisationFactor) {
        data.get(i, parallelisationFactor, input, output);
        const auto &hypothesis = net.predict(input);
        kernel.setArg(0, hypothesis).setArg(1, classCount).setArg(2, labels).setArg(3, classificationResult);
        device.queue().enqueue1Dim(kernel, parallelisationFactor, i);
    }
    
    // Compute the number of correct predictions
    Vector count(device, 1, ValueType(ValueType::Uint32));
    partialTrueCount(count, classificationResult);
    std::vector<uint32_t> hostCounts;
    count.copy(hostCounts);
    
    Result result;
    result.count = size;
    result.correctPredictions = hostCounts[0];
    return result;
}