#include "classificationEvaluator.h"

using namespace nnFit;

ClassificationEvaluator::ClassificationEvaluator(Dataset &data) : data(data) {
    assert(data.hasClassificationLabels());
}

ClassificationEvaluator::Result ClassificationEvaluator::evaluate(Network &net) {
    auto classCount = data.outputSize();
    auto size = data.size();
    auto &device = net.device();
    Vector input(device, data.inputSize());
    Vector output(device, data.outputSize());
    assert(data.hasClassificationLabels());
    const auto &labels = *data.classificationLabels();
    Vector classificationResult(device, labels.size(), ValueType(ValueType::Uint8));

    auto &kernel = net.context().floatKernels.evaluateClassification;
    
    for (size_t i = 0; i < size; ++i) {
        data.get(i, 1, input, output);
        const auto &hypothesis = net.predict(input);
        kernel.setArg(0, hypothesis).setArg(1, classCount).setArg(2, labels).setArg(3, classificationResult);
        device.queue().enqueue1Dim(kernel, 1, i);
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