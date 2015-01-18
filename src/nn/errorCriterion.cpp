#include "errorCriterion.h"
#include "network.h"

using namespace nnFit;

const Vector &MSECriterion::computeError(NNContext &ctx, const Vector &prediction, const Vector &expectedOutput, Vector &accumulatedErrors) {
    assert(prediction.size() == expectedOutput.size());
    auto &device = prediction.device();
    
    auto &kernel = ctx.floatKernels.meanSquaredError;
    kernel.setArg(0, prediction).setArg(1, expectedOutput).setArg(2, accumulatedErrors);
    device.queue().enqueue1Dim(kernel, prediction.size());
    return accumulatedErrors;
}

void MSECriterion::computeLastLayerError(NNContext &ctx, Layer &layer, const Vector &expectedOutput) {
    auto &device = layer.activation().device();
    assert(layer.activation().size() == expectedOutput.size());
    
    auto &k = ctx.floatKernels.computeMSELayerError;
    k.setArg(0, layer.activation()).setArg(1, expectedOutput).setArg(2, layer.derivative());
    device.queue().enqueue1Dim(k, layer.activation().size());
}

const Vector &CrossEntropyCriterion::computeError(NNContext &ctx, const Vector &prediction, const Vector &expectedOutput, Vector &accumulatedErrors) {
    assert(prediction.size() == expectedOutput.size());
    auto &device = prediction.device();
    
    auto &kernel = ctx.floatKernels.crossEntropyError;
    kernel.setArg(0, prediction).setArg(1, expectedOutput).setArg(2, accumulatedErrors);
    device.queue().enqueue1Dim(kernel, prediction.size());
    return accumulatedErrors;
}

void CrossEntropyCriterion::computeLastLayerError(NNContext &ctx, Layer &layer, const Vector &expectedOutput) {
    auto &device = layer.activation().device();
    assert(layer.activation().size() == expectedOutput.size());
    
    auto &k = ctx.floatKernels.computeCrossEntropyLayerError;
    k.setArg(0, layer.activation()).setArg(1, expectedOutput).setArg(2, layer.derivative());
    device.queue().enqueue1Dim(k, layer.activation().size());
}