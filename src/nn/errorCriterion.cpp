#include "errorCriterion.h"
#include "network.h"

using namespace nnFit;

const Vector &MSECriterion::computeError(NNContext &ctx, const Vector &prediction, const Vector &expectedOutput, Vector &accumulatedErrors) {
    assert(prediction.size() == expectedOutput.size());
    ctx.queue().enqueue1Dim(ctx.floatKernels.meanSquaredError(prediction, expectedOutput, accumulatedErrors), prediction.size());
    return accumulatedErrors;
}

void MSECriterion::computeLastLayerError(NNContext &ctx, Layer &layer, const Vector &expectedOutput) {
    assert(layer.activation().size() == expectedOutput.size());
    ctx.queue().enqueue1Dim(ctx.floatKernels.computeMSELayerError(layer.activation(), expectedOutput, layer.derivative()), layer.activation().size());
}

const Vector &CrossEntropyCriterion::computeError(NNContext &ctx, const Vector &prediction, const Vector &expectedOutput, Vector &accumulatedErrors) {
    assert(prediction.size() == expectedOutput.size());
    ctx.queue().enqueue1Dim(ctx.floatKernels.crossEntropyError(prediction, expectedOutput, accumulatedErrors), prediction.size());
    return accumulatedErrors;
}

void CrossEntropyCriterion::computeLastLayerError(NNContext &ctx, Layer &layer, const Vector &expectedOutput) {
    assert(layer.activation().size() == expectedOutput.size());
    ctx.queue().enqueue1Dim(ctx.floatKernels.computeCrossEntropyLayerError(layer.activation(), expectedOutput, layer.derivative()), layer.activation().size());
}