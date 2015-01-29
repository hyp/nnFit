#include "errorCriterion.h"
#include "network.h"

using namespace nnFit;

static void checkLayerParams(const Vector &prediction, const Vector &expectedOutput, const Vector &derivative, const Vector &errorTerm) {
    assert(prediction.size() == expectedOutput.size());
    assert(prediction.size() == derivative.size());
    assert(prediction.size() == errorTerm.size());
}

const Vector &MSECriterion::computeError(NNContext &ctx, const Vector &prediction, const Vector &expectedOutput, Vector &accumulatedErrors) {
    assert(prediction.size() == expectedOutput.size());
    ctx.queue().enqueue1Dim(ctx.floatKernels.meanSquaredError(prediction, expectedOutput, accumulatedErrors), prediction.size());
    return accumulatedErrors;
}

void MSECriterion::computeLayerError(NNContext &ctx, const Vector &prediction, const Vector &expectedOutput, const Vector &derivative, const Vector &errorTerm) {
    checkLayerParams(prediction, expectedOutput, derivative, errorTerm);
    ctx.queue().enqueue1Dim(ctx.floatKernels.computeMSELayerError(prediction, expectedOutput, derivative, errorTerm), prediction.size());
}

const Vector &CrossEntropyCriterion::computeError(NNContext &ctx, const Vector &prediction, const Vector &expectedOutput, Vector &accumulatedErrors) {
    assert(prediction.size() == expectedOutput.size());
    ctx.queue().enqueue1Dim(ctx.floatKernels.crossEntropyError(prediction, expectedOutput, accumulatedErrors), prediction.size());
    return accumulatedErrors;
}

void CrossEntropyCriterion::computeLayerError(NNContext &ctx, const Vector &prediction, const Vector &expectedOutput, const Vector &derivative, const Vector &errorTerm) {
    checkLayerParams(prediction, expectedOutput, derivative, errorTerm);
    ctx.queue().enqueue1Dim(ctx.floatKernels.computeCrossEntropyLayerError(prediction, expectedOutput, errorTerm), prediction.size());
}