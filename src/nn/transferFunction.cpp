#include "network.h"
#include "transferFunction.h"

using namespace nnFit;

static const Kernel &predictFunction(NNContext &ctx, TransferFunction::Kind kind) {
    switch (kind) {
    case TransferFunction::Sigmoid:
        return ctx.floatKernels.sigmoidPredict;
        break;
    case TransferFunction::RectifiedLinearUnit:
        return ctx.floatKernels.reluPredict;
        break;
    default: break;
    }
    assert(false && "Invalid transfer function");
}

static const Kernel &feedforwardFunction(NNContext &ctx, TransferFunction::Kind kind) {
    switch (kind) {
    case TransferFunction::Sigmoid:
        return ctx.floatKernels.sigmoidFeedforward;
        break;
    case TransferFunction::RectifiedLinearUnit:
        return ctx.floatKernels.reluFeedforward;
        break;
    default: break;
    }
    assert(false && "Invalid transfer function");
}

const Vector &TransferFunction::apply(NNContext &ctx, const Vector &input) const {
    if (kind == Linear)
        return input;
    ctx.queue().enqueue1Dim(predictFunction(ctx, kind)(input), input.size());
    return input;
}

const Vector &TransferFunction::apply(NNContext &ctx, const Vector &input, const Vector &derivative) const {
    assert(input.size() == derivative.size());
    assert(input.type() == derivative.type());
    if (kind == Linear) {
        derivative.ones();
        return input;
    }
    ctx.queue().enqueue1Dim(feedforwardFunction(ctx, kind)(input, derivative), input.size());
    return input;
}