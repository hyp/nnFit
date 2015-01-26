#pragma once

namespace nnFit {
    
class Vector;
class Kernel;
class NNContext;

class TransferFunction {
public:
    enum Kind {
        Linear,
        Sigmoid,
        RectifiedLinearUnit
    };
    
    TransferFunction(Kind kind) : kind(kind) { }
    
    // Applies the transfer function to the given input vector and returns it.
    const Vector &apply(NNContext &ctx, const Vector &input) const;
    
    // Applies the transfer function to the given input vector and returns it.
    // Also applies the derivative of the transfer function to the given derivative vector.
    const Vector &apply(NNContext &ctx, const Vector &input, const Vector &derivative) const;
    
private:
    Kind kind;
};

} // namespace nnFit