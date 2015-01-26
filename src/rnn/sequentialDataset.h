#pragma once

#include "core/vector.h"

namespace nnFit {
    
class Sequence {
public:
    Sequence(const Matrix &inputs, const Matrix &outputs);
    
    size_t length() const;
    void get(size_t i, Vector &input, Vector &output) const;
    
private:
    const Matrix &inputs;
    const Matrix &outputs;
};
    
class SequentialDataset {
public:
    virtual size_t size() const = 0;
    virtual size_t inputSize() const = 0;
    virtual size_t outputSize() const = 0;
    virtual Sequence get(size_t i) = 0;
};
    
} // namespace nnFit