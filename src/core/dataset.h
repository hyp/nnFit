#pragma once

#include "vector.h"

namespace nnFit {

class Dataset {
public:
    virtual size_t size() const = 0;
    virtual size_t inputSize() const = 0;
    virtual size_t outputSize() const = 0;
    virtual void get(size_t i, Vector &input, Vector &output) = 0;
    
    // Optional
    virtual const Vector *classificationLabels() {
        return nullptr;
    }
    
    bool hasClassificationLabels() {
        return classificationLabels() != nullptr;
    }
};

class SimpleDataset: public Dataset {
public:
    SimpleDataset(const Matrix &inputs, const Matrix &outputs);
    
    size_t size() const override;
    size_t inputSize() const override;
    size_t outputSize() const override;
    void get(size_t i, Vector &input, Vector &output) override;
    
private:
    const Matrix &inputs;
    const Matrix &outputs;
};

} // namespace nnFit