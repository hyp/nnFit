#pragma once

#include <vector>
#include "../src/core/vector.h"
#include "../src/core/dataset.h"

using namespace nnFit;

class MNIST: public Dataset {
public:
    MNIST(Device &device);
    
    // Return true on error.
    bool load(const char *imageFilename, const char *labelFilename);
    
    uint32_t imageWidth() const {
        return width;
    }
    
    uint32_t imageHeight() const {
        return height;
    }
    
    size_t size() const override;
    size_t inputSize() const override;
    size_t outputSize() const override;
    
    void get(size_t i, Vector &input, Vector &output) override;
    
    const Vector *classificationLabels() override;
    
private:
    Device &device;
    uint32_t width, height;
    Matrix images_;
    Matrix labelProbabilities_;
    std::vector<uint8_t> labelValues;
    Vector labels;
};