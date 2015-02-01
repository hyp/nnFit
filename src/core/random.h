#pragma once

#include "vector.h"

namespace nnFit {
    
class RandomGenerator {
public:
    RandomGenerator(Device &device, size_t size, uint32_t seed = 0);
    
    // Generates a uniform distribution of random floats in the range from 0 to 1.
    void uniformFloatDistribution(const Vector &dest);
    
    // Performs an inverted dropout on the given vector.
    void invertedDropout(const Vector &dest, float activationProbability);
    
private:
    Vector state;
    Program &program;
    Kernel uniformRandomKernel;
    Kernel invertedDropoutKernel;
};
    
} // namespace nnFit
