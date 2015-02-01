#include <random>
#include "random.h"

using namespace nnFit;
    
RandomGenerator::RandomGenerator(Device &device, size_t size, uint32_t seed) : state(device, size*4, ValueType::Uint32), program(device.getProgram("random.cl")) {
    // Initialize the state.
    std::mt19937 gen(seed);
    std::vector<uint32_t> init(state.size());
    for (auto &x : init)
        x = gen();
    state.write(init);
}

void RandomGenerator::uniformFloatDistribution(const Vector &dest) {
    if (!uniformRandomKernel) {
        uniformRandomKernel = std::move(Kernel(program, "uniformRandom"));
    }
    assert(dest.type() == valueType<float>());
    assert(dest.size()*4 == state.size());
    dest.device().queue().enqueue1Dim(uniformRandomKernel(dest, state), dest.size());
}

void RandomGenerator::invertedDropout(const Vector &dest, float activationProbability) {
    if (!invertedDropoutKernel) {
        invertedDropoutKernel = std::move(Kernel(program, "invertedDropout"));
    }
    assert(dest.type() == valueType<float>());
    assert(dest.size()*4 == state.size());
    dest.device().queue().enqueue1Dim(invertedDropoutKernel(dest, state, activationProbability), dest.size());
}