#include "sequentialDataset.h"

using namespace nnFit;

Sequence::Sequence(const Matrix &inputs, const Matrix &outputs) : inputs(inputs), outputs(outputs) {
    assert(inputs.rows() == outputs.rows());
}

size_t Sequence::length() const {
    return inputs.rows();
}

void Sequence::get(size_t i, Vector &input, Vector &output) const {
    inputs.row(i).copy(input);
    outputs.row(i).copy(output);
}