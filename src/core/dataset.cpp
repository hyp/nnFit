#include "dataset.h"

using namespace nnFit;

SimpleDataset::SimpleDataset(const Matrix &inputs, const Matrix &outputs) : inputs(inputs), outputs(outputs) {
}

size_t SimpleDataset::size() const {
    return inputs.rows();
}

size_t SimpleDataset::inputSize() const {
    return inputs.columns();
}

size_t SimpleDataset::outputSize() const {
    return outputs.columns();
}

void SimpleDataset::get(size_t i, Vector &input, Vector &output) {
    inputs.row(i).copy(input);
    outputs.row(i).copy(output);
}