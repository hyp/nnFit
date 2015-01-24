#pragma OPENCL EXTENSION cl_khr_fp64 : enable

typedef float Scalar;
typedef float4 Scalar4;

Scalar sigmoid(Scalar x) {
    return (Scalar)1.0/((Scalar)1.0 + exp(-x));
}

kernel void sigmoidPredict(global Scalar *x) {
    size_t i = get_global_id(0);
    x[i] = sigmoid(x[i]);
}

kernel void sigmoidFeedforward(global Scalar *x, global Scalar *derivative) {
    size_t i = get_global_id(0);
    Scalar y = sigmoid(x[i]);
    x[i] = y;
    derivative[i] = y*((Scalar)1.0 - y);
}

kernel void reluPredict(global Scalar *x) {
    size_t i = get_global_id(0);
    x[i] = max(x[i], (Scalar)0.0);
}

kernel void reluFeedforward(global Scalar *x, global Scalar *derivative) {
    size_t i = get_global_id(0);
    if (x[i] > 0.0) {
        derivative[i] = 1.0;
    } else {
        x[i] = 0.0;
        derivative[i] = 0.0;
    }
}

kernel void meanSquaredError(global Scalar *prediction, global Scalar *y, global Scalar *output) {
    size_t i = get_global_id(0);
    Scalar diff = y[i] - prediction[i];
    output[i] += diff * diff;
}

kernel void crossEntropyError(global Scalar *prediction, global Scalar *y, global Scalar *output) {
    size_t i = get_global_id(0);
    Scalar err = -(y[i]*log(prediction[i]) + ((Scalar)1.0 - y[i])*log((Scalar)1.0 - prediction[i]));
    output[i] += isnan(err)? (Scalar)0.0 : err;
}

// The "responsibility" of the last layer with MSE criterion.
kernel void computeMSELayerError(global Scalar *activation, global Scalar *y, global Scalar *derivative) {
    size_t i = get_global_id(0);
    derivative[i] = (activation[i] - y[i]) * derivative[i];
}

// The "responsibility" of the last layer with cross entropy error.
kernel void computeCrossEntropyLayerError(global Scalar *activation, global Scalar *y, global Scalar *derivative) {
    size_t i = get_global_id(0);
    derivative[i] = activation[i] - y[i];
}

// The "responsibility" of a layer in a NN (except for the last one).
// error = (W_next'*error_next .* derivative)
kernel void computeError(global Scalar *nextLayerWeights, global Scalar *nextLayerErrors, const uint nextLayerNeuronCount, const uint columns, global Scalar *derivatives) {
    size_t i = get_global_id(1);
    Scalar sum = 0;
    size_t offset = i; // index into the matrix column
    const global Scalar *nextLayerError = nextLayerErrors + get_global_id(0)*nextLayerNeuronCount;
    for (size_t j = 0; j < nextLayerNeuronCount; j++, offset+=columns) {
        sum += nextLayerWeights[offset] * nextLayerError[j];
    }
    derivatives[get_global_id(0)*get_global_size(1) + i] *= sum;
}

// gradients = error * input'
// bias gradients are just added
kernel void computeWeightGradient(global Scalar *errorTerm, global Scalar *input, global Scalar *weightGradients) {
    size_t row = get_global_id(0);
    size_t column = get_global_id(1);
    size_t columns = get_global_size(1);
    weightGradients[row*columns + column] += errorTerm[row] * input[column];
}

kernel void computeWeightGradient4(global Scalar *errorTerm, global Scalar4 *input, global Scalar4 *weightGradients) {
    size_t row = get_global_id(0);
    size_t column = get_global_id(1);
    size_t columns = get_global_size(1);
    weightGradients[row*columns + column] += errorTerm[row] * input[column];
}

kernel void computeWeightGradientParallel(global Scalar *errorTerm, global Scalar *input, global Scalar *weightGradients, const uint count) {
    size_t row = get_global_id(0);
    size_t rows = get_global_size(0);
    size_t column = get_global_id(1);
    size_t columns = get_global_size(1);
    Scalar sum = 0.0;
    for (size_t i = 0; i < count; ++i) {
        sum += errorTerm[i*rows + row] * input[i*columns + column];
    }
    weightGradients[row*columns + column] += sum;
}

kernel void computeWeightGradient4Parallel(global Scalar *errorTerm, global Scalar4 *input, global Scalar4 *weightGradients, const uint count) {
    size_t row = get_global_id(0);
    size_t rows = get_global_size(0);
    size_t column = get_global_id(1);
    size_t columns = get_global_size(1);
    Scalar4 sum = 0.0;
    for (size_t i = 0; i < count; ++i) {
        sum += errorTerm[i*rows + row] * input[i*columns + column];
    }
    weightGradients[row*columns + column] += sum;
}

kernel void computeBiasGradient(global Scalar *errorTerm, const uint count, global Scalar *biasGradients) {
    size_t i = get_global_id(0);
    Scalar sum = 0.0;
    const global Scalar *errorTermEl = errorTerm + i;
    for (size_t j = 0; j < count; ++j) {
        sum += errorTermEl[j*get_global_size(0)];
    }
    biasGradients[i] += sum;
}

kernel void evaluateClassification(global Scalar *outputs, const uint size, global ushort *labels, global uchar *dest) {
    size_t part = get_global_id(0);
    
    const global Scalar *output = outputs + size*(get_global_id(0) - get_global_offset(0));
    Scalar maxValue = output[0];
    size_t maxIndex = 0;
    for (size_t i = 1; i < size; ++i) {
        if (output[i] > maxValue) {
            maxValue = output[i];
            maxIndex = i;
        }
    }
    dest[part] = maxIndex == labels[part];
}

/*
TODO:

float parkMillerRandom(int *seed) {
    double const a    = 16807;      //ie 7**5
    double const m    = 2147483647; //ie 2**31-1
    double const reciprocal_m = 1.0/m;
    
    double temp = *seed * a;
    seed = (int)(temp - m * floor(temp * reciprocal_m));
    return float(temp * reciprocal_m);
}

kernel void dropout(global Scalar *x, global int* seeds, const float discardProbability) {
    size_t i = get_global_id(0);
    int seed = seeds[i];
    x[i] = parkMillerRandom(&seed);
    seeds[i] = seed;
}*/
