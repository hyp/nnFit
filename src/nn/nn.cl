#pragma OPENCL EXTENSION cl_khr_fp64 : enable

typedef float Scalar;

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
kernel void computeError(global Scalar *nextLayerWeights, global Scalar *nextLayerError, const uint nextLayerNeuronCount, const uint columns, global Scalar *derivative) {
    size_t i = get_global_id(0);
    Scalar sum = 0;
    size_t offset = i; // index into the matrix column
    for (size_t j = 0; j < nextLayerNeuronCount; j++, offset+=columns) {
        sum += nextLayerWeights[offset] * nextLayerError[j];
    }
    derivative[i] *= sum;
}

// gradients = error * input'
// bias gradients are just added
kernel void computeWeightGradient(global Scalar *errorTerm, global Scalar *input, global Scalar *weightGradients) {
    size_t row = get_global_id(0);
    size_t column = get_global_id(1);
    size_t columns = get_global_size(1);
    weightGradients[row*columns + column] += errorTerm[row] * input[column];
}


kernel void evaluateClassification(global Scalar *output, const uint n, const uint size, global ushort *labels, global uchar *dest) {
    size_t part = n;//Later: get_global_id(0);
    
    size_t i = 0, end = i + size;
    Scalar maxValue = output[i];
    size_t maxIndex = i;
    for (++i; i < end; ++i) {
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
