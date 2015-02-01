
// Use the generator from the book 'GPU Gems 3', Chapter 37.
// Source: http://http.developer.nvidia.com/GPUGems3/gpugems3_ch37.html
struct RandomResult {
    uint4 state;
    float value;
};

uint tausStep(uint z, int S1, int S2, int S3, uint M) {
    uint b = (((z << S1) ^ z) >> S2);
    return (((z & M) << S3) ^ b);
}

uint lcgStep(uint z, uint A, uint C) {
    return (A * z + C);
}

struct RandomResult random(uint4 state) {
    state.x = tausStep(state.x, 13, 19, 12, 4294967294);
    state.y = tausStep(state.y, 2, 25, 4, 4294967288);
    state.z = tausStep(state.z, 3, 11, 17, 4294967280);
    state.w = lcgStep(state.w, 1664525, 1013904223);
    
    struct RandomResult result;
    result.state = state;
    result.value = 2.3283064365387e-10 * (state.x ^ state.y ^ state.z ^ state.w);
    
    return result;
}

kernel void uniformRandom(global float *x, global uint4 *state) {
    size_t i = get_global_id(0);
    struct RandomResult result = random(state[i]);
    state[i] = result.state;
    x[i] = result.value;
}

typedef float Scalar;

kernel void invertedDropout(global Scalar *x, global uint4 *state, const float activationProbability) {
    size_t i = get_global_id(0);
    struct RandomResult result = random(state[i]);
    state[i] = result.state;
    x[i] *= (result.value < activationProbability? (Scalar)1.0 : (Scalar)0.0) / activationProbability;
}