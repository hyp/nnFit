// Basic linear algebra

typedef float Scalar;

kernel void fill(global Scalar *dest, const Scalar value) {
    size_t i = get_global_id(0);
    dest[i] = value;
}

// dest = x * k
kernel void constantMul(global Scalar *x, const Scalar k, global Scalar *dest) {
    size_t i = get_global_id(0);
    dest[i] = x[i] * k;
}

// dest = x / k
kernel void constantDiv(global Scalar *x, const Scalar k, global Scalar *dest) {
    size_t i = get_global_id(0);
    dest[i] = x[i] / k;
}

// dest = x - y
kernel void elementSub(global Scalar *x, global Scalar *y, global Scalar *dest) {
    size_t i = get_global_id(0);
    dest[i] = x[i] - y[i];
}

// dest = x + y
kernel void elementAdd(global Scalar *x, global Scalar *y, global Scalar *dest) {
    size_t i = get_global_id(0);
    dest[i] = x[i] + y[i];
}

// dest = x * y
kernel void elementMul(global Scalar *x, global Scalar *y, global Scalar *dest) {
    size_t i = get_global_id(0);
    dest[i] = x[i] * y[i];
}

kernel void partialSum(global Scalar *x, const uint size, const uint partSize, global Scalar *dest) {
    uint part = get_global_id(0);
    
    Scalar sum = 0.0f;
    for (size_t i = part*partSize, end = min(part*partSize + partSize, size); i < end; ++i)
        sum += x[i];
    dest[part] = sum;
}

