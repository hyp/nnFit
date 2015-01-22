// Basic linear algebra

typedef float Scalar;
typedef float4 Scalar4;

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

kernel void matrixIdentity(global Scalar *matrix, const uint columns) {
    size_t i = get_global_id(0);
    size_t j = get_global_id(1);
    matrix[i * columns + j] = i == j? 1.0 : 0.0;
}

kernel void matrixVectorMul(global Scalar *matrix, global Scalar *vector, const uint columns, const uint partSize, global Scalar *output, local Scalar *work) {
    // Compute partial dot product
    size_t i = get_global_id(0);
    size_t j = get_global_id(1);
    size_t parts = get_global_size(1);
    size_t k = j*partSize;
    
    size_t offset = i*columns + k;
    Scalar partialSum = 0;
    for (size_t end = k + partSize; k < end; k++, offset++) {
        partialSum += matrix[offset] * vector[k];
    }
    
    // Store the partial result in local work memory
    size_t ii = get_local_id(0);
    size_t jj = get_local_id(1);
    size_t workColumns = get_local_size(1);
    size_t workRowOffset = ii * workColumns;
    work[workRowOffset + jj] = partialSum;
    
    // Wait until all the threads in this group have computed their partial sums
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // The first thread in a given row sums all the partial sums produced by itself
    // and the other threads in this row.
    if (jj == 0) {
        Scalar sum = 0;
        for (size_t k = workRowOffset, end = workRowOffset + workColumns; k < end; ++k) {
            sum += work[k];
        }
        output[i] = sum;
    }
}

kernel void matrixVectorMul4(global Scalar4 *matrix, global Scalar4 *vector, const uint columns, const uint partSize, global Scalar *output, local Scalar *work) {
    // Compute partial dot product
    size_t i = get_global_id(0);
    size_t j = get_global_id(1);
    size_t parts = get_global_size(1);
    size_t k = j*partSize;
    
    size_t offset = i*columns + k;
    Scalar partialSum = 0;
    for (size_t end = k + partSize; k < end; k++, offset++) {
        partialSum += dot(matrix[offset], vector[k]);
    }
    
    // Store the partial result in local work memory
    size_t ii = get_local_id(0);
    size_t jj = get_local_id(1);
    size_t workColumns = get_local_size(1);
    size_t workRowOffset = ii * workColumns;
    work[workRowOffset + jj] = partialSum;
    
    // Wait until all the threads in this group have computed their partial sums
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // The first thread in a given row sums all the partial sums produced by itself
    // and the other threads in this row.
    if (jj == 0) {
        Scalar sum = 0;
        for (size_t k = workRowOffset, end = workRowOffset + workColumns; k < end; ++k) {
            sum += work[k];
        }
        output[i] = sum;
    }
}

