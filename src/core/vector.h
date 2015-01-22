#pragma once

#include <assert.h>
#include <vector>
#include "opencl.h"
#include "valueType.h"

namespace nnFit {

class TensorKernels {
public:
    TensorKernels(Device &device, std::ifstream &genericSource, std::ifstream &fixedSource);
    
    struct Specialization {
        Program program;
        Kernel constantMul;
        Kernel constantDiv;
        Kernel elementAdd;
        Kernel elementSub;
        Kernel elementMul;
        Kernel fill;
        Kernel partialSum;
        Kernel matrixIdentity;
        Kernel matrixVectorMul;
        Kernel matrixVectorMul4;
        
        Specialization(Device &device, std::ifstream &is);
    };
    Program program;
    Kernel partialTrueCount;
    Specialization floatKernels;
};

class VectorSlice {
public:
    VectorSlice(Device &device, const Storage &storage, size_t size, size_t offset, const ValueType &type);
    
    inline size_t size() const {
        return length;
    }
    
    inline size_t offset() const {
        return off;
    }
    
    inline const ValueType &type() const {
        return vtype;
    }
    
    void copy(Vector &dest) const;
    
private:
    Device &dev;
    StorageRef storage;
    size_t length, off;
    ValueType vtype;
};

class Vector {
public:
    Vector(Device &device, const ValueType &type = ValueType(ValueType::Float));
    Vector(Device &device, size_t size, const ValueType &type = ValueType(ValueType::Float));
    Vector(Device &device, std::initializer_list<float> init);
    Vector(Vector &&other);
    
    inline Device &device() const {
        return dev;
    }
    
    inline size_t size() const {
        return length;
    }
    
    const Storage &deviceStorage() const {
        return storage;
    }
    
    inline const ValueType &type() const {
        return vtype;
    }
    
    VectorSlice slice(size_t from) const;
    VectorSlice slice(size_t from, size_t to) const;
    
    void dump();
    void fill(float v);
    void ones();
    void zeros();
    
    template<typename T>
    void write(std::initializer_list<T> init) {
        assert(vtype == valueType<T>());
        write(init.begin(), init.size());
    }
    
    template<typename T>
    void write(const std::vector<T> &src) {
        assert(vtype == valueType<T>());
        write(src.data(), src.size());
    }
    
    void copy(Vector &dest) const;
    
    template<typename T>
    void copy(std::vector<T> &dest) const {
        assert(vtype == valueType<T>());
        if (dest.empty())
            dest.resize(length);
        read(dest.data(), dest.size());
    }
    
    void resize(size_t size);
private:
    void write(const void *data, size_t size);
    void read(void *data, size_t size) const;
    
    Vector(const Vector &) = delete;
    Device &dev;
    Storage storage;
    size_t length;
    ValueType vtype;
};

class Matrix: public Vector {
public:
    Matrix(Device &device);
    Matrix(Device &device, size_t rows, size_t columns);
    Matrix(Device &device, size_t rows, size_t columns, std::initializer_list<float> init);
    Matrix(Matrix &&other);
    
    size_t rows() const {
        return sizes[0];
    }
    size_t columns() const {
        return sizes[1];
    }
    
    // Return a slice containing row(s) of a matrix.
    VectorSlice row(size_t i, size_t count = 1) const;
    
    void identity();

    void resize(size_t rows, size_t columns);
private:
    Matrix(const Matrix &) = delete;
    size_t sizes[2];
};

// dest = x + y
void add(Vector &dest, const Vector &x, const Vector &y);
// x = x + y
void add(Vector &x, const Vector &y);
// dest = x - y
void sub(Vector &dest, const Vector &x, const Vector &y);
// x = x - y
void sub(Vector &x, const Vector &y);
// dest = x * k
void mul(Vector &dest, const Vector &x, float k);
// x = x * k
void mul(Vector &x, float k);
// dest = x / k
void div(Vector &dest, const Vector &x, float k);
// x = x / k
void div(Vector &x, float k);
// dest = x .* y
void elementwiseMul(Vector &dest, const Vector &x, const Vector &y);
// x = x .* y
void elementwiseMul(Vector &x, const Vector &y);

// Computes a partial sum
void partialSum(Vector &dest, const Vector &x);

// Counts the number of true values in x
// x must be a uint8 vector, while dest must be a uint32 vector
void partialTrueCount(Vector &dest, const Vector &x);

// Matrix by vector multiplication
// dest = x * y
void mul(Vector &dest, const Matrix &x, const Vector &y, const Range2D &workgroupSizes = Range2D());
    
} // namespace nnFit
