#include <iostream>
#include "vector.h"

using namespace nnFit;

TensorKernels::Specialization::Specialization(Device &device, std::ifstream &is) : program(device, is) {
    program.build();
    constantMul = Kernel(program, "constantMul");
    constantDiv = Kernel(program, "constantDiv");
    elementAdd = Kernel(program, "elementAdd");
    elementAddParallel = Kernel(program, "elementAddParallel");
    elementSub = Kernel(program, "elementSub");
    elementMul = Kernel(program, "elementMul");
    fill = Kernel(program, "fill");
    partialSum = Kernel(program, "partialSum");
    matrixIdentity = Kernel(program, "matrixIdentity");
    matrixVectorMul = Kernel(program, "matrixVectorMul");
    matrixVectorMul4 = Kernel(program, "matrixVectorMul4");
    matrixVectorMulParallel = Kernel(program, "matrixVectorMulParallel");
    matrixVectorMul4Parallel = Kernel(program, "matrixVectorMul4Parallel");
    transposeMatrixVectorMulParallel = Kernel(program, "transposeMatrixVectorMulParallel");
}

TensorKernels::TensorKernels(Device &device, std::ifstream &genericSource, std::ifstream &fixedSource) : floatKernels(device, genericSource), program(device, fixedSource) {
    program.build();
    partialTrueCount = Kernel(program, "partialTrueCount");
}

VectorSlice::VectorSlice(Device &device, const Storage &storage, size_t size, size_t offset, const ValueType &type) : dev(device), storage(storage), length(size), off(offset), vtype(type) { }

Vector::Vector(Device &device, const ValueType &type) : dev(device), length(0), vtype(type) {
}

Vector::Vector(Device &device, size_t size, const ValueType &type) : dev(device), storage(device, size*type.size()), length(size), vtype(type) {
}

Vector::Vector(Device &device, std::initializer_list<float> init) : dev(device), storage(device, init.size()*sizeof(float), init.begin()), length(init.size()), vtype(ValueType::Float) {
}

Vector::Vector(Vector &&other) : dev(other.dev), storage(std::move(other.storage)), length(other.length), vtype(other.vtype) {
}

VectorSlice Vector::slice(size_t from) const {
    return slice(from, length);
}

VectorSlice Vector::slice(size_t from, size_t to) const {
    assert(from <= to);
    assert(to <= length);
    return VectorSlice(dev, storage, to - from, from, vtype);
}

void Vector::dump() const {
    assert(vtype == ValueType::Float);
    std::vector<float> v;
    copy(v);
    std::cout << "[ ";
    for(auto x : v) {
        std::cout << x << ", ";
    }
    std::cout << "]\n";
}

void Vector::fill(float v) const {
    assert(vtype == ValueType::Float);
    dev.queue().enqueue1Dim(dev.tensorKernels().floatKernels.fill(*this, v), length);
    // FIXME:
    // Doesn't work reliably on 2014 Macbook Pro with Nvidia GPU
    // dev.queue().fill(storage, length*sizeof(float), 0, &v, sizeof(float));
}

void Vector::ones() const {
    fill(1.0f);
}

void Vector::zeros() const {
    fill(0.0f);
}

void Vector::write(const void *data, size_t size) {
    assert(length == size);
    dev.queue().blockingWrite(*this, data, length*vtype.size());
}

void Vector::read(void *data, size_t size) const {
    assert(length == size);
    dev.queue().blockingRead(*this, data, length*vtype.size());
}

void Vector::copy(Vector &dest) const {
    assert(vtype == dest.vtype);
    assert(size() == dest.size());
    dev.queue().copy(storage, dest.storage, size()*vtype.size(), 0, 0);
}

void Vector::copy(const VectorSlice &dest) const {
    assert(vtype == dest.type());
    assert(size() == dest.size());
    dev.queue().copy(storage, dest.deviceStorage(), size()*vtype.size(), 0, dest.offset()*vtype.size());
}

void VectorSlice::copy(Vector &dest) const {
    assert(vtype == dest.type());
    assert(size() == dest.size());
    auto elementSize = vtype.size();
    dev.queue().copy(storage, dest.deviceStorage(), size()*elementSize, off*elementSize, 0);
}

void VectorSlice::copy(VectorSlice &dest) const {
    assert(vtype == dest.type());
    assert(size() == dest.size());
    auto elementSize = vtype.size();
    dev.queue().copy(storage, dest.storage, size()*elementSize, off*elementSize, dest.off*elementSize);
}

void Vector::resize(size_t size) {
    storage = std::move(Storage(dev, size*vtype.size()));
    length = size;
}

Matrix::Matrix(Device &device) : Vector(device) {
    sizes[0] = 0;
    sizes[1] = 0;
}

Matrix::Matrix(Device &device, size_t rows, size_t columns) : Vector(device, rows * columns) {
    sizes[0] = rows;
    sizes[1] = columns;
}

Matrix::Matrix(Device &device, size_t rows, size_t columns, std::initializer_list<float> init) : Vector(device, init) {
    assert(rows*columns == init.size());
    sizes[0] = rows;
    sizes[1] = columns;
}

Matrix::Matrix(Matrix &&other) : Vector(std::move(other)) {
    sizes[0] = other.sizes[0];
    sizes[1] = other.sizes[1];
}

VectorSlice Matrix::row(size_t i, size_t count) const {
    assert(i < rows());
    assert((i + count) <= rows());
    return slice(i*columns(), (i+count)*columns());
}

void Matrix::identity() {
    device().queue().enqueue2Dim(device().tensorKernels().floatKernels.matrixIdentity(*this, columns()), Range2D(sizes[0], sizes[1]));
}

void Matrix::resize(size_t rows, size_t columns) {
    Vector::resize(rows*columns);
    sizes[0] = rows;
    sizes[1] = columns;
}

static void exec(Kernel &kernel, const Vector &dest, const Vector &x, const Vector &y) {
    assert(dest.type() == x.type() && y.type() == x.type());
    assert(dest.size() == x.size() && y.size() == x.size());
    
    dest.device().queue().enqueue1Dim(kernel(x, y, dest), x.size());
}

static void exec(Kernel &kernel, const Vector &x, const Vector &y) {
    assert(y.type() == x.type());
    assert(y.size() == x.size());

    x.device().queue().enqueue1Dim(kernel(x, y, x), x.size());
}

static void exec(Kernel &kernel, const Vector &dest, const Vector &x, float y) {
    assert(dest.type() == x.type());
    assert(dest.size() == x.size());
    
    dest.device().queue().enqueue1Dim(kernel(x, y, dest), x.size());
}

static void exec(Kernel &kernel, const Vector &x, float y) {
    assert(x.type() == ValueType::Float);

    x.device().queue().enqueue1Dim(kernel(x, y, x), x.size());
}

// Selects a decent work group size for a row
static size_t selectRowPartion(size_t size) {
    if (size > 8) {
        // Check for divisibility from 32 to 2
        for (size_t i = 5; i > 0; --i) {
            auto x = 1<<i;
            if ((size & (x - 1)) == 0)
                return x;
        }
    }
    return 1;
}

static const size_t commonColumnPartions[] = { 32, 16, 10, 8, 7, 5, 4, 3 };
static const size_t commonColumnPartionsCount = sizeof(commonColumnPartions) / sizeof(commonColumnPartions[0]);

static size_t selectColumnPartion(size_t size) {
    for (size_t i = 0; i < commonColumnPartionsCount; ++i) {
        if ((size % commonColumnPartions[i]) == 0)
            return commonColumnPartions[i];
    }
    return 1;
}

namespace nnFit {

void add(const Vector &dest, const Vector &x, const Vector &y) {
    assert(x.type() == ValueType::Float);
    exec(dest.device().tensorKernels().floatKernels.elementAdd, dest, x, y);
}
    
void parallelAdd(const Vector &dest, const Vector &x, const Vector &y) {
    size_t vectorCount = dest.size() / x.size();
    if (vectorCount == 1) {
        return add(dest, x, y);
    }
    
    assert(x.type() == ValueType::Float);
    assert(dest.type() == x.type() && y.type() == x.type());
    assert(dest.size() == y.size());
    assert((dest.size() % x.size()) == 0);
           
    auto task = dest.device().tensorKernels().floatKernels.elementAddParallel(x, y, dest);
    dest.device().queue().enqueue2Dim(task, Range2D(vectorCount, x.size()));
}

void add(const Vector &x, const Vector &y) {
    assert(x.type() == ValueType::Float);
    exec(x.device().tensorKernels().floatKernels.elementAdd, x, y);
}

void sub(const Vector &dest, const Vector &x, const Vector &y) {
    assert(x.type() == ValueType::Float);
    exec(dest.device().tensorKernels().floatKernels.elementSub, dest, x, y);
}

void sub(const Vector &x, const Vector &y) {
    assert(x.type() == ValueType::Float);
    exec(x.device().tensorKernels().floatKernels.elementSub, x, y);
}

void mul(const Vector &dest, const Vector &x, float k) {
    assert(x.type() == ValueType::Float);
    exec(dest.device().tensorKernels().floatKernels.constantMul, dest, x, k);
}

void mul(const Vector &x, float k) {
    assert(x.type() == ValueType::Float);
    exec(x.device().tensorKernels().floatKernels.constantMul, x, k);
}

void div(const Vector &dest, const Vector &x, float k) {
    assert(x.type() == ValueType::Float);
    exec(dest.device().tensorKernels().floatKernels.constantDiv, dest, x, k);
}

void div(const Vector &x, float k) {
    assert(x.type() == ValueType::Float);
    exec(x.device().tensorKernels().floatKernels.constantDiv, x, k);
}

void elementwiseMul(const Vector &dest, const Vector &x, const Vector &y) {
    assert(x.type() == ValueType::Float);
    exec(dest.device().tensorKernels().floatKernels.elementMul, dest, x, y);
}

void elementwiseMul(const Vector &x, const Vector &y) {
    assert(x.type() == ValueType::Float);
    exec(x.device().tensorKernels().floatKernels.elementMul, x, y);
}

void partialSum(const Vector &dest, const Vector &x) {
    assert(x.type() == ValueType::Float);
    assert(dest.type() == x.type());
    
    size_t partCount = dest.size();
    size_t partSize = x.size()/partCount + (x.size()%partCount == 0? 0 : 1);
    
    x.device().queue().enqueue1Dim(x.device().tensorKernels().floatKernels.partialSum(x, x.size(), partSize, dest), partCount);
}

void partialTrueCount(const Vector &dest, const Vector &x) {
    assert(x.type() == ValueType::Uint8);
    assert(dest.type() == ValueType::Uint32);
    
    size_t partCount = dest.size();
    size_t partSize = x.size()/partCount + (x.size()%partCount == 0? 0 : 1);
    
    x.device().queue().enqueue1Dim(x.device().tensorKernels().partialTrueCount(x, x.size(), partSize, dest), partCount);
}
    
void mvmul(const Vector &dest, const Matrix &x, const Vector &y, const Range2D &workgroupSizes) {
    // Compute workgroup
    size_t rowsPerWorkgroup = workgroupSizes[0];
    size_t parts = workgroupSizes[1];
    if (parts == 0) {
        parts = selectRowPartion(x.columns());
        rowsPerWorkgroup = selectColumnPartion(x.rows());
    }
    
    assert(x.type() == y.type());
    assert(x.type() == dest.type());
    assert(x.columns() == y.size());
    assert(x.rows() == dest.size());
    
    size_t partSize = x.columns()/parts;
    if (partSize % 4 == 0) {
        assert(x.columns() % 4 == 0);
        auto &kernel = x.device().tensorKernels().floatKernels.matrixVectorMul4;
        x.device().queue().enqueue2Dim(kernel(x, y, x.columns()/4, partSize/4, dest, LocalStorage(parts*rowsPerWorkgroup*x.type().size())), Range2D(x.rows(), parts), Range2D(), Range2D(rowsPerWorkgroup, parts));
        return;
    }
    
    // Shedule
    auto &kernel = x.device().tensorKernels().floatKernels.matrixVectorMul;
    x.device().queue().enqueue2Dim(kernel(x, y, x.columns(), partSize, dest, LocalStorage(parts*rowsPerWorkgroup*x.type().size())), Range2D(x.rows(), parts), Range2D(), Range2D(rowsPerWorkgroup, parts));
}
    
void parallelMvmul(const Vector &dest, const Matrix &x, const Vector &y, const Range2D &workgroupSizes) {
    size_t vectorCount = y.size() / x.columns();
    if (vectorCount == 1) {
        return mvmul(dest, x, y, workgroupSizes);
    }
    
    // Compute workgroup
    size_t rowsPerWorkgroup = workgroupSizes[0];
    size_t parts = workgroupSizes[1];
    if (parts == 0) {
        parts = selectRowPartion(x.columns());
        rowsPerWorkgroup = selectColumnPartion(x.rows());
    }
    
    assert(x.type() == y.type());
    assert(x.type() == dest.type());
    assert((y.size() % x.columns()) == 0);
    assert((dest.size() % x.rows()) == 0);
    assert(vectorCount == dest.size() / x.rows());
    
    size_t partSize = x.columns()/parts;
    if (partSize % 4 == 0) {
        assert(x.columns() % 4 == 0);
        auto &kernel = x.device().tensorKernels().floatKernels.matrixVectorMul4Parallel;
        x.device().queue().enqueue3Dim(kernel(x, y, x.columns()/4, partSize/4, dest, LocalStorage(parts*rowsPerWorkgroup*x.type().size())), Range3D(vectorCount, x.rows(), parts), Range3D(), Range3D(1, rowsPerWorkgroup, parts));
        return;
    }
    
    auto &kernel = x.device().tensorKernels().floatKernels.matrixVectorMulParallel;
    x.device().queue().enqueue3Dim(kernel(x, y, x.columns(), partSize, dest, LocalStorage(parts*rowsPerWorkgroup*x.type().size())), Range3D(vectorCount, x.rows(), parts), Range3D(), Range3D(1, rowsPerWorkgroup, parts));
}
    
void transposeMvmul(const Vector &dest, const Matrix &x, const Vector &y, size_t vectorCount) {
    assert(x.columns() * vectorCount == dest.size());
    assert(x.rows() * vectorCount == y.size());
    
    auto task = x.device().tensorKernels().floatKernels.transposeMatrixVectorMulParallel(x, y, x.rows(), dest);
    x.device().queue().enqueue2Dim(task, Range2D(vectorCount, x.columns()));
}
    
} // namespace nnFit
