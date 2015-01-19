#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#ifdef __APPLE__
#include "OpenCL/opencl.h"
#else
#include "CL/cl.h"
#endif

namespace nnFit {

class Device;
class CommandQueue;
class Program;
class Kernel;
class Storage;
class StorageRef;
class Vector;
class TensorKernels;

class Device {
public:
    ~Device();
    Device(Device &&other);
    
    inline cl_device_id id() const {
        return device;
    }
    cl_context context();
    
    bool isGPU() const {
        return type == CL_DEVICE_TYPE_GPU;
    }
    
    bool isCPU() const {
        return type == CL_DEVICE_TYPE_CPU;
    }
    
    std::string name();
    std::string vendor();
    std::string version();
    
    size_t maxThreadsPerWorkgroup() const {
        return maxThreadsInWorkgroup;
    }
    
    void init();
    
    void queue(CommandQueue &q) {
        defaultQueue = &q;
    }
    
    CommandQueue &queue() {
        return *defaultQueue;
    }
    
    TensorKernels &tensorKernels() {
        return *tensorKernel;
    }
    
    void error(int errorCode, const char *msg);
    
    Program &getProgram(const char *name);
    
    static std::vector<Device> findAll();
    static std::vector<Device> findGPUs();
private:
    Device(const Device &) = delete;
    Device(cl_device_id device);
    
    cl_device_id device;
    CommandQueue *defaultQueue;
    cl_context ctx;
    cl_device_type type;
    size_t maxThreadsInWorkgroup;
    std::unique_ptr<TensorKernels> tensorKernel;
    std::unordered_map<std::string, std::unique_ptr<Program>> programs;
};

class CommandQueue {
public:
    CommandQueue(Device &device);
    ~CommandQueue();
    
    void enqueue1Dim(const Kernel &kernel, size_t size, size_t offset = 0);
    void enqueue2Dim(const Kernel &kernel, size_t rows, size_t columns);
    
    void fill(const Storage &dest, size_t size, size_t offset, const void *pattern, size_t patternSize);
    void copy(const StorageRef &src, const StorageRef &dest, size_t size, size_t srcOffset = 0, size_t destOffset = 0);
    void blockingRead(const Vector &src, void *dest, size_t size, size_t offset = 0);
    void blockingWrite(Vector &dest, const void *src, size_t size, size_t offset = 0);
    
    void finish();
    void flush();
private:
    CommandQueue(const CommandQueue &) = delete;
    Device &device;
    cl_command_queue queue;
};

class Program {
public:
    Program(Device &device, const char *src, size_t length);
    Program(Device &device, std::ifstream &is);
    Program(Program &&other);
    ~Program();
    
    inline cl_program id() const {
        return program;
    }
    Device &device() const {
        return dev;
    }
    
    void build();
private:
    Program(const Program &) = delete;
    Device &dev;
    cl_program program;
};

class Kernel {
public:
    Kernel();
    Kernel(Program &program, const char *name);
    Kernel(Kernel &&other);
    ~Kernel();
    
    inline cl_kernel id() const {
        return kernel;
    }
    
    Kernel &operator =(Kernel &&other);
    
    Kernel &setArg(unsigned i, const Vector &x);
    Kernel &setArg(unsigned i, float x);
    Kernel &setArg(unsigned i, double x);
    Kernel &setArg(unsigned i, size_t x);
    Kernel &allocateLocalMemory(unsigned i, size_t size);
private:
    Kernel(const Kernel &) = delete;
    cl_kernel kernel;
};

class Storage {
public:
    Storage();
    Storage(Device &device, size_t size, const void *data = nullptr);
    Storage(Storage &&other);
    ~Storage();
    
    Storage &operator = (Storage &&other);
    
    inline cl_mem id() const {
        return buffer;
    }
private:
    Storage(const Storage &) = delete;
    cl_mem buffer;
};

class StorageRef {
public:
    StorageRef(const Storage &storage) : buffer(storage.id()) { }
    
    inline cl_mem id() const {
        return buffer;
    }
private:
    cl_mem buffer;
};

} // namespace nnFit
