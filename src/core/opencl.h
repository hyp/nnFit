#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#ifdef __APPLE__
#include "OpenCL/opencl.h"
#else
#include "CL/cl.h"
#endif
#include "range.h"

namespace nnFit {

class Device;
class CommandQueue;
class Program;
class Kernel;
class KernelInvocation;
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
    
    double profile(std::function<void (void)> f);
    
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
    CommandQueue(Device &device, bool profile = false);
    ~CommandQueue();
    
    void enqueue1Dim(const KernelInvocation &kernel, size_t size, size_t offset = 0);
    void enqueue2Dim(const KernelInvocation &kernel, const Range2D &size, const Range2D &offset = Range2D());
    void enqueue2Dim(const KernelInvocation &kernel, const Range2D &size, const Range2D &offset, const Range2D &workgroupSize);
    void enqueue3Dim(const KernelInvocation &kernel, const Range3D &size, const Range3D &offset = Range3D());
    void enqueue3Dim(const KernelInvocation &kernel, const Range3D &size, const Range3D &offset, const Range3D &workgroupSize);
    
    void fill(const Storage &dest, size_t size, size_t offset, const void *pattern, size_t patternSize);
    void copy(const StorageRef &src, const StorageRef &dest, size_t size, size_t srcOffset = 0, size_t destOffset = 0);
    void blockingRead(const Storage &src, void *dest, size_t size, size_t offset = 0);
    void blockingWrite(const Storage &dest, const void *src, size_t size, size_t offset = 0);
    
    void finish();
    void flush();
    void dumpProfilingInfo();
    double totalKernelProfilingTime() const;
private:
    void enqueueKernel(const Kernel &kernel, unsigned dimensions, const size_t *globalSize, const size_t *globalOffset, const size_t *workgroupSize = nullptr);
    void profileKernel(cl_event event, const Kernel &kernel);
    CommandQueue(const CommandQueue &) = delete;
    
    Device &device;
    cl_command_queue queue;
    bool profile;
    struct ProfileInfo {
        size_t invocations;
        double totalTime;
        
        ProfileInfo() : invocations(0), totalTime(0) { }
    };
    std::unordered_map<std::string, ProfileInfo> profileRecords;
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
    
class KernelInvocation {
public:
    explicit KernelInvocation(const Kernel &kernel) : kernel(kernel), parameterId(0) { }
    
    void pushArg(const void *p, size_t size);
    
    template<typename T>
    KernelInvocation &pushArgs(const T &x) {
        return *this << x;
    }
    
    template<typename T, typename... Args>
    KernelInvocation &pushArgs(const T &first, const Args &...args) {
        *this << first;
        return pushArgs(args...);
    }
    
    const Kernel &kernel;
private:
    unsigned parameterId;
};
    
KernelInvocation &operator <<(KernelInvocation &kernel, float x);
    
KernelInvocation &operator <<(KernelInvocation &kernel, double x);
    
KernelInvocation &operator <<(KernelInvocation &kernel, size_t x);

class Kernel {
public:
    Kernel();
    Kernel(Program &program, const char *name);
    Kernel(Kernel &&other);
    ~Kernel();
    
    const char *kernelName() const {
        return name;
    }
    
    inline cl_kernel id() const {
        return kernel;
    }
    
    inline operator bool() const {
        return kernel != nullptr;
    }
    
    Kernel &operator =(Kernel &&other);
    
    template<typename T>
    KernelInvocation operator ()(const T &x) const {
        KernelInvocation invocation(*this);
        invocation << x;
        return invocation;
    }
    
    template<typename T, typename... Args>
    KernelInvocation operator ()(const T &x, const Args &...args) const {
        KernelInvocation invocation(*this);
        invocation << x;
        invocation.pushArgs(args...);
        return invocation;
    }
private:
    Kernel(const Kernel &) = delete;
    cl_kernel kernel;
    const char *name;
};
    
// LocalStorage - a utility structure that allow the user to allocate local memory
// for an OpenCL workgroup.
struct LocalStorage {
    size_t size;
        
    LocalStorage(size_t size) : size(size) { }
};
    
KernelInvocation &operator <<(KernelInvocation &kernel, LocalStorage x);

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
    
    // Shares the data with another storage object.
    void shareWith(Storage &other) const;
private:
    Storage(const Storage &) = delete;
    cl_mem buffer;
};
    
KernelInvocation &operator <<(KernelInvocation &kernel, const Storage &storage);

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
