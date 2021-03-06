#include <iostream>
#include <fstream>
#include "opencl.h"
#include "vector.h"

using namespace nnFit;

Device::Device(cl_device_id device) : device(device), ctx(nullptr) {
    auto error = clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(type), &type, nullptr);
    if (error != CL_SUCCESS) {
        type = CL_DEVICE_TYPE_DEFAULT;
    }
    maxThreadsInWorkgroup = 0;
}

Device::Device(Device &&other) : device(std::move(other.device)), ctx(std::move(other.ctx)), type(other.type), maxThreadsInWorkgroup(other.maxThreadsInWorkgroup), tensorKernel(std::move(other.tensorKernel)), programs(std::move(other.programs)) {
    other.device = nullptr;
    other.ctx = nullptr;
}

Device::~Device() {
    if (ctx)
        clReleaseContext(ctx);
}

void Device::init() {
    std::ifstream generic("generic.cl");
    std::ifstream fixed("fixed.cl");
    tensorKernel.reset(new TensorKernels(*this, generic, fixed));
    
    auto errorCode = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxThreadsInWorkgroup), &maxThreadsInWorkgroup, nullptr);
    if (errorCode != CL_SUCCESS) {
        error(errorCode, "Failed to get the max work group size");
    }
}

cl_context Device::context() {
    if (ctx)
        return ctx;
    
    cl_uint platformIdCount = 0;
    clGetPlatformIDs (0, nullptr, &platformIdCount);
    
    std::vector<cl_platform_id> platformIds (platformIdCount);
    clGetPlatformIDs (platformIdCount, platformIds.data (), nullptr);
    
    const cl_context_properties contextProperties [] =
    {
        CL_CONTEXT_PLATFORM,
        reinterpret_cast<cl_context_properties> (platformIds [0]),
        0, 0
    };
    
    cl_int error;
    ctx = clCreateContext (contextProperties, 1, &device, nullptr, nullptr, &error);
    if (!ctx || error != CL_SUCCESS) {
        this->error(error, "Failed to create context");
    }
    return ctx;
}

static std::string getDeviceString(cl_device_id device, cl_device_info value) {
    size_t size;
    auto error = clGetDeviceInfo(device, value, 0, nullptr, &size);
    std::string result(size, ' ');
    error |= clGetDeviceInfo(device, value, size, const_cast<char*>(result.data()), nullptr);
    if (error != CL_SUCCESS) {
        return "Unknown";
    }
    return result;
}

std::string Device::name() {
    return getDeviceString(device, CL_DEVICE_NAME);
}

std::string Device::vendor() {
    return getDeviceString(device, CL_DEVICE_VENDOR);
}

std::string Device::version() {
    return getDeviceString(device, CL_DEVICE_VERSION);
}

void Device::error(int errorCode, const char *msg) {
    std::cerr << "OpenCL error (" << errorCode << "): " << msg << "\n";
}

Program &Device::getProgram(const char *name) {
    std::string key(name);
    if (programs.find(key) == programs.end()) {
        std::ifstream is(name);
        std::unique_ptr<Program> p(new Program(*this, is));
        p->build();
        auto &result = *p;
        programs.insert(std::make_pair(key, std::move(p)));
        return result;
    }
    return *programs[key];
}

std::vector<Device> Device::findAll() {
    cl_uint platformIdCount = 0;
    clGetPlatformIDs (0, nullptr, &platformIdCount);
    
    std::vector<cl_platform_id> platformIds (platformIdCount);
    clGetPlatformIDs (platformIdCount, platformIds.data (), nullptr);
    
    cl_uint deviceIdCount = 0;
    clGetDeviceIDs (platformIds [0], CL_DEVICE_TYPE_ALL, 0, nullptr,
                    &deviceIdCount);
    std::vector<cl_device_id> deviceIds (deviceIdCount);
    clGetDeviceIDs (platformIds [0], CL_DEVICE_TYPE_ALL, deviceIdCount,
                    deviceIds.data (), nullptr);
    
    std::vector<Device> devices;
    for (auto id : deviceIds) {
        devices.push_back(Device(id));
    }
    
    return devices;
}

std::vector<Device> Device::findGPUs() {
    cl_uint platformIdCount = 0;
    clGetPlatformIDs (0, nullptr, &platformIdCount);
    
    std::vector<cl_platform_id> platformIds (platformIdCount);
    clGetPlatformIDs (platformIdCount, platformIds.data (), nullptr);
    
    cl_uint deviceIdCount = 0;
    clGetDeviceIDs (platformIds [0], CL_DEVICE_TYPE_GPU, 0, nullptr,
                    &deviceIdCount);
    std::vector<cl_device_id> deviceIds (deviceIdCount);
    clGetDeviceIDs (platformIds [0], CL_DEVICE_TYPE_GPU, deviceIdCount,
                    deviceIds.data (), nullptr);
    
    std::vector<Device> devices;
    for (auto id : deviceIds) {
        devices.push_back(Device(id));
    }
    
    return devices;
}

double Device::profile(std::function<void (void)> f) {
    CommandQueue q(*this, true);
    auto prevQueue = defaultQueue;
    defaultQueue = &q;
    f();
    defaultQueue = prevQueue;
    return q.totalKernelProfilingTime();
}

CommandQueue::CommandQueue(Device &device, bool profile) : device(device), profile(profile) {
    cl_int error = 0;
    queue = clCreateCommandQueue(device.context(), device.id(), profile? CL_QUEUE_PROFILING_ENABLE : 0, &error);
    if (!queue || error != CL_SUCCESS) {
        device.error(error, "Failed to create command queue");
        queue = nullptr;
    }
}

CommandQueue::~CommandQueue() {
    if (queue)
        clReleaseCommandQueue(queue);
}

void CommandQueue::dumpProfilingInfo() {
    if (profile) {
        for (auto &record : profileRecords) {
            auto info = record.second;
            std::cout<< "Kernel '" << record.first << "' was called " << info.invocations << " times " << info.totalTime << "ms " << " avg " << (info.totalTime/double(info.invocations)) << "ms\n";
        }
    }
}

double CommandQueue::totalKernelProfilingTime() const {
    double total = 0;
    for (auto &record : profileRecords) {
        total += record.second.totalTime;
    }
    return total;
}

void CommandQueue::profileKernel(cl_event event, const Kernel &kernel) {
    auto error = clWaitForEvents(1, &event);
    if (error != CL_SUCCESS) {
        device.error(error, "Failed to wait for an event");
    }
    cl_ulong start = 0, end = 0;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, nullptr);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, nullptr);
    auto time = (cl_double)(end - start)*(cl_double)(1e-06);
    std::string key(kernel.kernelName());
    auto i = profileRecords.insert(std::make_pair(key, ProfileInfo()));
    i.first->second.invocations++;
    i.first->second.totalTime += time;
}

void CommandQueue::enqueueKernel(const Kernel &kernel, unsigned dimensions, const size_t *globalSize, const size_t *globalOffset, const size_t *workgroupSize) {
    cl_event event = nullptr;
    auto error = clEnqueueNDRangeKernel(queue, kernel.id(), dimensions, globalOffset, globalSize, workgroupSize, 0, nullptr, profile? &event : nullptr);
    if (error != CL_SUCCESS) {
        device.error(error, "Failed to enqueue a kernel");
    } else if (profile) {
        profileKernel(event, kernel);
    }
}

void CommandQueue::enqueue1Dim(const KernelInvocation &kernel, size_t size, size_t offset) {
    size_t sizes[] = { size, 0, 0 };
    size_t offsets[] = { offset, 0, 0 };
    enqueueKernel(kernel.kernel, 1, sizes, offsets);
}

void CommandQueue::enqueue2Dim(const KernelInvocation &kernel, const Range2D &size, const Range2D &offset) {
    size_t sizes[] = { size[0], size[1], 0 };
    size_t offsets[] = { offset[0], offset[1], 0 };
    enqueueKernel(kernel.kernel, 2, sizes, offsets);
}

void CommandQueue::enqueue2Dim(const KernelInvocation &kernel, const Range2D &size, const Range2D &offset, const Range2D &workgroupSize) {
    size_t sizes[] = { size[0], size[1], 0 };
    size_t offsets[] = { offset[0], offset[1], 0 };
    size_t localSizes[] = { workgroupSize[0], workgroupSize[1], 0 };
    enqueueKernel(kernel.kernel, 2, sizes, offsets, localSizes);
}

void CommandQueue::enqueue3Dim(const KernelInvocation &kernel, const Range3D &size, const Range3D &offset) {
    size_t sizes[] = { size[0], size[1], size[2] };
    size_t offsets[] = { offset[0], offset[1], offset[2] };
    enqueueKernel(kernel.kernel, 3, sizes, offsets);
}

void CommandQueue::enqueue3Dim(const KernelInvocation &kernel, const Range3D &size, const Range3D &offset, const Range3D &workgroupSize) {
    size_t sizes[] = { size[0], size[1], size[2] };
    size_t offsets[] = { offset[0], offset[1], offset[2] };
    size_t localSizes[] = { workgroupSize[0], workgroupSize[1], workgroupSize[2] };
    enqueueKernel(kernel.kernel, 3, sizes, offsets, localSizes);
}

void CommandQueue::fill(const Storage &dest, size_t size, size_t offset, const void *pattern, size_t patternSize) {
    auto error = clEnqueueFillBuffer(queue, dest.id(), pattern, patternSize, offset, size, 0, nullptr, nullptr);
    if (error != CL_SUCCESS) {
        device.error(error, "Failed to fill a buffer");
    }
}

void CommandQueue::copy(const StorageRef &src, const StorageRef &dest, size_t size, size_t srcOffset, size_t destOffset) {
    auto error = clEnqueueCopyBuffer(queue, src.id(), dest.id(), srcOffset, destOffset, size, 0, nullptr, nullptr);
    if (error != CL_SUCCESS) {
        device.error(error, "Failed to copy a buffer");
    }
}

void CommandQueue::blockingRead(const Storage &src, void *dest, size_t size, size_t offset) {
    auto error = clEnqueueReadBuffer(queue, src.id(), CL_TRUE, offset, size, dest, 0, nullptr, nullptr);
    if (error != CL_SUCCESS) {
        device.error(error, "Failed to read a buffer");
    }
}

void CommandQueue::blockingWrite(const Storage &dest, const void *src, size_t size, size_t offset) {
    auto error = clEnqueueWriteBuffer(queue, dest.id(), CL_TRUE, offset, size, src, 0, nullptr, nullptr);
    if (error != CL_SUCCESS) {
        device.error(error, "Failed to write to a buffer");
    }
}

void CommandQueue::finish() {
    clFinish(queue);
}

void CommandQueue::flush() {
    clFlush(queue);
}

static cl_program initProgram(Device &device, const char *src, size_t length) {
    const char *sources[] = { src };
    cl_int error;
    auto program = clCreateProgramWithSource(device.context(), 1, sources, &length, &error);
    if (!program || error != CL_SUCCESS) {
        device.error(error, "Failed to create program");
    }
    return program;
}

Program::Program(Device &device, const char *src, size_t length) : dev(device) {
    program = initProgram(device, src, length);
}

Program::Program(Device &device, std::ifstream &is) : dev(device) {
    is.seekg(0, std::ios::end);
    size_t size = is.tellg();
    std::vector<char> buffer(size+1, ' ');
    is.seekg(0);
    is.read(buffer.data(), size);
    buffer[size] = '\0';
    program = initProgram(device, buffer.data(), size);
}

Program::Program(Program &&other) : dev(other.dev), program(std::move(other.program)) {
    other.program = nullptr;
}

Program::~Program() {
    if (program)
        clReleaseProgram(program);
}

void Program::build() {
    auto deviceId = dev.id();
    auto error = clBuildProgram(program, 1, &deviceId, "", nullptr, nullptr);
    if (error != CL_SUCCESS) {
        dev.error(error, "Failed to build program");
    }
}

Kernel::Kernel() : kernel(nullptr), name("") { }

Kernel::Kernel(Program &program, const char *name) {
    cl_int error;
    kernel = clCreateKernel(program.id(), name, &error);
    this->name = name;
    if (!kernel || error != CL_SUCCESS) {
        program.device().error(error, "Failed to create kernel");
    }
}

Kernel::Kernel(Kernel &&other) : kernel(std::move(other.kernel)), name(other.name) {
    other.kernel = nullptr;
}

Kernel::~Kernel() {
    if (kernel)
        clReleaseKernel(kernel);
}

Kernel &Kernel::operator =(Kernel &&other) {
    kernel = std::move(other.kernel);
    name = other.name;
    other.kernel = nullptr;
    return *this;
}

void KernelInvocation::pushArg(const void *p, size_t size) {
    clSetKernelArg(kernel.id(), parameterId, size, p);
    parameterId++;
}

Storage::Storage() : buffer(nullptr) {
}

Storage::Storage(Device &device, size_t size, const void *data) {
    cl_int error;
    buffer = clCreateBuffer(device.context(), CL_MEM_READ_WRITE | (data == nullptr? 0 : CL_MEM_COPY_HOST_PTR), size, const_cast<void*>(data), &error);
    if (!buffer || error != CL_SUCCESS) {
        device.error(error, "Failed to create buffer");
    }
}

Storage::Storage(Storage &&other) : buffer(std::move(other.buffer)) {
    other.buffer = nullptr;
}

Storage::~Storage() {
    if (buffer)
        clReleaseMemObject(buffer);
}

Storage &Storage::operator = (Storage &&other) {
    if (buffer)
        clReleaseMemObject(buffer);
    buffer = other.buffer;
    other.buffer = nullptr;
    return *this;
}

void Storage::shareWith(Storage &other) const {
    if (other.buffer)
        clReleaseMemObject(other.buffer);
    clRetainMemObject(buffer);
    other.buffer = buffer;
}

namespace nnFit {
    
KernelInvocation &operator <<(KernelInvocation &kernel, float x) {
    kernel.pushArg(&x, sizeof(x));
    return kernel;
}
    
KernelInvocation &operator <<(KernelInvocation &kernel, double x) {
    kernel.pushArg(&x, sizeof(x));
    return kernel;
}
    
KernelInvocation &operator <<(KernelInvocation &kernel, size_t n) {
    assert(n <= std::numeric_limits<cl_uint>::max());
    cl_uint x = (cl_uint)n;
    kernel.pushArg(&x, sizeof(x));
    return kernel;
}

KernelInvocation &operator <<(KernelInvocation &kernel, LocalStorage x) {
    kernel.pushArg(nullptr, x.size);
    return kernel;
}
    
KernelInvocation &operator <<(KernelInvocation &kernel, const Storage &storage) {
    auto mem = storage.id();
    kernel.pushArg(&mem, sizeof(mem));
    return kernel;
}
    
}