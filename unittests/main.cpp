
#include <iostream>
#include <fstream>
#include "core/opencl.h"
#include "core/vector.h"
#include "nn/network.h"
#include "nn/trainer.h"
#include "nn/errorCriterion.h"
#include "nn/classificationEvaluator.h"
#include "optimizers/gradientDescent.h"
#include "mnistDataset.h"

using namespace nnFit;

template<typename T>
void assertEquals(const std::vector<T> &x, std::initializer_list<T> y) {
    assert(x.size() == y.size());
    for (size_t i = 0; i < std::min(x.size(), y.size()); ++i) {
        assert(x[i] == y.begin()[i]);
    }
}

template<typename T>
void assertEquals(const Vector &x, std::initializer_list<T> y) {
    std::vector<T> dest;
    x.copy(dest);
    x.device().queue().finish();
    assertEquals(dest, y);
}

static Device selectDevice() {
    auto devices = Device::findGPUs();
    for (auto &device : devices) {
        if (device.vendor().find("NVIDIA") != std::string::npos)
            return std::move(device);
        else if (device.vendor().find("AMD") != std::string::npos)
            return std::move(device);
    }
    if (devices.empty())
        return std::move(Device::findAll()[0]);
    return std::move(devices[0]);
}

void testVectors(Device &device) {
    Vector x(device, {1.0f,2.0f,3.0f,4.0f});
    Vector y(device, {0.0f,1.0f,5.0f,10.0f});
    Vector dest(device, x.size());
    
    add(dest, x, y);
    assertEquals(dest, {1.0f,3.0f,8.0f,14.0f});
    sub(dest, x, y);
    assertEquals(dest, {1.0f,1.0f,-2.0f,-6.0f});
    mul(dest, x, 2.0f);
    assertEquals(dest, {2.0f,4.0f,6.0f,8.0f});
    div(dest, x, 2.0f);
    assertEquals(dest, {0.5f,1.0f,1.5f,2.0f});
    elementwiseMul(dest, x, y);
    assertEquals(dest, {0.0f,2.0f,15.0f,40.0f});
    
    x.copy(dest);
    assertEquals(dest, {1.0f,2.0f,3.0f,4.0f});
    add(dest, y);
    assertEquals(dest, {1.0f,3.0f,8.0f,14.0f});
    
    for (int i = 0; i < 100; ++i) {
        dest.zeros();
        assertEquals(dest, {0.0f,0.0f,0.0f,0.0f});
        dest.ones();
        assertEquals(dest, {1.0f,1.0f,1.0f,1.0f});
    }
}

void testSum(Device &device) {
    Vector x(device, {1.0f,2.0f,3.0f,4.0f});
    Vector single(device, 1);
    partialSum(single, x);
    assertEquals(single, {10.0f});
    
    Vector y(device, {1.0f,2.0f,3.0f,4.0f,5.0f,6.0f,7.0f,8.0f,9.0f,10.0f});
    partialSum(x, y);
    assertEquals(x, {6.0f, 15.0f, 24.0f, 10.0f});
    
    Vector z(device, {1.0f,2.0f,3.0f,4.0f,5.0f,6.0f,7.0f,8.0f,9.0f,10.0f,11.0f,12.0f});
    partialSum(x, z);
    assertEquals(x, {6.0f, 15.0f, 24.0f, 33.0f});
}

void testBooleanOperations(Device &device) {
    Vector x(device, 4, ValueType(ValueType::Uint8));
    x.write({ uint8_t(0), uint8_t(1), uint8_t(1), uint8_t(0) });
    Vector single(device, 1, ValueType(ValueType::Uint32));
    partialTrueCount(single, x);
    assertEquals(single, { uint32_t(2) });
    
    Vector y(device, 10, ValueType(ValueType::Uint8));
    y.write({ uint8_t(0), uint8_t(1), uint8_t(1), uint8_t(0), uint8_t(0), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(0), uint8_t(0) });
    Vector result(device, 4, ValueType(ValueType::Uint32));
    partialTrueCount(result, y);
    assertEquals(result, { uint32_t(2), uint32_t(1), uint32_t(2), uint32_t(0) });
    
    Vector z(device, 12, ValueType(ValueType::Uint8));
    z.write({ uint8_t(0), uint8_t(1), uint8_t(1), uint8_t(0), uint8_t(0), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(0), uint8_t(0), uint8_t(1), uint8_t(0) });
    partialTrueCount(result, z);
    assertEquals(result, { uint32_t(2), uint32_t(1), uint32_t(2), uint32_t(1) });
}

void testLayers(Device &device) {
    Network net(device);
    Layer layer(device, 2, 2, Layer::Linear);
    layer.neuronWeights().write({ 1.0f, 1.0f, 2.0f, 0.5f });
    layer.neuronBiases().write({ 0.0f, 1.0f });
    Vector input(device, { 1.0f, 2.0f });
    auto &output = layer.predict(net.context(), input);
    assertEquals(output, { 3.0f, 4.0f });
}

void assertEquals(const Vector &x, bool y) {
    std::vector<float> dest;
    x.copy(dest);
    x.device().queue().finish();
    assert(dest[0] > 0.5 == y);
};

void testLogicGates(Device &device) {
    Vector t00(device, {0.0f, 0.0f});
    Vector t10(device, {1.0f, 0.0f});
    Vector t01(device, {0.0f, 1.0f});
    Vector t11(device, {1.0f, 1.0f});
    
    auto assertEquals = [] (const Vector &x, bool y) {
        std::vector<float> dest;
        x.copy(dest);
        x.device().queue().finish();
        assert(dest[0] > 0.5 == y);
    };
    
    {
        // OR gate
        Network net(device);
        std::unique_ptr<Layer> layer(new Layer(device, 1, 2, Layer::Sigmoid));
        layer->neuronWeights().write({ 20.0f, 20.0f });
        layer->neuronBiases().write({ -10.0f });
        net.inputLayer(2).add(std::move(layer));
        assertEquals(net.predict(t00), false);
        assertEquals(net.predict(t10), true);
        assertEquals(net.predict(t01), true);
        assertEquals(net.predict(t11), true);
    }
    
    {
        // AND gate
        Network net(device);
        std::unique_ptr<Layer> layer(new Layer(device, 1, 2, Layer::Sigmoid));
        layer->neuronWeights().write({ 10.0f, 10.0f });
        layer->neuronBiases().write({ -10.0f });
        net.inputLayer(2).add(std::move(layer));
        assertEquals(net.predict(t00), false);
        assertEquals(net.predict(t10), false);
        assertEquals(net.predict(t01), false);
        assertEquals(net.predict(t11), true);
    }
}

void testBackprop(Device &device) {
    Network net(device);
    Layer lastLayer(device, 2, 3, Layer::Sigmoid);
    lastLayer.neuronWeights().write({ 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f });
    lastLayer.errorTerm().write({ 2.5f, 0.75f });
    Layer layer(device, 3, 1, Layer::Sigmoid);
    layer.derivative().write({ 1.0f, 1.0f, 1.0f });
    layer.computeErrorTerm(net.context(), lastLayer);
    assertEquals(layer.errorTerm(), { 5.75f, 9.0f, 12.25f });
}

void testTrainer(Device &device) {
    // Training set
    Matrix inputs(device, 4, 2, { 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f });
    Matrix outputs(device, 4, 1, { 0.0f, 1.0f, 1.0f, 0.0f });
    
    SimpleDataset data(inputs, outputs);
    
    // Test set
    Vector t00(device, {0.0f, 0.0f});
    Vector t10(device, {1.0f, 0.0f});
    Vector t01(device, {0.0f, 1.0f});
    Vector t11(device, {1.0f, 1.0f});
    
    {
        // 2 - 3 Sigmoid - 1 Sigmoid XOR NN
        Network net(device);
        net.inputLayer(2).add(std::unique_ptr<Layer>(new Layer(device, 3, 2, Layer::Sigmoid))).add(std::unique_ptr<Layer>(new Layer(device, 1, 3, Layer::Sigmoid)));
        net.init(/* seed= */12);

        GradientDescent opt(device, 3.0);
        MSECriterion criterion;
        Trainer trainer(net, criterion, data);
        float previousError = 0.0f;
        trainer.afterIteration = [&] (size_t i, size_t batch, float error) {
            if (i != 0) {
                // Make sure the error is going down
                assert(previousError >= error);
            }
            previousError = error;
        };
        trainer.gradientDescent(opt, 300);
        
        assertEquals(net.predict(t00), false);
        assertEquals(net.predict(t10), true);
        assertEquals(net.predict(t01), true);
        assertEquals(net.predict(t00), false);
    }
    
    {
        // 2 - 5 ReLU - 1 Sigmoid XOR NN
        Network net(device);
        net.inputLayer(2).add(std::unique_ptr<Layer>(new Layer(device, 5, 2, Layer::RectifiedLinearUnit))).add(std::unique_ptr<Layer>(new Layer(device, 1, 5, Layer::Sigmoid)));
        net.init(/* seed= */12);
        
        GradientDescent opt(device, 0.3);
        CrossEntropyCriterion criterion;
        Trainer trainer(net, criterion, data);
        trainer.gradientDescent(opt, 300);
        
        assertEquals(net.predict(t00), false);
        assertEquals(net.predict(t10), true);
        assertEquals(net.predict(t01), true);
        assertEquals(net.predict(t00), false);
    }
}

void testMNIST(Device &device) {
    std::cout << "Loading MNIST dataset...\n";
    
    MNIST trainingSet(device);
    if (trainingSet.load("train-images.idx3-ubyte", "train-labels.idx1-ubyte"))
        return;
    
    MNIST testSet(device);
    if (testSet.load("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte"))
        return;
    
    std::cout << "Finished loading MNIST dataset\n";
    
    Network net(device);
    size_t imageSize = trainingSet.imageWidth()*trainingSet.imageHeight();
    net.inputLayer(imageSize);
    net.add(std::unique_ptr<Layer>(new Layer(device, 400, imageSize, Layer::RectifiedLinearUnit)));
    net.add(std::unique_ptr<Layer>(new Layer(device, 10, 400, Layer::Sigmoid)));
    uint32_t seed = 12;
    std::cout << "Random initialization using seed '" << seed << "'\n";
    net.init(seed);
    
    GradientDescent opt(device, 0.3);
    CrossEntropyCriterion criterion;
    Trainer trainer(net, criterion, trainingSet);
    ClassificationEvaluator evaluator(testSet);
    trainer.reshuffleIndices = true;
    // Train & evaluate
    trainer.afterIteration = [&] (size_t i, size_t batch, float cost) {
        if (batch == 0 && i != 0) {
            std::cout << "Evaluating perfomance after " << i << " iteration(s):\n";
            auto result = evaluator.evaluate(net);
            std::cout << "Cost (of last batch) " << cost << ", test set accuracy: " << result.correctPredictions << "/" << result.count << ", " << result.percentageOfCorrectPredictions() << "%\n";
        }
    };
    trainer.miniBatchGradientDescent(opt, 30, 50);
}

int main(int argc, const char * argv[]) {
    auto device = selectDevice();
    device.init();
    std::cout << "Using device '" << device.name() << "' by '" << device.vendor() << "', version '" << device.version() << "'\n";
    CommandQueue queue(device);
    device.queue(queue);

    testVectors(device);
    testSum(device);
    testBooleanOperations(device);
    testLayers(device);
    testLogicGates(device);
    testBackprop(device);
    testTrainer(device);
    testMNIST(device);
    
    return 0;
}
