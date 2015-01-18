#include <iostream>
#include "mnistDataset.h"

MNIST::MNIST(Device &device) : device(device), images_(device), labelProbabilities_(device), labels(device,ValueType::Uint16) {
}

size_t MNIST::size() const {
    return images_.rows();
}

size_t MNIST::inputSize() const {
    return width*height;
}

size_t MNIST::outputSize() const {
    return 10;
}

void MNIST::get(size_t i, Vector &input, Vector &output) {
    images_.row(i).copy(input);
    labelProbabilities_.row(i).copy(output);
}

const Vector *MNIST::classificationLabels() {
    if (!labels.size()) {
        // Lazily upload the labels
        labels.resize(labelValues.size());
        std::vector<uint16_t> values(labelValues.size());
        for (size_t i = 0; i < values.size(); ++i)
            values[i] = labelValues[i];
        labels.write(values);
    }
    return &labels;
}

struct LabelFileHeader {
    uint32_t magic;
    uint32_t numberOfLabels;
};

struct ImageFileHeader {
    uint32_t magic;
    uint32_t numberOfImages;
    uint32_t width;
    uint32_t height;
};

static uint32_t swap_uint32( uint32_t val ) {
    val = ((val << 8) & 0xFF00FF00 ) | ((val >> 8) & 0xFF00FF );
    return (val << 16) | (val >> 16);
}

bool MNIST::load(const char *imageFilename, const char *labelFilename) {
    FILE *labelsFile = fopen(labelFilename, "rb");
    if (!labelsFile) {
        std::cerr << "Error reading '" << labelFilename << "'\n";
        return true;
    }
    FILE *imagesFile = fopen(imageFilename, "rb");
    if (!imagesFile) {
        std::cerr << "Error reading '" << imageFilename << "'\n";
        return true;
    }
    LabelFileHeader labelsHeader;
    fread(&labelsHeader, sizeof(labelsHeader), 1, labelsFile);
    labelsHeader.magic = swap_uint32(labelsHeader.magic);
    labelsHeader.numberOfLabels = swap_uint32(labelsHeader.numberOfLabels);
    labelValues.resize(labelsHeader.numberOfLabels);
    fread(labelValues.data(), sizeof(uint8_t), labelValues.size(), labelsFile);
    fclose(labelsFile);
    
    ImageFileHeader imagesHeader;
    fread(&imagesHeader, sizeof(imagesHeader), 1, imagesFile);
    imagesHeader.magic = swap_uint32(imagesHeader.magic);
    imagesHeader.numberOfImages = swap_uint32(imagesHeader.numberOfImages);
    width = imagesHeader.width = swap_uint32(imagesHeader.width);
    height = imagesHeader.height = swap_uint32(imagesHeader.height);
    std::vector<uint8_t> pixels;
    pixels.resize(imagesHeader.width*imagesHeader.height*imagesHeader.numberOfImages);
    fread(pixels.data(), sizeof(uint8_t), pixels.size(), imagesFile);
    fclose(imagesFile);
    
    // Upload to device
    std::vector<float> fpixels(pixels.size());
    for (size_t i = 0; i < pixels.size(); ++i) {
        fpixels[i] = float(pixels[i])/255.0f;
    }
    size_t imageSize = imagesHeader.width * imagesHeader.height;
    images_.resize(imagesHeader.numberOfImages, imageSize);
    images_.write(fpixels);
    
    std::vector<float> probabilities(labelValues.size()*10, 0.0f);
    for (size_t i = 0; i < labelValues.size(); ++i) {
        probabilities[i*10 + labelValues[i]] = 1.0f;
    }
    labelProbabilities_.resize(labelValues.size(), 10);
    labelProbabilities_.write(probabilities);
    return false;
}
