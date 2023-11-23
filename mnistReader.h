#ifndef MNIST_READER_H
#define MNIST_READER_H

#include <stdint.h>

#define MNIST_OFFSET_IMAGES 16
#define MNIST_OFFSET_LABELS 8
#define MNIST_TRAINING_COUNT 60000
#define MNIST_TEST_COUNT 10000
#define MNIST_IMAGE_SIZE 784
#define MNIST_TRAINING_IMAGES_FILE "train-images.idx3-ubyte.bin"
#define MNIST_TRAINING_LABELS_FILE "train-labels.idx1-ubyte.bin"
#define MNIST_TEST_IMAGES_FILE "t10k-images.idx3-ubyte.bin"
#define MNIST_TEST_LABELS_FILE "t10k-labels.idx1-ubyte.bin"

typedef struct
{
    float pixels[MNIST_IMAGE_SIZE];
} mnistImage;

typedef enum
{
    MNIST_TRAINING,
    MNIST_TEST
} mnistType;

typedef struct 
{
    uint32_t imagesCount;
    mnistImage *images;
    uint8_t *labels;
} mnistData;


mnistData *readMnistN(uint32_t count, mnistType type);
mnistData *readMnistAll(mnistType type);

#endif // MNIST_READER_H