#include <stdio.h>
#include "mnistReader.h"

mnistData *readMnistAll(mnistType type)
{
    uint32_t count = type == MNIST_TRAINING ? MNIST_TRAINING_COUNT : MNIST_TEST_COUNT;
    mnistData *data = readMnistN(count, type);
    return data;
}

mnistData *readMnistN(uint32_t count, mnistType type)
{
    mnistData *data = malloc(sizeof(mnistData));
    data->imagesCount = count;
    data->images = malloc(MNIST_IMAGE_SIZE * count);
    data->labels = malloc(count);

    char imagesPath[100] = type == MNIST_TRAINING ? MNIST_TRAINING_IMAGES_FILE : MNIST_TEST_IMAGES_FILE;
    char labelsPath[100] = type == MNIST_TRAINING ? MNIST_TRAINING_LABELS_FILE : MNIST_TEST_LABELS_FILE;

    FILE *imagesFile = fopen(imagesPath, "rb");
    FILE *labelsFile = fopen(labelsPath, "rb");
    fseek(imagesFile, MNIST_OFFSET_IMAGES, SEEK_SET);
    fseek(labelsFile, MNIST_OFFSET_LABELS, SEEK_SET);
    fread(data->images, MNIST_IMAGE_SIZE, count, imagesFile);
    fread(data->labels, 1, count, labelsFile);
    fclose(imagesFile);
    fclose(labelsFile);

    return data;
}