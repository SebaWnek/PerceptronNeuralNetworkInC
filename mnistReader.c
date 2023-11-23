#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "mnistReader.h"

mnistData *readMnistAll(mnistType type)
{
    uint32_t count = type == MNIST_TRAINING ? MNIST_TRAINING_COUNT : MNIST_TEST_COUNT;
    mnistData *data = readMnistN(count, type);
    return data;
}

mnistData *readMnistN(uint32_t count, mnistType type)
{
    char imagesPath[100];
    char labelsPath[100];
    mnistImage *tmpImage = malloc(MNIST_IMAGE_SIZE * sizeof(float));
    uint8_t *tmpData = malloc(MNIST_IMAGE_SIZE * count);
    mnistData *data = malloc(sizeof(mnistData));
    data->imagesCount = count;
    data->images = malloc(MNIST_IMAGE_SIZE * count * sizeof(float));
    data->labels = malloc(count);

    if(type == MNIST_TRAINING)
    {
        strcpy(imagesPath, MNIST_TRAINING_IMAGES_FILE);
        strcpy(labelsPath, MNIST_TRAINING_LABELS_FILE);
    }
    else
    {
        strcpy(imagesPath, MNIST_TEST_IMAGES_FILE);
        strcpy(labelsPath, MNIST_TEST_LABELS_FILE);
    }

    FILE *imagesFile = fopen(imagesPath, "rb");
    FILE *labelsFile = fopen(labelsPath, "rb");
    if(imagesFile == NULL || labelsFile == NULL)
    {
        printf("Error opening files!\n");
        exit(1);
    }
    fseek(imagesFile, MNIST_OFFSET_IMAGES, SEEK_SET);
    fseek(labelsFile, MNIST_OFFSET_LABELS, SEEK_SET);
    fread(tmpData, MNIST_IMAGE_SIZE, count, imagesFile);
    fread(data->labels, 1, count, labelsFile);
    fclose(imagesFile);
    fclose(labelsFile);

    for(int i = 0; i < count * MNIST_IMAGE_SIZE; i++)
    {
        ((float*)(data->images))[i] = (float)tmpData[i] / 255.0f;
    }

    free(tmpData);

    return data;
}