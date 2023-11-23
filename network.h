#ifndef NETWORK_H
#define NETWORK_H

#include <stdbool.h>
#include "functions.h"
#include "layer.h"

bool createNetwork(int inputsCount, int outputsCount, int layersCount, int *layersSizes, functionType *layersFunctions);
bool createNetworkWithMultipliers( int inputsCount, int outputsCount, int layersCount, int *layersSizes, functionType *layersFunctions, float biasMultiplier, float weightsMultiplier, float learningRate, float randRange);
int getOutputs(float *outputs);
bool calculateNetwork(float *inputs, int inputsCount);
bool trainNetwork(float *inputs, int inputsCount, float *expected, int expectedCount);

typedef struct
{
    float *inputs;
    int inputsCount;
    layer **layers;
    int layersCount;
} network;

#endif // NETWORK_H


