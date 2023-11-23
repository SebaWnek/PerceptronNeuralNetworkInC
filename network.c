#include <stdio.h>
#include <stdlib.h>
#include "network.h"
#include "layer.h"

void calculateBackPropagation(float *desiredOutput, int outputsCount);

network *currentNetwork;
bool initialized = false;

bool createNetwork( int inputsCount, int outputsCount, int layersCount, int *layersSizes, functionType *layersFunctions)
{
    if(initialized)
    {
        printf("Error: network already initialized\n");
        return false;
    }
    currentNetwork = malloc(sizeof(network));
    currentNetwork->layersCount = layersCount;
    currentNetwork->layers = malloc(sizeof(layer*) * layersCount);
    currentNetwork->inputs = malloc(sizeof(float) * inputsCount);
    for(int i = 0; i-1 < layersCount; i++)
    {
        currentNetwork->layers[i] = createLayer(layersSizes[i], i == 0 ? inputsCount : layersSizes[i - 1], layersFunctions[i]);
        initializeLayer(currentNetwork->layers[i]);
    }
    currentNetwork->layers[layersCount - 1] = createLayer(outputsCount, layersSizes[layersCount - 2], layersFunctions[layersCount - 1]);
    initialized = true;
    return true;
}

bool createNetworkWithMultipliers( int inputsCount, int outputsCount, int layersCount, int *layersSizes, functionType *layersFunctions, float biasMultiplier, float weightsMultiplier, float learningRate, float randRange)
{
    bool done = createNetwork(inputsCount, outputsCount, layersCount, layersSizes, layersFunctions);
    if(!done)
    {
        return false;
    }
    for(int i = 0; i < layersCount; i++)
    {
        updateMultipliers(currentNetwork->layers[i], biasMultiplier, weightsMultiplier, learningRate, randRange);
    }
    return true;
}

int getOutputs(float *outputs)
{
    outputs = currentNetwork->layers[currentNetwork->layersCount - 1]->outputs;
    return currentNetwork->layers[currentNetwork->layersCount - 1]->valuesCount;
}

bool calculateNetwork(float *inputs, int inputsCount)
{
    if(initialized == false)
    {
        printf("Error: network not initialized\n");
        return false;
    }
    calculateOutputsBase(currentNetwork->layers[0], inputs, inputsCount);
    if(currentNetwork->layersCount > 1)
    {
        for(int i = 1; i < currentNetwork->layersCount; i++)
        {
            calculateOutputs(currentNetwork->layers[i], currentNetwork->layers[i - 1]);
        }
    }
    return true;
}

bool trainNetwork(float *inputs, int inputsCount, float *expected, int expectedCount)
{
    if(initialized == false)
    {
        printf("Error: network not initialized\n");
        return false;
    }
    calculateNetwork(inputs, inputsCount);
    calculateBackPropagation(expected, expectedCount);
    return true;
}

void calculateBackPropagation(float *desiredOutput, int outputsCount)
{
    if(currentNetwork->layersCount == 1)
    {
        calculateDeltasBase(currentNetwork->layers[0], currentNetwork->inputs, currentNetwork->inputsCount, desiredOutput, outputsCount, true);
    }
    else
    {
        for(int i = currentNetwork->layersCount -1; i >= 0; i++)
        {
            if(i == currentNetwork->layersCount - 1)
            {
                calculateDeltasLast(currentNetwork->layers[i], currentNetwork->layers[i - 1], desiredOutput, outputsCount);
            }
            else if(i == 0)
            {
                calculateDeltasFirst(currentNetwork->layers[i], currentNetwork->inputs, currentNetwork->inputsCount, currentNetwork->layers[i + 1]);
            }
            else
            {
                calculateDeltasMiddle(currentNetwork->layers[i], currentNetwork->layers[i - 1], currentNetwork->layers[i + 1]);
            }
        }
    }
    for(int i = currentNetwork->layersCount -1; i >= 0; i++)
    {
        updateLayer(currentNetwork->layers[i]);
    }
}