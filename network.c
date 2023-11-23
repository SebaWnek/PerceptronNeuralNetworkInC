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
    currentNetwork->inputsCount = inputsCount;
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
        for(int i = currentNetwork->layersCount -1; i >= 0; i--)
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
    for(int i = currentNetwork->layersCount -1; i >= 0; i--)
    {
        updateLayer(currentNetwork->layers[i]);
    }
}

void printNetworkInfo(bool detailed)
{
    printf("Network info:\n");
    printf("Inputs count: %d\n", currentNetwork->inputsCount);
    printf("Layers count: %d\n", currentNetwork->layersCount);
    printf("Layers info:\n");
    for(int i = 0; i < currentNetwork->layersCount; i++)
    {
        printf("Layer %d:\n", i);
        printf("Values count: %d\n", currentNetwork->layers[i]->valuesCount);
        printf("Function: %d (0 - TanH, 1 - Sigmoid, 2 - ReLu, 3 - LeakyReLu, 4 - Linear)\n", currentNetwork->layers[i]->type);
        printf("Function pointer: %p\n", currentNetwork->layers[i]->function);
        printf("Bias multiplier: %f\n", currentNetwork->layers[i]->biasMultiplier);
        printf("Weights multiplier: %f\n", currentNetwork->layers[i]->weightsMultiplier);
        printf("Learning rate: %f\n", currentNetwork->layers[i]->learningRate);
        printf("Random range: %f\n", currentNetwork->layers[i]->randRange);
        if(detailed) printLayerValues(&currentNetwork->layers[i]);
    }
}

void printLayerValues(layer *layer)
{
    printf("Weights:\n");
    for(int i = 0; i < layer->valuesCount; i++)
    {
        for(int j = 0; j < layer->previosValuesCount; j++)
        {
            printf("%f ", layer->weights[i * layer->previosValuesCount + j]);
        }
        printf("\n");
    }
    printf("Biases:\n");
    for(int i = 0; i < layer->valuesCount; i++)
    {
        printf("%f ", layer->biases[i]);
    }
    printf("\n");
}