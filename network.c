#include "network.h"
#include "layer.h"

void calculateBackPropagation(float *desiredOutput, int outputsCount);

network *currentNetwork;

void createNetwork( int inputsCount, int layersCount, int *layersSizes, functionType *layersFunctions)
{
    currentNetwork = malloc(sizeof(network));
    currentNetwork->layersCount = layersCount;
    currentNetwork->layers = malloc(sizeof(layer*) * layersCount);
    currentNetwork->inputs = malloc(sizeof(float) * inputsCount);
    for(int i = 0; i < layersCount; i++)
    {
        currentNetwork->layers[i] = createLayer(layersSizes[i], i == 0 ? inputsCount : layersSizes[i - 1], layersFunctions[i]);
        initializeLayer(currentNetwork->layers[i]);
    }
}

void createNetworkWithMultipliers( int inputsCount, int layersCount, int *layersSizes, functionType *layersFunctions, float biasMultiplier, float weightsMultiplier, float learningRate, float randRange)
{
    createNetwork(inputsCount, layersCount, layersSizes, layersFunctions);
    for(int i = 0; i < layersCount; i++)
    {
        updateMultipliers(currentNetwork->layers[i], biasMultiplier, weightsMultiplier, learningRate, randRange);
    }
}

int getOutputs(float *outputs)
{
    outputs = currentNetwork->layers[currentNetwork->layersCount - 1]->outputs;
    return currentNetwork->layers[currentNetwork->layersCount - 1]->valuesCount;
}

void calculateNetwork(float *inputs, int inputsCount)
{
    calculateFirstOutputs(currentNetwork->layers[0], inputs, inputsCount);
    if(currentNetwork->layersCount > 1)
    {
        for(int i = 1; i < currentNetwork->layersCount; i++)
        {
            calculateOutputs(currentNetwork->layers[i], currentNetwork->layers[i - 1]);
        }
    }
}
void trainNetwork(float *inputs, int inputsCount, float *expected, int expectedCount)
{
    calculateNetwork(inputs, inputsCount);
    calculateBackPropagation(expected, expectedCount);
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