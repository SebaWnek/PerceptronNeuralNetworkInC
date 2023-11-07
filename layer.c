#include <stdlib.h>
#include <stdbool.h>
#include "layer.h"

layer* createLayer(int count, int prevCount, functionType type)
{
    layer *newLayer = malloc(sizeof(layer));
    newLayer->function = getActivationFunction(type, false);
    newLayer->derivative = getActivationFunction(type, true);
    newLayer->valuesCount = count;
    newLayer->previosValuesCount = prevCount;
    newLayer->weights = malloc(sizeof(float) * count * prevCount);
    newLayer->dweights = malloc(sizeof(float) * count * prevCount);
    newLayer->biases = malloc(sizeof(float) * count);
    newLayer->dbiases = malloc(sizeof(float) * count);
    newLayer->outputs = malloc(sizeof(float) * count);
    newLayer->gammas = malloc(sizeof(float) * count);
    newLayer->derivatives = malloc(sizeof(float) * count);
    return newLayer;
}

float getRandom()
{
    float randomValue = (float)rand() / (float)RAND_MAX;
    return randomValue * (MAX_RAND - MIN_RAND) + MIN_RAND;
}

void initializeLayer(layer *layer)
{
    for(int i = 0; i < layer->valuesCount; i++)
    {
        layer->biases[i] = getRandom() * biasMultiplier;
        layer->dbiases[i] = 0;
        layer->gammas[i] = 1;
        for(int j = 0; j < layer->previosValuesCount; j++)
        {
            layer->weights[i * layer->previosValuesCount + j] = getRandom() * weightsMultiplier;
            layer->dweights[i * layer->previosValuesCount + j] = 0;
        }
    }
}

void calculateOutputs(layer *currentLayer, layer *previousLayer)
{
    calculateOutputsBase(currentLayer, previousLayer->outputs, previousLayer->valuesCount);
}

void calculateOutputsBase(layer *currentLayer, float *inputs, int inputsCount)
{
    float sum = 0;
    for(int i = 0; i < currentLayer->valuesCount; i++)
    {
        sum = 0;
        for(int j = 0; j < inputsCount; j++)
        {
            sum += inputs[j] * currentLayer->weights[i * inputsCount + j];
        }
        sum += currentLayer->biases[i];
        currentLayer->outputs[i] = currentLayer->function(sum);
    }
}

void calculateDeltasBase(layer *currentLayer, float *previousValues, int previousValuesCount, float *nextValues, int nextValuesCount, bool outputs)
{
    for(int i = 0; i < currentLayer->valuesCount; i++)
    {
        currentLayer->derivatives[i] = currentLayer->derivative(currentLayer->outputs[i]);
    }
    if(outputs)
    {
        for(int i = 0; i < currentLayer->valuesCount; i++)
        {
            currentLayer->gammas[i] = (nextValues[i] - currentLayer->outputs[i]) * currentLayer->derivatives[i];
        }
    }
    else
    {
        memset(currentLayer->gammas, 0, sizeof(float) * currentLayer->valuesCount);
        for(int i = 0; i < currentLayer->valuesCount; i++)
        {
            currentLayer->gammas[i] = 0;
            for(int j = 0; j < nextValuesCount; j++)
            {
                currentLayer->gammas[i] += nextValues[j] * currentLayer->weights[j * currentLayer->valuesCount + i];
            }
            currentLayer->gammas[i] *= currentLayer->derivatives[i];
        }
    }
}
void calculateDeltasMiddle(layer *currentLayer, layer *previousLayer, layer *nextLayer)
{
    calculateDeltasBase(currentLayer, previousLayer->outputs, previousLayer->valuesCount, nextLayer->gammas, nextLayer->valuesCount, false);
}
void calculateDeltasLast(layer *currentLayer, layer *previousLayer, float *nextValues, int nextValuesCount)
{
    calculateDeltasBase(currentLayer, previousLayer->outputs, previousLayer->valuesCount, nextValues, nextValuesCount, true);
}
void calculateDeltasFirst(layer *currentLayer, float *previousValues, int previousValuesCount, layer *nextLayer)
{
    calculateDeltasBase(currentLayer, previousValues, previousValuesCount, nextLayer->gammas, nextLayer->valuesCount, false);
}
