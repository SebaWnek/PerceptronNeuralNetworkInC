#ifndef LAYER_H
#define LAYER_H

#include "functions.h"

typedef struct
{
    activationFunction function;
    functionType type;
    activationFunction derivative;
    int valuesCount;
    int previosValuesCount;
    float *weights;
    float *dweights;
    float *biases;
    float *dbiases;
    float *outputs;
    float *gammas;
    float *derivatives;
    float learningRate;
    float biasMultiplier;
    float weightsMultiplier;
    float randRange;
} layer;

layer* createLayer(int count, int prevCount, functionType type);
void initializeLayer(layer *layer);

void calculateOutputs(layer *currentLayer, layer *previousLayer);
void calculateOutputsBase(layer *currentLayer, float *inputs, int inputsCount);

void calculateDeltasBase(layer *currentLayer, float *previousValues, int previousValuesCount, float *nextValues, int nextValuesCount, bool outputs);
void calculateDeltasMiddle(layer *currentLayer, layer *previousLayer, layer *nextLayer);
void calculateDeltasLast(layer *currentLayer, layer *previousLayer, float *nextValues, int nextValuesCount);
void calculateDeltasFirst(layer *currentLayer, float *previousValues, int previousValuesCount, layer *nextLayer);

void updateLayer(layer *layer);

void updateMultipliers(layer *layer, float biasMultiplier, float weightsMultiplier, float learningRate, float randRange);

#endif // LAYER_H
