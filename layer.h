#ifndef LAYER_H
#define LAYER_H

#include "functions.h"

//Allowing to change random numbers generator behavoiur
#define MIN_RAND = -1f
#define MAX_RAND = 1f

float learningRate = 0.00005;
float biasMultiplier = 10;
float weightsMultiplier = 2;

typedef struct
{
    activationFunction function;
    activationFunction derivative;
    int valuesCount;
    int previosValuesCount;
    float *weights;
    float *dweights;
    float *biases;
    float *dbiases;
    float *outputs;
    float *gammas;
} layer;

layer* createLayer(int count, int prevCount, functionType type);
void initializeLayer(layer *layer);
void calculateOutputs(layer *currentLayer, layer *previousLayer);



#endif // LAYER_H
