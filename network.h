#ifndef NETWORK_H
#define NETWORK_H

void createNetwork( int inputsCount, int layersCount, int *layersSizes, functionType *layersFunctions);
int getOutputs(float *outputs);
void calculateNetwork(float *inputs, int inputsCount);
void trainNetwork(float *inputs, int inputsCount, float *expected, int expectedCount);

typedef struct
{
    float *inputs;
    int inputsCount;
    layer **layers;
    int layersCount;
} network;

#endif // NETWORK_H


