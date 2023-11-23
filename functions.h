#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <stdbool.h>

typedef float (*activationFunction)(float);

typedef enum {
    TANH,
    SIGMOID,
    RELU,
    LEAKYRELU,
    LINEAR,
    functionTypesCount
} functionType;

activationFunction getActivationFunction(functionType type, bool derivative);

#endif // FUNCTIONS_H
