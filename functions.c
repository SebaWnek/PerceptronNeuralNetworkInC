#include <stdbool.h>
#include <stddef.h>
#include <math.h>
#include "functions.h"

// Function declarations
float tanH(float input);
float dTanH(float input);
float sigmoid(float input);
float dSigmoid(float inputs);
float reLU(float input);
float dReLU(float input);
float leakyReLU(float input);
float dLeakyReLU(float input);
float linear(float input);
float dLinear(float input);

// Function definitions
activationFunction getActivationFunction(functionType type, bool derivative)
{
    if(!derivative)
    {
        switch (type)
        {
            case TANH:
                return tanH;
            case SIGMOID:
                return sigmoid;
            case RELU:
                return reLU;
            case LEAKYRELU:
                return leakyReLU;
            case LINEAR:
                return linear;
            default:
                return NULL;
        }
    }
    else
    {
        switch (type)
        {
            case TANH:
                return dTanH;
            case SIGMOID:
                return dSigmoid;
            case RELU:
                return dReLU;
            case LEAKYRELU:
                return dLeakyReLU;
            case LINEAR:
                return dLinear;
            default:
                return NULL;
        }
    }
}

float tanH(float input)
{
    return tanhf(input);
}

float dTanH(float input)
{
    float t = tanhf(input);
    return 1 - t * t;
}

float sigmoid(float input)
{
    return 1 / (1 + expf(-input));
}

float dSigmoid(float input)
{
    float s = 1 / (1 + expf(-input));
    return s * (1 - s);
}

float reLU(float input)
{
    return fmaxf(0, input);
}

float dReLU(float input)
{
    return input > 0 ? 1 : 0;
}

float leakyReLU(float input)
{
    return fmaxf(0.1 * input, input);
}

float dLeakyReLU(float input)
{
    return input > 0 ? 1 : 0.1;
}

float linear(float input)
{
    return input;
}

float dLinear(float input)
{
    return 1;
}



