#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "commands.h"
#include "mnistReader.h"
#include "network.h"

#define TEXT_BUFFER_SIZE 30
#define ARG_BUFFER_SIZE 2

bool readInput(command *cmd, uint16_t *arg);
bool getCommand(command *cmdOut, char *buffer);

void assignCommands();

void showHelp(uint16_t *empty);
void showLoadedCount(uint16_t *empty);
void loadNImages(uint16_t *arg);
void loadAllImages(uint16_t *arg);
void createNewNetwork(uint16_t *arg);
void updateNetwork(uint16_t *arg);
void trainNewNetwork(uint16_t *arg);
void testNetwork(uint16_t *arg);
void displayImage(uint16_t *arg);

mnistData *trainingData;
mnistData *testData;
bool networkInitialized = false;

int main() 
{
    uint16_t arg[ARG_BUFFER_SIZE];
    command cmd;
    
    assignCommands();
    printf("Welcome to neural network test program\n\n");
    showHelp(NULL); 
    
    while (true)
    {
        printf("Enter command:\n");
        while(!readInput(&cmd, arg));
#ifdef DEBUG
        printf("Command addr in main: %p\n", &(cmd.name));
        printf("Command: %s\n", cmd.name);
        printf("Arguments: ");
        for(uint8_t i=0; i < cmd.argCount; i++)
        {
            printf("%d ", arg[i]);
        }
        printf("\n");
#endif
        cmd.function(arg);
    }
    return 0;
}

bool readInput(command *cmd, uint16_t *arg)
{
    char buffer[TEXT_BUFFER_SIZE];
    char c;
    char *token;
    uint8_t i=0;

    while((c = getchar()) != EOF && c != '\n')
    {
        buffer[i] = c;
        if(++i == TEXT_BUFFER_SIZE)
        {
            printf("Error: input too long\n");
            return false;
        }
    }
#ifdef DEBUG
    printf("Buffer: %s\n", buffer);
#endif
    token = strtok(buffer, " ");
#ifdef DEBUG
    printf("Token: %s\n", token);
    printf("Command addr in read: %p\n", cmd->name);
#endif
    if(!getCommand(cmd, token))
    {
        printf("Error: invalid command\n");
        return false;
    }
    for(uint8_t i; i < cmd->argCount; i++)
    {
        token = strtok(NULL, " ");
#ifdef DEBUG
        printf("Token: %s\n", token);
#endif
        if(token == NULL)
        {
            printf("Error: too few arguments\n");
            return false;
        }
        arg[i] = atoi(token);
    }
    return true;
}

//buffer input, cmd output
bool getCommand(command *cmdOut, char *bufferIn)
{
#ifdef DEBUG
    printf("Buffer: %s\n", bufferIn);
#endif
    for(uint8_t i=0; i < commandCount; i++)
    {
        if(strcmp(bufferIn, commands[i].name) == 0)
        {
            memccpy(cmdOut, &commands[i], sizeof(command), sizeof(command));
#ifdef DEBUG
            printf("Command: %s\n", cmdOut->name);
            printf("Command addr in get: %p\n", cmdOut->name);
#endif
            return true;
        }
    }
    return false;
}

void assignCommands()
{
    commands[0].function = createNewNetwork;    //create
    commands[1].function = updateNetwork;    //update
    commands[2].function = loadNImages;    //loadn
    commands[3].function = loadAllImages;    //loadall
    commands[4].function = showLoadedCount;    //showloaded
    commands[5].function = trainNewNetwork;    //train
    commands[6].function = testNetwork;    //test
    commands[7].function = showHelp;    //help
    commands[8].function = displayImage;    //display
}

void showHelp(uint16_t *empty)
{
    for(uint8_t i=0; i < commandCount; i++)
    {
        printf("command: %s\n", commands[i].name);
        printf("description: %s\n\n", commands[i].description);
    }
}

void showLoadedCount(uint16_t *empty)
{
    printf("Loaded images count:\ntraining set - %d\ntest set - %d\n", trainingData->imagesCount, testData->imagesCount);
}

void loadNImages(uint16_t *args)
{
    uint32_t count = args[0];
    mnistType type = (mnistType)args[1];
    if(type == MNIST_TRAINING)
    {
        if(trainingData != NULL)
        {
            printf("Error: training set already loaded\n");
            return;
        }
        trainingData = readMnistN(count, type);
    }
    else
    {
        if(testData != NULL)
        {
            printf("Error: test set already loaded\n");
            return;
        }
        testData = readMnistN(count, type);
    }
}

void loadAllImages(uint16_t *arg)
{
    mnistType type = (mnistType)arg[0];
    if(type == MNIST_TRAINING)
    {
        if(trainingData != NULL)
        {
            printf("Error: training set already loaded\n");
            return;
        }
        trainingData = readMnistAll(type);
    }
    else
    {
        if(testData != NULL)
        {
            printf("Error: test set already loaded\n");
            return;
        }
        testData = readMnistAll(type);
    }
}

void createNewNetwork(uint16_t *arg)
{
    if(networkInitialized)
    {
        printf("Error: network already initialized\n");
        return;
    }
    // createNetwork( int inputsCount, int outputsCount, int layersCount, int *layersSizes, functionType *layersFunctions)
    int inputsCount = MNIST_IMAGE_SIZE;
    int outputsCount = 10; // Obviously 10 possible values :D 
    int layersCount = 3;
    int layersSizes[] = {32, 32, 32};
    functionType layersFunctions[] = {SIGMOID, SIGMOID, SIGMOID};
    float biasMultiplier = 1;
    float weightsMultiplier = 1;
    float learningRate = 0.00005;
    float randRange = 1;

    printf("Creating network\n");

    printf("Use default values? (y/n)\n");
    char c;
    while((c = getchar()) != 'y' && c != 'n')
    {
        printf("Error: invalid input\n");
    }
    while((c = getchar()) != '\n' && c != EOF); // remove rest of input
#ifdef DEBUG
    printf("c: %c\n", c);
    printf("c: %d\n", c);
#endif
    if(c == 'y')
    {
        createNetworkWithMultipliers(inputsCount, outputsCount, layersCount, layersSizes, layersFunctions, biasMultiplier, weightsMultiplier, learningRate, randRange);
    }
    // createNetworkWithMultipliers( int inputsCount, int outputsCount, int layersCount, int *layersSizes, functionType *layersFunctions, float biasMultiplier, float weightsMultiplier, float learningRate, float randRange)
    else
    {
        while(true)
        {
            printf("Enter layers count:\n");
            scanf("%d", &layersCount);
            while((c = getchar()) != '\n' && c != EOF); // remove rest of input
            if(layersCount < 2)
            {
                printf("Error: layers count must be at least 2\n");
                continue;
            }
            break;
        }

        for(int i = 0; i < layersCount; i++)
        {
            // read layers sizes
            while(true)
            {
                printf("Enter size of layer %d:\n", i);
                scanf("%d", &layersSizes[i]);
                while((c = getchar()) != '\n' && c != EOF); // remove rest of input
                if(layersSizes[i] < 1)
                {
                    printf("Error: layer size must be at least 1\n");
                    continue;
                }
                break;
            }
            /*  
                Read function of layer
                TANH,
                SIGMOID,
                RELU,
                LEAKYRELU,
                LINEAR
            */
            while(true)
            {
                printf("Enter function of layer %d:\n", i);
                printf("0 - TANH, 1 - SIGMOID, 2 - RELU, 3 - LEAKYRELU, 4 - LINEAR\n");
                scanf("%d", &layersFunctions[i]);
                while((c = getchar()) != '\n' && c != EOF); // remove rest of input
                if(layersFunctions[i] < 0 || layersFunctions[i] >= functionTypesCount)
                {
                    printf("Error: invalid function\n");
                    continue;
                }
                break;
            }
        }
        // read other parameters
        while(true)
        {
            printf("Enter bias multiplier:\n");
            scanf("%f", &biasMultiplier);
            while((c = getchar()) != '\n' && c != EOF); // remove rest of input
            if(biasMultiplier < 0)
            {
                printf("Error: bias multiplier must be at least 0\n");
                continue;
            }
            break;
        }

        while(true)
        {
            printf("Enter weights multiplier:\n");
            scanf("%f", &weightsMultiplier);
            while((c = getchar()) != '\n' && c != EOF); // remove rest of input
            if(weightsMultiplier < 0)
            {
                printf("Error: weights multiplier must be at least 0\n");
                continue;
            }
            break;
        }

        while(true)
        {
            printf("Enter learning rate:\n");
            scanf("%f", &learningRate);
            while((c = getchar()) != '\n' && c != EOF); // remove rest of input
            if(learningRate < 0)
            {
                printf("Error: learning rate must be at least 0\n");
                continue;
            }
            break;
        }

        while(true)
        {
            printf("Enter random range:\n");
            scanf("%f", &randRange);
            while((c = getchar()) != '\n' && c != EOF); // remove rest of input
            if(randRange < 0)
            {
                printf("Error: random range must be at least 0\n");
                continue;
            }
            break;
        }
        createNetworkWithMultipliers(inputsCount, outputsCount, layersCount, layersSizes, layersFunctions, biasMultiplier, weightsMultiplier, learningRate, randRange);
    }
    networkInitialized = true;
}

void updateNetwork(uint16_t *arg)
{
    printf("TBD\n");
}

void trainNewNetwork(uint16_t *arg)
{
    uint16_t epochs = arg[0];
    uint16_t imagesCount = trainingData->imagesCount;
    float expected[10];
    for(uint16_t i = 0; i < epochs; i++)
    {
        for(uint16_t j = 0; j < imagesCount; j++)
        {
            memset(expected, 0, sizeof(float) * 10);
            expected[trainingData->labels[j]] = 1;
            trainNetwork(trainingData->images[j].pixels, MNIST_IMAGE_SIZE, expected, 10);
        }
    }
}

void testNetwork(uint16_t *arg)
{
    float *results = malloc(sizeof(float) * 10);
    uint8_t *labels = malloc(sizeof(uint8_t) * testData->imagesCount);
    float *tmp;
    float tmpMax;
    int count = 10;

    // calculate results
    for(int i = 0; i < trainingData->imagesCount; i++)
    {
        tmpMax = 0;
        calculateNetwork(trainingData->images[i].pixels, MNIST_IMAGE_SIZE);
        getOutputs(tmp); // I ignore return value of 10 as obviously we only have 10 numbers in decimal system... 
        memcpy(results + count * i, tmp, count * sizeof(float)); 
    }
    // find biggest values
    for(int i = 0; i < count * trainingData->imagesCount; i+=10)
    {
        for(int j = 0; j < count; j++)
        {
            if(results[i + j] > tmpMax)
            {
                tmpMax = results[i + j];
                labels[i / 10] = j;
            }
        }
    }
    // compare with labels
    for(int i = 0; i < trainingData->imagesCount; i++)
    {
        if(labels[i] == trainingData->labels[i])
        {
            count++;
        }
    }
    printf("Correctly recognized %d of %d images\n", count, trainingData->imagesCount);
}

void displayImage(uint16_t *arg)
{
    int index = arg[0];
    mnistType type = (mnistType)arg[1];
    mnistData *data = type == MNIST_TRAINING ? trainingData : testData;
    if(data == NULL)
    {
        printf("Error: data not loaded\n");
        return;
    }
    if(index < 0 || index >= data->imagesCount)
    {
        printf("Error: invalid index\n");
        return;
    }
    printf("Image %d:\n", index);
    printf("Label: %d\n", data->labels[index]);
    for(int i = 0; i < MNIST_IMAGE_SIZE; i++)
    {
        if(i % 28 == 0)
        {
            printf("\n");
        }
        printf("%.2f", data->images[index].pixels[i]);
    }
    printf("\n");
}

