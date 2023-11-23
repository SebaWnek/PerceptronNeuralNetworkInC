#ifndef COMMANDS_H
#define COMMANDS_H

#include <stdint.h>

#define CREATE_NETWORK_CMD "create"
#define UPDATE_NETWORK_CMD "update"
#define LOAD_N_IMAGES_CMD "loadn"
#define LOAD_ALL_IMAGES_CMD "loadall"
#define SHOW_LOADED_IMAGES_COUNT_CMD "showloaded"
#define TRAIN_CMD "train"
#define TEST_CMD "test"
#define HELP_CMD "help"
#define DISPLAY_CMD "display"
#define PRINT_INFO_CMD "printinfo"

#define CREATE_NETWORK_ARG_COUNT 0
#define UPDATE_NETWORK_ARG_COUNT 0
#define LOAD_N_IMAGES_ARG_COUNT 2 // argument 1 - number of images, argument 2 - 0 for training set, 1 for test set
#define LOAD_ALL_IMAGES_ARG_COUNT 1 // 0 for training set, 1 for test set
#define SHOW_LOADED_IMAGES_COUNT_ARG_COUNT 0
#define TRAIN_ARG_COUNT 1 // 1 argument - number of epochs
#define TEST_ARG_COUNT 0
#define HELP_ARG_COUNT 0
#define DISPLAY_ARG_COUNT 2 // 1 argument - image index, 2 argument - 0 for training set, 1 for test set
#define PRINT_INFO_ARG_COUNT 1 // 1 argument - 0 for base info, 1 for detailed info

#define CREATE_NETWORK_DESCRIPTION "Initializes neural network"
#define UPDATE_NETWORK_DESCRIPTION "Updates neural network"
#define LOAD_N_IMAGES_DESCRIPTION "Loads n images from MNIST database\nArguments: 1: number of images, 2: 0 for training set, 1 for test set"
#define LOAD_ALL_IMAGES_DESCRIPTION "Loads all images from MNIST database\nArguments: 1: 0 for training set, 1 for test set"
#define SHOW_LOADED_IMAGES_COUNT_DESCRIPTION "Shows number of loaded images"
#define TRAIN_DESCRIPTION "Trains neural network\nArguments: 1: number of epochs"
#define TEST_DESCRIPTION "Tests neural network"
#define HELP_DESCRIPTION "Shows help"
#define DISPLAY_DESCRIPTION "Displays image\nArguments: 1: image index, 2: 0 for training set, 1 for test set"
#define PRINT_INFO_DESCRIPTION "Prints info about neural network\nArguments: 1: 0 for base info, 1 for detailed info"

typedef void (*commandFunction)(uint16_t *arg);

typedef struct
{
    char name[15];
    uint8_t argCount;
    commandFunction function;
    char description[200];
} command;

command commands[] = 
{
    {CREATE_NETWORK_CMD, CREATE_NETWORK_ARG_COUNT, NULL, CREATE_NETWORK_DESCRIPTION},
    {UPDATE_NETWORK_CMD, UPDATE_NETWORK_ARG_COUNT, NULL, UPDATE_NETWORK_DESCRIPTION},
    {LOAD_N_IMAGES_CMD, LOAD_N_IMAGES_ARG_COUNT, NULL, LOAD_N_IMAGES_DESCRIPTION},
    {LOAD_ALL_IMAGES_CMD, LOAD_ALL_IMAGES_ARG_COUNT, NULL, LOAD_ALL_IMAGES_DESCRIPTION},
    {SHOW_LOADED_IMAGES_COUNT_CMD, SHOW_LOADED_IMAGES_COUNT_ARG_COUNT, NULL, SHOW_LOADED_IMAGES_COUNT_DESCRIPTION},
    {TRAIN_CMD, TRAIN_ARG_COUNT, NULL, TRAIN_DESCRIPTION},
    {TEST_CMD, TEST_ARG_COUNT, NULL, TEST_DESCRIPTION},
    {HELP_CMD, HELP_ARG_COUNT, NULL, HELP_DESCRIPTION},
    {DISPLAY_CMD, DISPLAY_ARG_COUNT, NULL, DISPLAY_DESCRIPTION},
    {PRINT_INFO_CMD, PRINT_INFO_ARG_COUNT, NULL, PRINT_INFO_DESCRIPTION}
};

uint8_t commandCount = sizeof(commands) / sizeof(command);

#endif // COMMANDS_H
