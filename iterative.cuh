#pragma once
#include "common.cuh"

struct FrontierDevice {
    unsigned int* current;   // nodes whose closure changed last iteration
    unsigned int* next;      // nodes whose closure changes this iteration (being built)
    int           words;     // number of unsigned ints in each bitset
    int           index_size;
};

FrontierDevice run_algoB_initial(const CSRDevice &graph_dev,
                                 BitsetMatrixDevice &closure_dev);

bool run_algoB_iterations(const CSRDevice &graph_dev,
                          BitsetMatrixDevice &closure_dev,
                          FrontierDevice &frontier_dev);

void free_frontier_device(FrontierDevice &frontier_dev);
