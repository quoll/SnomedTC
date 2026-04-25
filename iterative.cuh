#pragma once
#include "common.cuh"

void run_algoB_initial(const CSRDevice &graph_dev,
                       BitsetMatrixDevice &closure_dev,
                       bool &changed);

bool run_algoB_iterations(const CSRDevice &graph_dev,
                          BitsetMatrixDevice &closure_dev);
