#pragma once
#include "common.cuh"

void run_algoA_initial(const CSRDevice &graph_dev, BitsetMatrixDevice &closure_dev);

bool run_algoA_iterations(BitsetMatrixDevice &closure_in, BitsetMatrixDevice &closure_out);
