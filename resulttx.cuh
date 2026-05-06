#pragma once

#include "common.cuh"

ClosurePairs retrieve_results(const BitsetMatrixDevice &closure_dev,
                              const DestMapping &mapping,
                              const std::vector<Edge> &external_edges);
