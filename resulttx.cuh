#pragma once

#include "common.cuh"

ClosurePairs convert_internal_closure_to_pairs(const BitsetMatrixDevice &closure_dev,
                                               const DestMapping &mapping);

ClosurePairs compute_external_closure_gpu(const BitsetMatrixDevice &closure_dev,
                                          const DestMapping &mapping,
                                          const std::vector<Edge> &external_edges);
