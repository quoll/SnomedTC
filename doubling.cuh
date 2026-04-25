#pragma once

#include "common.cuh"

#include <string>

std::vector<Edge> load_isA_edges(const std::string &input_path);

DestMapping build_dest_mapping(const std::vector<Edge> &edges);

CSRGraph build_csr_internal(const std::vector<Edge> &edges, const DestMapping &mapping,
                            std::size_t &num_internal_edges, std::size_t &num_external_edges,
                            std::vector<Edge> &external_edges_out);

void run_algoA_initial(const CSRDevice &graph_dev, BitsetMatrixDevice &closure_dev);

bool run_algoA_iterations(BitsetMatrixDevice &closure_in, BitsetMatrixDevice &closure_out);
