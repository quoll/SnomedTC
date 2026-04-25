#pragma once

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <unordered_map>
#include <utility>
#include <vector>

// Note: comments and printed results include unicode characters: ⋃∈∉

// SNOMED 'isA' relationship typeId.
constexpr const char* kIsATypeId = "116680003";

// Use 32-bit words for the bitset matrix.
constexpr int kBitsPerWord = 8 * sizeof(unsigned int);

// ----- Host-side types -----

struct ColumnIndices {
    int source_idx = -1;
    int dest_idx   = -1;
    int type_idx   = -1;
    int active_idx = -1;
};

// compressed sparse row (CSR) representation of a boolean matrix stored on the host
struct CSRGraph {
    int num_rows = 0;                   // number of nodes in T
    std::vector<int> row_offsets;       // size num_rows + 1
    std::vector<int> col_indices;       // size = #internal edges
};

// CSR representation of a boolean matrix stored on the GPU.
// Used for both the internal graph and the external-source CSR;
// num_rows is the number of rows (nodes in T, or external sources).
struct CSRDevice {
    int num_rows = 0;
    int nnz = 0;
    int* d_row_offsets = nullptr;
    int* d_col_indices = nullptr;
};

// SNOMED ids are sparse. Map each SNOMED id to an integer in the range 0-N
// where N is the number of SNOMED ids
struct DestMapping {
    std::unordered_map<std::int64_t, int> id_to_index; // destinationId -> t_idx
    std::vector<std::int64_t> index_to_id;             // t_idx -> destinationId
};

// CSR of edges that terminate paths through the graph, on the host
struct ExternalCSRHost {
    std::vector<std::int64_t> src_ids;   // unique external sourceIds
    std::vector<int> row_offsets;        // size = num_srcs + 1
    std::vector<int> dst_indices;        // internal t_idx for each edge
};


// mapping of index values 0-index_size to the associated SNOMED id, on the GPU
struct DestMappingDevice {
    int index_size = 0;
    std::int64_t* d_index_to_id = nullptr;
};

using Edge = std::pair<std::int64_t, std::int64_t>;  // (src_id, dst_id)
using ClosurePairs = std::vector<Edge>;

// small timing helper
using Clock = std::chrono::steady_clock;

// ----- Bitset matrix on device -----

struct BitsetMatrixDevice {
    int index_size = 0;
    std::size_t words_per_row = 0;    // number of 32-bit words per row
    std::size_t num_words_total = 0;
    unsigned int* data = nullptr;     // device pointer
};

#ifdef __CUDACC__

// CUDA error check
inline void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in " << msg << ": "
                  << cudaGetErrorString(err) << "\n";
        std::exit(1);
    }
}

// ----- Device CSR -----

// Allocates a device buffer and copies a host int vector into it.
static inline int* upload_int_array(const std::vector<int> &v, const char* label) {
    int* d = nullptr;
    check_cuda(cudaMalloc(&d, v.size() * sizeof(int)), label);
    check_cuda(cudaMemcpy(d, v.data(), v.size() * sizeof(int),
                          cudaMemcpyHostToDevice), label);
    return d;
}

inline CSRDevice upload_csr_to_device(const CSRGraph &graph) {
    CSRDevice d;
    d.num_rows = graph.num_rows;
    d.nnz      = static_cast<int>(graph.col_indices.size());
    d.d_row_offsets = upload_int_array(graph.row_offsets, "upload row_offsets");
    d.d_col_indices = upload_int_array(graph.col_indices, "upload col_indices");
    return d;
}

inline void free_csr_device(CSRDevice &d) {
    if (d.d_row_offsets) cudaFree(d.d_row_offsets);
    if (d.d_col_indices) cudaFree(d.d_col_indices);
    d.d_row_offsets = nullptr;
    d.d_col_indices = nullptr;
}

inline BitsetMatrixDevice allocate_bitset_matrix_device(int index_size) {
    BitsetMatrixDevice m;
    m.index_size = index_size;
    m.words_per_row = (static_cast<std::size_t>(index_size) + kBitsPerWord - 1) / kBitsPerWord;
    m.num_words_total = m.words_per_row * static_cast<std::size_t>(index_size);

    check_cuda(cudaMalloc(&m.data,
                          m.num_words_total * sizeof(unsigned int)),
               "cudaMalloc bitset matrix");
    check_cuda(cudaMemset(m.data, 0,
                          m.num_words_total * sizeof(unsigned int)),
               "cudaMemset bitset matrix");
    return m;
}

inline void free_bitset_matrix_device(BitsetMatrixDevice &m) {
    if (m.data) cudaFree(m.data);
    m.data = nullptr;
}

#endif // __CUDACC__
