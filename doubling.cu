#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <omp.h>

// Note: comments and printed results includes unicode characters: ⋃∈∉

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

// CUDA error check
inline void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in " << msg << ": "
                  << cudaGetErrorString(err) << "\n";
        std::exit(1);
    }
}

// ----- Parsing -----

std::vector<std::string> split_tab(const std::string &line) {
    std::vector<std::string> fields;
    std::string field;
    std::stringstream ss(line);
    while (std::getline(ss, field, '\t')) {
        fields.push_back(field);
    }
    return fields;
}

ColumnIndices parse_header(const std::string &header_line) {
    auto fields = split_tab(header_line);
    ColumnIndices idx;

    for (int i = 0; i < static_cast<int>(fields.size()); ++i) {
        const auto &name = fields[i];
        if (name == "sourceId") {
            idx.source_idx = i;
        } else if (name == "destinationId") {
            idx.dest_idx = i;
        } else if (name == "typeId") {
            idx.type_idx = i;
        } else if (name == "active") {
            idx.active_idx = i;
        }
    }

    if (idx.source_idx < 0 || idx.dest_idx < 0 || idx.type_idx < 0 || idx.active_idx < 0) {
        throw std::runtime_error("Failed to locate sourceId/destinationId/typeId/active in header");
    }

    return idx;
}

std::vector<Edge> load_isA_edges(const std::string &input_path) {
    std::ifstream in(input_path);
    if (!in) throw std::runtime_error("Failed to open input file: " + input_path);

    std::string line;
    if (!std::getline(in, line)) throw std::runtime_error("Input file is empty");

    ColumnIndices idx = parse_header(line);
    int max_idx = std::max({idx.source_idx, idx.dest_idx, idx.type_idx, idx.active_idx});

    std::vector<Edge> edges;
    std::size_t total_rows = 0;

    while (std::getline(in, line)) {
        if (line.empty()) continue;
        ++total_rows;

        auto fields = split_tab(line);
        if (max_idx >= static_cast<int>(fields.size())) continue;

        if (fields[idx.type_idx] == kIsATypeId && fields[idx.active_idx] == "1") {
            edges.push_back(Edge{std::stoll(fields[idx.source_idx]), std::stoll(fields[idx.dest_idx])});
        }
    }

    std::cout << "Total data rows (excluding header): " << total_rows << "\n";
    std::cout << "Loaded isA edges (src,dst pairs): " << edges.size() << "\n";

    return edges;
}

// ----- DestMapping -----

DestMapping build_dest_mapping(const std::vector<Edge> &edges) {
    DestMapping mapping;
    mapping.id_to_index.reserve(edges.size() * 2);

    int next_index = 0;
    for (const auto &e : edges) {
        auto [it, inserted] = mapping.id_to_index.try_emplace(e.second, next_index);
        if (inserted) {
            ++next_index;
        }
    }

    mapping.index_to_id.resize(next_index);
    for (const auto &kv : mapping.id_to_index) {
        mapping.index_to_id[kv.second] = kv.first;
    }

    return mapping;
}

// ----- CSR + external edges -----

CSRGraph build_csr_internal(const std::vector<Edge> &edges, const DestMapping &mapping,
                            std::size_t &num_internal_edges, std::size_t &num_external_edges,
                            std::vector<Edge> &external_edges_out) {
    const auto &index_to_id = mapping.index_to_id;
    const auto &id_to_index = mapping.id_to_index;
    const int index_size = static_cast<int>(index_to_id.size());

    std::vector<std::pair<int,int>> internal_edges;
    internal_edges.reserve(edges.size());

    num_external_edges = 0;
    external_edges_out.clear();

    auto map_id = [&](std::int64_t id) -> int {
        auto it = id_to_index.find(id);
        return (it == id_to_index.end()) ? -1 : it->second;
    };

    for (const auto &e : edges) {
        int src_index = map_id(e.first);
        int dst_index = map_id(e.second);

        if (dst_index < 0) {
            // In principle this shouldn't happen, but keep stats honest.
            ++num_external_edges;
            continue;
        }

        if (src_index >= 0) {
            // Internal edge: both ends in T.
            internal_edges.emplace_back(src_index, dst_index);
        } else {
            // External source, internal destination.
            ++num_external_edges;
            external_edges_out.push_back(e);
        }
    }

    num_internal_edges = internal_edges.size();

    CSRGraph graph;
    graph.num_rows = index_size;
    graph.row_offsets.assign(index_size + 1, 0);
    graph.col_indices.resize(num_internal_edges);

    // Count out-degree
    for (const auto &p : internal_edges) {
        ++graph.row_offsets[p.first + 1];
    }

    // Prefix sum
    for (int i = 0; i < index_size; ++i) {
        graph.row_offsets[i + 1] += graph.row_offsets[i];
    }

    // Fill adjacency
    std::vector<int> cursor = graph.row_offsets;
    for (const auto &p : internal_edges) {
        int src_index = p.first;
        int dst_index = p.second;
        int pos = cursor[src_index]++;
        graph.col_indices[pos] = dst_index;
    }

    // Sort neighbors in each row for determinism
    for (int u = 0; u < index_size; ++u) {
        int begin = graph.row_offsets[u];
        int end   = graph.row_offsets[u + 1];
        std::sort(graph.col_indices.begin() + begin,
                  graph.col_indices.begin() + end);
    }

    return graph;
}

ExternalCSRHost build_external_csr(
    const std::vector<Edge> &external_edges,
    const DestMapping &mapping)
{
    ExternalCSRHost csr;
    if (external_edges.empty()) {
        return csr;
    }

    const auto &id_to_index = mapping.id_to_index;

    // Map each external sourceId -> row index in src_ids
    std::unordered_map<std::int64_t, int> src_to_row;
    src_to_row.reserve(external_edges.size() / 4);

    for (const auto &e : external_edges) {
        const std::int64_t src_id = e.first;
        auto [it, inserted] = src_to_row.try_emplace(src_id,
                                                     static_cast<int>(csr.src_ids.size()));
        if (inserted) {
            csr.src_ids.push_back(src_id);
        }
    }

    const int num_srcs = static_cast<int>(csr.src_ids.size());
    csr.row_offsets.assign(num_srcs + 1, 0);

    // Count edges per external source row.
    for (const auto &e : external_edges) {
        const std::int64_t src_id = e.first;
        const std::int64_t dst_id = e.second;

        auto it_dst = id_to_index.find(dst_id);
        if (it_dst == id_to_index.end()) {
            continue; // should not happen, but be defensive
        }

        int row = src_to_row[src_id];
        ++csr.row_offsets[row + 1];
    }

    // Prefix-sum
    for (int i = 0; i < num_srcs; ++i) {
        csr.row_offsets[i + 1] += csr.row_offsets[i];
    }

    csr.dst_indices.resize(csr.row_offsets[num_srcs]);
    std::vector<int> cursor = csr.row_offsets;

    // Fill dst_indices
    for (const auto &e : external_edges) {
        const std::int64_t src_id = e.first;
        const std::int64_t dst_id = e.second;

        auto it_dst = id_to_index.find(dst_id);
        if (it_dst == id_to_index.end()) {
            continue;
        }

        int dst_idx = it_dst->second;
        int row = src_to_row[src_id];

        int pos = cursor[row]++;
        csr.dst_indices[pos] = dst_idx;
    }

    return csr;
}

// ----- Device CSR -----

// Allocates a device buffer and copies a host int vector into it.
static int* upload_int_array(const std::vector<int> &v, const char* label) {
    int* d = nullptr;
    check_cuda(cudaMalloc(&d, v.size() * sizeof(int)), label);
    check_cuda(cudaMemcpy(d, v.data(), v.size() * sizeof(int),
                          cudaMemcpyHostToDevice), label);
    return d;
}

CSRDevice upload_csr_to_device(const CSRGraph &graph) {
    CSRDevice d;
    d.num_rows = graph.num_rows;
    d.nnz      = static_cast<int>(graph.col_indices.size());
    d.d_row_offsets = upload_int_array(graph.row_offsets, "upload row_offsets");
    d.d_col_indices = upload_int_array(graph.col_indices, "upload col_indices");
    return d;
}

void free_csr_device(CSRDevice &d) {
    if (d.d_row_offsets) cudaFree(d.d_row_offsets);
    if (d.d_col_indices) cudaFree(d.d_col_indices);
    d.d_row_offsets = nullptr;
    d.d_col_indices = nullptr;
}

// ----- Bitset matrix on device -----

struct BitsetMatrixDevice {
    int index_size = 0;
    std::size_t words_per_row = 0;    // number of 32-bit words per row
    std::size_t num_words_total = 0;
    unsigned int* data = nullptr;     // device pointer
};

BitsetMatrixDevice allocate_bitset_matrix_device(int index_size) {
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

void free_bitset_matrix_device(BitsetMatrixDevice &m) {
    if (m.data) cudaFree(m.data);
    m.data = nullptr;
}

CSRDevice upload_external_csr_to_device(const ExternalCSRHost &csr) {
    CSRDevice d;
    d.num_rows = static_cast<int>(csr.src_ids.size());
    if (d.num_rows == 0) {
        return d;
    }
    d.nnz           = static_cast<int>(csr.dst_indices.size());
    d.d_row_offsets = upload_int_array(csr.row_offsets,  "upload ext row_offsets");
    d.d_col_indices = upload_int_array(csr.dst_indices,  "upload ext dst_indices");
    return d;
}

DestMappingDevice upload_dest_mapping_device(const DestMapping &mapping) {
    DestMappingDevice d;
    d.index_size = static_cast<int>(mapping.index_to_id.size());
    if (d.index_size == 0) return d;

    check_cuda(cudaMalloc(&d.d_index_to_id,
                          d.index_size * sizeof(std::int64_t)),
               "cudaMalloc d_index_to_id");
    check_cuda(cudaMemcpy(d.d_index_to_id,
                          mapping.index_to_id.data(),
                          d.index_size * sizeof(std::int64_t),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy index_to_id");

    return d;
}

void free_dest_mapping_device(DestMappingDevice &d) {
    if (d.d_index_to_id) cudaFree(d.d_index_to_id);
    d.d_index_to_id = nullptr;
    d.index_size = 0;
}

/******************************
 *  CUDA kernels (Algorithm A)
 ******************************/

// Initial kernel: populate closure with direct edges CSR[u] -> v
// closure is an index_size x index_size bit matrix stored row-major,
// each row having `words_per_row` 32-bit unsigned ints.
__global__ void algoA_initial_kernel(const int* __restrict__ row_offsets,
                                     const int* __restrict__ col_indices,
                                     int index_size, int words_per_row,
                                     unsigned int* __restrict__ closure) {
    int src = blockIdx.x * blockDim.x + threadIdx.x;
    if (src >= index_size) return;

    int row_start = row_offsets[src];
    int row_end   = row_offsets[src + 1];

    // Pointer to the beginning of this row in the bitset matrix
    unsigned int* row = closure + static_cast<std::size_t>(src) * words_per_row;

    for (int e = row_start; e < row_end; ++e) {
        int dst = col_indices[e];
        int word_idx = dst / kBitsPerWord;
        int bit_pos  = dst % kBitsPerWord;
        unsigned int mask = 1u << bit_pos;

        // One thread per row: no race on this row.
        // TODO: accumulate this for every 32nd bit (PAG)
        row[word_idx] |= mask;
    }
}

// Upper bound on how many mids (reachable nodes) a row can have in the closure.
// Based on SNOMED-CT stats, 256 is comfortably above the observed max (~141).
constexpr int kMaxMids = 1024;

// For each row 'a':
//  1) Build mids[] = { j | closure_in[a,j] == 1 } from the full row.
//  2) For each word w, compute closure_out[a,w] = closure_in[a,w] OR
//     (OR over all mids of closure_in[mid,w]).
// If any word in the row changes, mark d_changed = 1.
__global__ void algoA_iter_kernel(int index_size, int words_per_row,
                                  const unsigned int* __restrict__ closure_in,
                                  unsigned int* __restrict__ closure_out,
                                  int* __restrict__ d_changed) {
    int a = blockIdx.x;  // row index
    if (a >= index_size) return;

    __shared__ int mids[kMaxMids];
    __shared__ int mids_len;
    __shared__ int block_changed;
    __shared__ bool overflow_flag;

    if (threadIdx.x == 0) {
        mids_len = 0;
        block_changed = 0;
        overflow_flag = false;
    }
    __syncthreads();

    const std::size_t row_offset = static_cast<std::size_t>(a) * words_per_row;

    // Step 1: build mids[] from the full row (scan bits across all words).
    for (int w = threadIdx.x; w < words_per_row; w += blockDim.x) {
        unsigned int word = closure_in[row_offset + w];
        if (word == 0u) continue;

        unsigned int mask = word;
        while (mask) {
            int bit = __ffs(mask) - 1;     // position of lowest set bit [0..31]
            mask &= (mask - 1);            // clear that bit
            int mid = w * kBitsPerWord + bit;
            if (mid >= index_size) break;  // safety for last partial word

            int pos = atomicAdd(&mids_len, 1);
            if (pos < kMaxMids) {
                mids[pos] = mid;
            } else {
                overflow_flag = true;
            }
        }
    }

    __syncthreads();

    // Assume kMaxMids is large enough for this dataset.
    // For datasets where this is not the case, this requires an overflow into global memory
    int used_mids = mids_len;
    if (used_mids > kMaxMids) {
        used_mids = kMaxMids;
        overflow_flag = true;
    }

    bool local_changed = false;

    // Step 2: update words using mids[].
    for (int w = threadIdx.x; w < words_per_row; w += blockDim.x) {
        unsigned int word = closure_in[row_offset + w];
        unsigned int acc = word;

        // OR in contributions from all mids.
        for (int i = 0; i < used_mids; ++i) {
            int mid = mids[i];
            const std::size_t mid_offset = static_cast<std::size_t>(mid) * words_per_row;
            acc |= closure_in[mid_offset + w];
        }

        if (acc != word) {
            closure_out[row_offset + w] = acc;
            local_changed = true;
        } else {
            closure_out[row_offset + w] = word;
        }
    }

    if (local_changed) {
        block_changed = 1;
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        if (block_changed) {
          atomicOr(d_changed, 1);
        }
        if (overflow_flag) {
          printf("Overflow triggered\n");
        }
    }
}

__global__ void external_count_kernel(const int* __restrict__ ext_row_offsets,
                                      const int* __restrict__ ext_dst_indices,
                                      int num_srcs,
                                      const unsigned int* __restrict__ closure_in,
                                      int index_size, int words_per_row,
                                      unsigned int* __restrict__ counts,
                                      unsigned int* __restrict__ merged_rows) {
    int s = blockIdx.x;
    if (s >= num_srcs) return;

    int start = ext_row_offsets[s];
    int end   = ext_row_offsets[s + 1];

    // One block per external source; each thread accumulates over its word subset.
    __shared__ unsigned int partial[256];  // assumes blockDim.x <= 256

    unsigned int local = 0;

    for (int w = threadIdx.x; w < words_per_row; w += blockDim.x) {
        unsigned int acc = 0u;

        // OR together the rows for each internal dst, plus the direct dst bit.
        for (int e = start; e < end; ++e) {
            int d_idx = ext_dst_indices[e];

            int word_for_d = d_idx / kBitsPerWord;
            int bit_for_d  = d_idx % kBitsPerWord;
            if (word_for_d == w) {
                acc |= (1u << bit_for_d);
            }

            const std::size_t d_offset =
                static_cast<std::size_t>(d_idx) * words_per_row;
            acc |= closure_in[d_offset + w];
        }

        merged_rows[static_cast<std::size_t>(s) * words_per_row + w] = acc;
        local += __popc(acc);
    }

    partial[threadIdx.x] = local;
    __syncthreads();

    // Reduce within block
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            partial[threadIdx.x] += partial[threadIdx.x + offset];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        counts[s] = partial[0];
    }
}

// Finds all of the destinations for a given source, when the source is a
// value that never appears as a destination (an external edge source).
// Destination SNOMED ids are written to out_dests; row_cursors tracks the
// write position per source so that the host can pair each dst with its src.
__global__ void external_emit_kernel(int num_srcs,
                                     int index_size, int words_per_row,
                                     const unsigned int* __restrict__ merged_rows,
                                     const std::int64_t* __restrict__ index_to_id,
                                     int* __restrict__ row_cursors,
                                     std::int64_t* __restrict__ out_dests) {
    int s = blockIdx.x;
    if (s >= num_srcs) return;

    for (int w = threadIdx.x; w < words_per_row; w += blockDim.x) {
        unsigned int acc = merged_rows[static_cast<std::size_t>(s) * words_per_row + w];

        // Turn bits in `acc` into destination ids.
        while (acc) {
            int bit = __ffs(acc) - 1;
            acc &= (acc - 1);

            int dst_idx = w * kBitsPerWord + bit;
            if (dst_idx >= index_size) {
                continue;
            }

            int pos = atomicAdd(&row_cursors[s], 1);
            out_dests[pos] = index_to_id[dst_idx];
        }
    }
}

/*************************************
 *  Host wrappers for algorithm A
 *************************************/

void run_algoA_initial(const CSRDevice &graph_dev, BitsetMatrixDevice &closure_dev) {
    if (graph_dev.num_rows == 0) return;

    dim3 block(128);
    dim3 grid((graph_dev.num_rows + block.x - 1) / block.x);

    algoA_initial_kernel<<<grid, block>>>(graph_dev.d_row_offsets, graph_dev.d_col_indices,
                                          graph_dev.num_rows,
                                          static_cast<int>(closure_dev.words_per_row),
                                          closure_dev.data);
    check_cuda(cudaDeviceSynchronize(), "algoA_initial_kernel");
}

bool run_algoA_iterations(BitsetMatrixDevice &closure_in,
                          BitsetMatrixDevice &closure_out) {
    if (closure_in.index_size == 0) {
        return false;
    }

    // Sanity: matrices must match shape.
    if (closure_in.index_size != closure_out.index_size ||
        closure_in.words_per_row != closure_out.words_per_row) {
        throw std::runtime_error("run_algoA_iterations: closure_in/out shape mismatch");
    }

    // flag to indicate if a join has not yet reached a fixpoint
    int *d_changed = nullptr;
    check_cuda(cudaMalloc(&d_changed, sizeof(int)), "cudaMalloc d_changed");
    int zero = 0;
    check_cuda(cudaMemcpy(d_changed, &zero, sizeof(int), cudaMemcpyHostToDevice),
               "cudaMemcpy d_changed init");

    dim3 block(256);
    dim3 grid(closure_in.index_size);  // one block per row

    algoA_iter_kernel<<<grid, block>>>(closure_in.index_size, static_cast<int>(closure_in.words_per_row),
                                       closure_in.data, closure_out.data, d_changed);
    check_cuda(cudaDeviceSynchronize(), "algoA_iter_kernel");

    int h_changed = 0;
    check_cuda(cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost),
               "cudaMemcpy d_changed back");

    cudaFree(d_changed);

    return (h_changed != 0);
}

// Connects all of the external (path-terminating) edges to the rest of the graph, on the GPU.
// Return the result to the host. This means:
// 1. converting the external edges to CSR form
// 2. uploading those external edges and the SNOMED-id mapping to the GPU
// 3. determining the row sizes needed for the final results, saving merged rows for reuse
// 4. finding the locations of the row starts for the output via prefix sum (on host)
// 5. allocating the destination id array and row cursors on the device
// 6. emitting destination ids into the output array
// 7. copying the destination ids back to the host
// 8. pairing each source id (already on host) with its destination ids
ClosurePairs compute_external_closure_gpu(const BitsetMatrixDevice &closure_dev,
                                          const DestMapping &mapping,
                                          const std::vector<Edge> &external_edges) {
    ClosurePairs result;

    if (external_edges.empty()) {
        return result;
    }

    // 1. Build external CSR on host
    ExternalCSRHost ext_csr_host = build_external_csr(external_edges, mapping);
    if (ext_csr_host.src_ids.empty()) {
        return result;
    }

    // 2. Upload external CSR and mapping to device
    CSRDevice ext_csr_dev = upload_external_csr_to_device(ext_csr_host);
    DestMappingDevice mapping_dev = upload_dest_mapping_device(mapping);

    const int num_srcs = ext_csr_dev.num_rows;
    const int index_size = closure_dev.index_size;
    const int words_per_row = static_cast<int>(closure_dev.words_per_row);

    // 3. Count how many pairs we will emit per external source,
    //    and save the merged rows for reuse by the emit kernel.
    unsigned int* d_counts = nullptr;
    check_cuda(cudaMalloc(&d_counts, num_srcs * sizeof(unsigned int)),
               "cudaMalloc d_counts");
    check_cuda(cudaMemset(d_counts, 0, num_srcs * sizeof(unsigned int)),
               "cudaMemset d_counts");

    unsigned int* d_merged = nullptr;
    check_cuda(cudaMalloc(&d_merged,
                          static_cast<std::size_t>(num_srcs) * words_per_row * sizeof(unsigned int)),
               "cudaMalloc d_merged");

    dim3 block(256);
    dim3 grid(num_srcs);

    external_count_kernel<<<grid, block>>>(
        ext_csr_dev.d_row_offsets,
        ext_csr_dev.d_col_indices,
        num_srcs,
        closure_dev.data,
        index_size,
        words_per_row,
        d_counts,
        d_merged
    );
    check_cuda(cudaDeviceSynchronize(), "external_count_kernel");

    // 4. Prefix-sum counts on host to get row offsets
    std::vector<unsigned int> counts_host(num_srcs);
    check_cuda(cudaMemcpy(counts_host.data(), d_counts,
                          num_srcs * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy counts_host");
    cudaFree(d_counts);

    std::vector<int> offsets(num_srcs + 1);
    offsets[0] = 0;
    for (int i = 0; i < num_srcs; ++i) {
        offsets[i + 1] = offsets[i] + static_cast<int>(counts_host[i]);
    }

    const int total_pairs = offsets[num_srcs];
    if (total_pairs == 0) {
        cudaFree(d_merged);
        free_csr_device(ext_csr_dev);
        free_dest_mapping_device(mapping_dev);
        return result;
    }

    // 5. Allocate output destinations + row cursors on device
    std::int64_t* d_dests = nullptr;
    check_cuda(cudaMalloc(&d_dests, total_pairs * sizeof(std::int64_t)),
               "cudaMalloc d_dests");

    int* d_row_cursors = nullptr;
    check_cuda(cudaMalloc(&d_row_cursors, num_srcs * sizeof(int)),
               "cudaMalloc d_row_cursors");
    check_cuda(cudaMemcpy(d_row_cursors, offsets.data(),
                          num_srcs * sizeof(int),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy d_row_cursors");

    // 6. Emit destination ids using the pre-computed merged rows
    external_emit_kernel<<<grid, block>>>(
        num_srcs,
        index_size,
        words_per_row,
        d_merged,
        mapping_dev.d_index_to_id,
        d_row_cursors,
        d_dests
    );
    check_cuda(cudaDeviceSynchronize(), "external_emit_kernel");

    // 7. Copy destination ids back to host
    std::vector<std::int64_t> dests_host(total_pairs);
    check_cuda(cudaMemcpy(dests_host.data(), d_dests,
                          total_pairs * sizeof(std::int64_t),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy dests_host");
    cudaFree(d_dests);
    cudaFree(d_row_cursors);
    cudaFree(d_merged);

    free_csr_device(ext_csr_dev);
    free_dest_mapping_device(mapping_dev);

    // 8. Pair each source id (from host) with its destination ids
    result.reserve(total_pairs);
    for (int s = 0; s < num_srcs; ++s) {
        const std::int64_t src_id = ext_csr_host.src_ids[s];
        for (int i = offsets[s]; i < offsets[s + 1]; ++i) {
            result.emplace_back(src_id, dests_host[i]);
        }
    }

    return result;
}

/***********************************************
 *  Conversion from bitset -> (src,dst) pairs
 ***********************************************/

// Helper function to call fn(dst_idx) for each set bit in row `row_idx`.
// Iterates across the row SERIALLY
template <typename Fn>
static void for_each_set_bit_in_row(const std::vector<unsigned int> &closure_host,
                                    int index_size, std::size_t words_per_row,
                                    int row_idx, Fn &&fn) {
    const std::size_t row_offset = static_cast<std::size_t>(row_idx) * words_per_row;

    for (int w = 0; w < static_cast<int>(words_per_row); ++w) {
        unsigned int word = closure_host[row_offset + w];
        if (word == 0u) continue;

        while (word) {
            int bit = __builtin_ctz(word);   // position [0..31] of the lowest order bit
            word &= (word - 1);              // clear that bit

            int dst_idx = w * kBitsPerWord + bit;
            if (dst_idx >= index_size) {
                break;  // last partial word can overshoot
            }

            fn(dst_idx);
        }
    }
}

// retrieves the matrix containing the graph of non-terminal (non-external) edges
// converts this matrix into source/destination pairs of connected SNOMED codes
// conversion is SERIAL
ClosurePairs convert_internal_closure_to_pairs(const BitsetMatrixDevice &closure_dev,
                                               const DestMapping &mapping) {
    ClosurePairs result;

    const int index_size = closure_dev.index_size;
    if (index_size == 0) {
        return result;
    }

    const std::size_t words_per_row = closure_dev.words_per_row;
    const std::size_t num_words     = closure_dev.num_words_total;

    std::vector<unsigned int> closure_host(num_words);
    check_cuda(cudaMemcpy(closure_host.data(), closure_dev.data,
                          num_words * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy closure_dev -> host");

    // Rough heuristic reserve
    result.reserve(8 * mapping.index_to_id.size());

    for (int src_idx = 0; src_idx < index_size; ++src_idx) {
        const std::int64_t src_id = mapping.index_to_id[src_idx];

        for_each_set_bit_in_row(closure_host, index_size, words_per_row, src_idx,
                                [&](int dst_idx) {
                                    std::int64_t dst_id = mapping.index_to_id[dst_idx];
                                    result.emplace_back(src_id, dst_id);
                                });
    }

    return result;
}


/*****************
 *  Main program
 *****************/

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "Usage: snomed_tc <input_snomed_file> <output_file>\n";
        return 1;
    }

    const std::string input_path  = argv[1];
    const std::string output_path = argv[2];

    try {
        // 1. Load data
        auto t0 = Clock::now();
        auto edges = load_isA_edges(input_path);
        auto t1 = Clock::now();
        std::cout << "Step load_isA_edges: "
                  << std::chrono::duration<double, std::milli>(t1 - t0).count()
                  << " ms\n";

        // 2. Build destinationId <-> index mapping
        t0 = Clock::now();
        auto tCommon0 = t0;
        DestMapping mapping = build_dest_mapping(edges);
        t1 = Clock::now();
        int index_size = static_cast<int>(mapping.index_to_id.size());
        std::cout << "Unique destinationIds (index_size): " << index_size << "\n";
        std::cout << "Step build_dest_mapping: "
                  << std::chrono::duration<double, std::milli>(t1 - t0).count()
                  << " ms\n";

        // 3. Build CSR + external edge list
        std::size_t num_internal_edges = 0;
        std::size_t num_external_edges = 0;
        std::vector<Edge> external_edges;

        CSRGraph graph = build_csr_internal(edges, mapping, num_internal_edges, num_external_edges, external_edges);
        t1 = Clock::now();

        std::cout << "CSR graph over T:\n";
        std::cout << "  num_rows (|T|)        : " << graph.num_rows << "\n";
        std::cout << "  Internal edges (src∈T): " << num_internal_edges << "\n";
        std::cout << "  External edges (src∉T): " << num_external_edges << "\n";
        std::cout << "  External edge records : " << external_edges.size() << "\n";
        std::cout << "Step build_csr_internal: "
                  << std::chrono::duration<double, std::milli>(t1 - t0).count()
                  << " ms\n";

        // 4. Upload CSR to device
        t0 = Clock::now();
        CSRDevice graph_dev = upload_csr_to_device(graph);
        t1 = Clock::now();
        auto tCommon1 = t1;
        std::cout << "Step upload_csr_to_device: "
                  << std::chrono::duration<double, std::milli>(t1 - t0).count()
                  << " ms\n";

        auto common_time = std::chrono::duration<double, std::milli>(tCommon1 - tCommon0).count();

        // 5. Allocate device memory for results
        t0 = Clock::now();
        auto tA0 = t0;
        BitsetMatrixDevice closureA_in = allocate_bitset_matrix_device(index_size);
        BitsetMatrixDevice closureA_out = allocate_bitset_matrix_device(index_size);
        t1 = Clock::now();
        std::cout << "Algorithm A initial fill: "
                  << std::chrono::duration<double, std::milli>(t1 - t0).count()
                  << " ms\n";

        // 6. Algorithm A: initial adjacency fill
        run_algoA_initial(graph_dev, closureA_in);

        // 7. Algorithm A: iterative expansion using bitset-only kernel
        bool changed = true;
        BitsetMatrixDevice* in = &closureA_in;
        BitsetMatrixDevice* out = &closureA_out;

        int iter_countA = 0;
        while (changed) {
            auto ti0 = Clock::now();
            changed = run_algoA_iterations(*in, *out);
            auto ti1 = Clock::now();
            ++iter_countA;
            std::cout << "Algorithm A iteration " << iter_countA << " took "
                      << std::chrono::duration<double, std::milli>(ti1 - ti0).count()
                      << " ms, changed=" << (changed ? "true" : "false") << "\n";
            std::swap(in, out);
        }
        // final output will be referenced by `in`
        std::cout << "Algorithm A iterations until fixed point: " << iter_countA << "\n";

        // 8. Convert internal closure to (srcId, dstId) pairs
        t0 = Clock::now();
        ClosurePairs internal_pairs = convert_internal_closure_to_pairs(*in, mapping);

        // 8a. Compute external closure on the GPU and append
        ClosurePairs external_pairs_gpu = compute_external_closure_gpu(*in, mapping, external_edges);

        ClosurePairs closureA_pairs;
        closureA_pairs.reserve(internal_pairs.size() + external_pairs_gpu.size());
        closureA_pairs.insert(closureA_pairs.end(), internal_pairs.begin(), internal_pairs.end());
        closureA_pairs.insert(closureA_pairs.end(), external_pairs_gpu.begin(), external_pairs_gpu.end());
        t1 = Clock::now();
        std::cout << "Algorithm A closure total pairs (including external): " << closureA_pairs.size() << "\n";
        auto tA1 = t1;
        std::cout << "Algorithm A total time: "
                  << (common_time + std::chrono::duration<double, std::milli>(tA1 - tA0).count()) << " ms\n";

        // 12. Cleanup device resources
        free_csr_device(graph_dev);
        free_bitset_matrix_device(closureA_in);
        free_bitset_matrix_device(closureA_out);

        // 13. Write results
        t0 = Clock::now();
        std::ofstream file_out(output_path);
        for (const auto &p : closureA_pairs) {
            file_out << p.first << '\t' << p.second << '\n';
        }
        t1 = Clock::now();
        std::cout << "Writing output file: "
                  << std::chrono::duration<double, std::milli>(t1 - t0).count()
                  << " ms\n";

    } catch (const std::exception &ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
