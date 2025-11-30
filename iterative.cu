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
#include <unordered_set>
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

struct CSRGraph {
    int index_size = 0;                 // number of nodes in T
    std::vector<int> row_offsets;       // size index_size + 1
    std::vector<int> col_indices;       // size = #internal edges
};

struct CSRDevice {
    int index_size = 0;
    int nnz = 0;
    int* d_row_offsets = nullptr;
    int* d_col_indices = nullptr;
};

struct DestMapping {
    std::unordered_map<std::int64_t, int> id_to_index; // destId -> t_idx
    std::vector<std::int64_t> index_to_id;             // t_idx -> destinationId
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
    graph.index_size = index_size;
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

// ----- Device CSR -----

CSRDevice upload_csr_to_device(const CSRGraph &graph) {
    CSRDevice d;
    d.index_size = graph.index_size;
    d.nnz = static_cast<int>(graph.col_indices.size());

    check_cuda(cudaMalloc(&d.d_row_offsets,
                          (graph.index_size + 1) * sizeof(int)),
               "cudaMalloc d_row_offsets");
    check_cuda(cudaMalloc(&d.d_col_indices,
                          graph.col_indices.size() * sizeof(int)),
               "cudaMalloc d_col_indices");

    check_cuda(cudaMemcpy(d.d_row_offsets,
                          graph.row_offsets.data(),
                          (graph.index_size + 1) * sizeof(int),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy row_offsets");
    check_cuda(cudaMemcpy(d.d_col_indices,
                          graph.col_indices.data(),
                          graph.col_indices.size() * sizeof(int),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy col_indices");

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

/******************************
 *  CUDA kernels (Algorithm B)
 ******************************/

// Algorithm B uses the same initialisation as A: fill closure with direct edges from the CSR.
__global__ void algoB_initial_kernel(const int* __restrict__ row_offsets,
                                     const int* __restrict__ col_indices,
                                     int index_size, int words_per_row,
                                     unsigned int* __restrict__ closure) {
    int src = blockIdx.x * blockDim.x + threadIdx.x;
    if (src >= index_size) return;

    int row_start = row_offsets[src];
    int row_end   = row_offsets[src + 1];

    unsigned int* row = closure + static_cast<std::size_t>(src) * words_per_row;

    for (int e = row_start; e < row_end; ++e) {
        int dst = col_indices[e];
        int word_idx = dst / kBitsPerWord;
        int bit_pos  = dst % kBitsPerWord;
        unsigned int mask = 1u << bit_pos;
        row[word_idx] |= mask;
    }
}

// Algorithm B iterative step:
// For each node u:
//   closure[u] := closure[u] ∪ (⋃_{v ∈ Adj[u]} closure[v])
// This is done in-place; each block owns a row u exclusively, so no
// write-write races. Reads from closure[v] are read-only.
__global__ void algoB_iter_kernel(const int* __restrict__ row_offsets,
                                  const int* __restrict__ col_indices,
                                  int index_size, int words_per_row,
                                  unsigned int* __restrict__ closure,
                                  int* __restrict__ d_changed) {
    int u = blockIdx.x;
    if (u >= index_size) return;

    int row_start = row_offsets[u];
    int row_end   = row_offsets[u + 1];

    const std::size_t row_offset = static_cast<std::size_t>(u) * words_per_row;

    bool local_changed = false;

    for (int w = threadIdx.x; w < words_per_row; w += blockDim.x) {
        unsigned int word = closure[row_offset + w];
        unsigned int acc = word;

        // OR in closure[v,*] for all neighbors v of u
        for (int e = row_start; e < row_end; ++e) {
            int v = col_indices[e];
            const std::size_t v_offset = static_cast<std::size_t>(v) * words_per_row;
            acc |= closure[v_offset + w];
        }

        if (acc != word) {
            closure[row_offset + w] = acc;
            local_changed = true;
        }
    }

    if (local_changed) {
        atomicOr(d_changed, 1);
    }
}

/*************************************
 *  Host wrappers for algorithm B
 *************************************/

void run_algoB_initial(const CSRDevice &graph_dev,
                       BitsetMatrixDevice &closure_dev,
                       bool &changed) {
    if (graph_dev.index_size == 0) {
        changed = false;
        return;
    }

    dim3 block(128);
    dim3 grid((graph_dev.index_size + block.x - 1) / block.x);

    algoB_initial_kernel<<<grid, block>>>(graph_dev.d_row_offsets, graph_dev.d_col_indices,
                                          graph_dev.index_size,
                                          static_cast<int>(closure_dev.words_per_row),
                                          closure_dev.data);
    check_cuda(cudaDeviceSynchronize(), "algoB_initial_kernel");

    changed = true;  // we just wrote direct edges; we definitely need iterations
}

bool run_algoB_iterations(const CSRDevice &graph_dev,
                          BitsetMatrixDevice &closure_dev) {
    if (graph_dev.index_size == 0) return false;

    int *d_changed = nullptr;
    check_cuda(cudaMalloc(&d_changed, sizeof(int)), "cudaMalloc d_changed");
    int zero = 0;
    check_cuda(cudaMemcpy(d_changed, &zero, sizeof(int), cudaMemcpyHostToDevice),
               "cudaMemcpy d_changed init (B)");

    dim3 block(256);
    dim3 grid(graph_dev.index_size);

    algoB_iter_kernel<<<grid, block>>>(graph_dev.d_row_offsets, graph_dev.d_col_indices,
                                       graph_dev.index_size, static_cast<int>(closure_dev.words_per_row),
                                       closure_dev.data, d_changed);
    check_cuda(cudaDeviceSynchronize(), "algoB_iter_kernel");

    int h_changed = 0;
    check_cuda(cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost),
               "cudaMemcpy d_changed back (B)");

    cudaFree(d_changed);

    return (h_changed != 0);
}

/***********************************************
 *  Conversion from bitset -> (src,dst) pairs
 ***********************************************/

// Helper function to call fn(dst_idx) for each set bit in row `row_idx`.
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

ClosurePairs convert_closure_to_pairs(const BitsetMatrixDevice &closure_dev,
                                      const DestMapping &mapping,
                                      ClosurePairs &external_edges) {
    ClosurePairs result;

    const int index_size = closure_dev.index_size;
    if (index_size == 0) {
        // No internal nodesl closure is just external edges
        return external_edges;
    }

    const std::size_t words_per_row = closure_dev.words_per_row;
    const std::size_t num_words = closure_dev.num_words_total;

    // Copy bitset matrix from device to host.
    std::vector<unsigned int> closure_host(num_words);
    check_cuda(cudaMemcpy(closure_host.data(), closure_dev.data, num_words * sizeof(unsigned int), cudaMemcpyDeviceToHost),
               "cudaMemcpy closure_dev -> host");

    constexpr std::size_t kExpectedInternalEdges = 7600000;
    result.reserve(external_edges.size() + kExpectedInternalEdges);

    // Scan each row of the closure.
    for (int src_idx = 0; src_idx < index_size; ++src_idx) {
        const std::int64_t src_id = mapping.index_to_id[src_idx];

        for_each_set_bit_in_row(closure_host, index_size, words_per_row, src_idx,
                                [&](int dst_idx) {
                                    std::int64_t dst_id = mapping.index_to_id[dst_idx];
                                    result.emplace_back(src_id, dst_id);
                                });
    }

    // For each external source s_ext:
    //   dests(s_ext) = { direct dst_id } ∪ { all ancestors of each dst_id via closure matrix }.
    // Hash sets dedupe everything for a given s_ext, even if multiple external edges
    // point into overlapping parts of T.

    using DestSet  = std::unordered_set<std::int64_t>;

    // group the external edges by sources
    std::unordered_map<std::int64_t, std::vector<std::int64_t>> src_to_dsts;
    src_to_dsts.reserve(external_edges.size());

    for (const auto &e : external_edges) {
        src_to_dsts[e.first].push_back(e.second);
    }

    // get the keys from the source->dest_list map
    std::vector<std::int64_t> sources;
    sources.reserve(src_to_dsts.size());
    for (const auto &kv : src_to_dsts) {
        sources.push_back(kv.first);
    }

    // create a set of outputs for each source
    std::vector<DestSet> source_dests(sources.size());

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < static_cast<int>(sources.size()); ++i) {
        const std::int64_t src_ext = sources[i];
        const auto &dst_list = src_to_dsts[src_ext];

        DestSet &dests = source_dests[i];
        dests.reserve(dst_list.size() * 4);  // heuristic

        for (const std::int64_t dst_id : dst_list) {
            // Insert direct dest; only expand if this is the first time
            // we've seen this dst_id for this src_ext.
            auto [_, inserted] = dests.insert(dst_id);
            if (!inserted) {
                continue;
            }

            auto it = mapping.id_to_index.find(dst_id);
            if (it == mapping.id_to_index.end()) {
                continue;  // dst not in T; nothing to expand via matrix.
            }

            const int d_idx = it->second;
            for_each_set_bit_in_row(closure_host, index_size, words_per_row, d_idx,
                                    [&](int anc_idx) {
                                        std::int64_t anc_id = mapping.index_to_id[anc_idx];
                                        dests.insert(anc_id);
                                    });
        }
    }

    // Append external closures to result
    for (int i = 0; i < static_cast<int>(sources.size()); ++i) {
        const std::int64_t src_ext = sources[i];
        const DestSet &dests = source_dests[i];

        for (const auto &dst_id : dests) {
            result.emplace_back(src_ext, dst_id);
        }
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
        // Load data
        auto t0 = Clock::now();
        auto edges = load_isA_edges(input_path);
        auto t1 = Clock::now();
        std::cout << "Step load_isA_edges: "
                  << std::chrono::duration<double, std::milli>(t1 - t0).count()
                  << " ms\n";

        // Build destinationId <-> index mapping
        t0 = Clock::now();
        auto tCommon0 = t0;
        DestMapping mapping = build_dest_mapping(edges);
        t1 = Clock::now();
        int index_size = static_cast<int>(mapping.index_to_id.size());
        std::cout << "Unique destinationIds (index_size): " << index_size << "\n";
        std::cout << "Step build_dest_mapping: "
                  << std::chrono::duration<double, std::milli>(t1 - t0).count()
                  << " ms\n";

        // Build CSR + external edge list
        std::size_t num_internal_edges = 0;
        std::size_t num_external_edges = 0;
        std::vector<Edge> external_edges;

        CSRGraph graph = build_csr_internal(edges, mapping, num_internal_edges, num_external_edges, external_edges);
        t1 = Clock::now();

        std::cout << "CSR graph over T:\n";
        std::cout << "  index_size (|T|)      : " << graph.index_size << "\n";
        std::cout << "  Internal edges (src∈T): " << num_internal_edges << "\n";
        std::cout << "  External edges (src∉T): " << num_external_edges << "\n";
        std::cout << "  External edge records : " << external_edges.size() << "\n";
        std::cout << "Step build_csr_internal: "
                  << std::chrono::duration<double, std::milli>(t1 - t0).count()
                  << " ms\n";

        // Upload CSR to device
        t0 = Clock::now();
        CSRDevice graph_dev = upload_csr_to_device(graph);
        t1 = Clock::now();
        auto tCommon1 = t1;
        std::cout << "Step upload_csr_to_device: "
                  << std::chrono::duration<double, std::milli>(t1 - t0).count()
                  << " ms\n";

        auto common_time = std::chrono::duration<double, std::milli>(tCommon1 - tCommon0).count();

        // Algorithm B: bitset matrix, initial + iterations
        t0 = Clock::now();
        auto tB0 = t0;
        BitsetMatrixDevice closureB_dev = allocate_bitset_matrix_device(index_size);
        t1 = Clock::now();
        std::cout << "Step allocate_bitset_matrix_device (B): "
                  << std::chrono::duration<double, std::milli>(t1 - t0).count()
                  << " ms\n";

        bool changed = false;

        t0 = Clock::now();
        run_algoB_initial(graph_dev, closureB_dev, changed);
        t1 = Clock::now();
        std::cout << "Algorithm B initial fill: "
                  << std::chrono::duration<double, std::milli>(t1 - t0).count()
                  << " ms, changed=" << (changed ? "true" : "false") << "\n";

        int iter_countB = 0;
        while (changed) {
            auto ti0 = Clock::now();
            changed = run_algoB_iterations(graph_dev, closureB_dev);
            auto ti1 = Clock::now();
            ++iter_countB;
            std::cout << "Algorithm B iteration " << iter_countB << " took "
                      << std::chrono::duration<double, std::milli>(ti1 - ti0).count()
                      << " ms, changed=" << (changed ? "true" : "false") << "\n";
        }
        std::cout << "Algorithm B iterations until fixed point: " << iter_countB << "\n";

        t0 = Clock::now();
        ClosurePairs closureB_pairs = convert_closure_to_pairs(closureB_dev, mapping, external_edges);
        t1 = Clock::now();
        auto tB1 = t1;
        std::cout << "Algorithm B convert_closure_to_pairs: "
                  << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n";
        std::cout << "Algorithm B closure total pairs (including external): "
                  << closureB_pairs.size() << "\n";
        std::cout << "Algorithm B total time: "
                  << (common_time + std::chrono::duration<double, std::milli>(tB1 - tB0).count()) << " ms\n";

        // 12. Cleanup device resources
        free_csr_device(graph_dev);
        free_bitset_matrix_device(closureB_dev);

        // 13. Write results
        t0 = Clock::now();
        std::ofstream file_out(output_path);
        for (const auto &p : closureB_pairs) {
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
