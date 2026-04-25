#include "doubling.cuh"
#include "resulttx.cuh"

#include <fstream>
#include <omp.h>
#include <stdexcept>
#include <string>
#include <unordered_set>

using AdjMap = std::unordered_map<std::int64_t, std::unordered_set<std::int64_t>>; // (src_id -> set(dst_id))

// ----- Host Adjacency -----
AdjMap build_adjacency_from_edges(const std::vector<Edge> &edges) {
    AdjMap conn;
    conn.reserve(edges.size() / 2); // very rough guess

    for (const auto &e : edges) {
        const auto src = e.first;
        const auto dst = e.second;
        conn[src].insert(dst);
    }

    return conn;
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

    // Reduce changed flag within block using shared memory
    __shared__ int block_changed;
    if (threadIdx.x == 0) block_changed = 0;
    __syncthreads();

    if (local_changed) {
        atomicOr(&block_changed, 1);
    }
    __syncthreads();

    if (threadIdx.x == 0 && block_changed) {
        atomicOr(d_changed, 1);
    }
}

/*************************************
 *  Host wrappers for algorithm A
 *************************************/

// run_algoA_initial and run_algoA_iterations are provided by doubling.cu

/*************************************
 *  Host wrappers for algorithm B
 *************************************/

void run_algoB_initial(const CSRDevice &graph_dev,
                       BitsetMatrixDevice &closure_dev,
                       bool &changed) {
    if (graph_dev.num_rows == 0) {
        changed = false;
        return;
    }

    dim3 block(128);
    dim3 grid((graph_dev.num_rows + block.x - 1) / block.x);

    algoB_initial_kernel<<<grid, block>>>(graph_dev.d_row_offsets, graph_dev.d_col_indices,
                                          graph_dev.num_rows,
                                          static_cast<int>(closure_dev.words_per_row),
                                          closure_dev.data);
    check_cuda(cudaDeviceSynchronize(), "algoB_initial_kernel");

    changed = true;  // we just wrote direct edges; we definitely need iterations
}

bool run_algoB_iterations(const CSRDevice &graph_dev,
                          BitsetMatrixDevice &closure_dev) {
    if (graph_dev.num_rows == 0) return false;

    int *d_changed = nullptr;
    check_cuda(cudaMalloc(&d_changed, sizeof(int)), "cudaMalloc d_changed");
    int zero = 0;
    check_cuda(cudaMemcpy(d_changed, &zero, sizeof(int), cudaMemcpyHostToDevice),
               "cudaMemcpy d_changed init (B)");

    dim3 block(256);
    dim3 grid(graph_dev.num_rows);

    algoB_iter_kernel<<<grid, block>>>(graph_dev.d_row_offsets, graph_dev.d_col_indices,
                                       graph_dev.num_rows, static_cast<int>(closure_dev.words_per_row),
                                       closure_dev.data, d_changed);
    check_cuda(cudaDeviceSynchronize(), "algoB_iter_kernel");

    int h_changed = 0;
    check_cuda(cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost),
               "cudaMemcpy d_changed back (B)");

    cudaFree(d_changed);

    return (h_changed != 0);
}

/**************************************************************
 * Host iteration algorithm C (serialized form of Algorithm A)
 **************************************************************/
AdjMap compute_transitive_closure_serial(AdjMap conn, int max_iterations = 64) {
    for (int iter = 0; iter < max_iterations; ++iter) {
        AdjMap nxt;          // new edges of doubled length
        bool any_new = false;

        for (auto &entry : conn) {
            const std::int64_t s = entry.first;
            auto &tset = entry.second;

            for (const auto t : tset) {
                auto it_tTargets = conn.find(t);
                if (it_tTargets == conn.end()) {
                    continue; // t is not a source; nothing to join
                }

                const auto &tTargets = it_tTargets->second;

                for (const auto u : tTargets) {
                    // Check if we already know s -> u from previous iterations
                    auto &s_targets = conn[s];
                    if (s_targets.find(u) != s_targets.end()) {
                        continue;
                    }

                    // Check if it's already scheduled to be added this iteration
                    auto &nxt_targets = nxt[s];
                    auto [_, inserted] = nxt_targets.insert(u);
                    if (inserted) {
                        any_new = true;
                    }
                }
            }
        }

        if (!any_new) {
            break; // reached fixed point
        }

        // Merge nxt into conn
        for (auto &entry : nxt) {
            const std::int64_t s = entry.first;
            auto &new_targets = entry.second;
            auto &existing = conn[s];  // creates empty set if absent
            existing.insert(new_targets.begin(), new_targets.end());
        }
    }

    return conn;
}

/****************************************************************
 *  Host conversion of map (source -> set(destination)) to pairs
 ****************************************************************/

ClosurePairs flatten_closure(const AdjMap &conn) {
    ClosurePairs pairs;
    // Rough guess: closure is usually several times bigger than |edges|
    pairs.reserve(conn.size() * 8);

    for (const auto &entry : conn) {
        const std::int64_t src = entry.first;
        const auto &tset = entry.second;
        for (const auto dst : tset) {
            pairs.emplace_back(src, dst);
        }
    }

    return pairs;
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

        // 8. Convert internal closure to pairs, then compute external closure on GPU
        t0 = Clock::now();
        ClosurePairs internal_pairsA = convert_internal_closure_to_pairs(*in, mapping);
        ClosurePairs external_pairsA = compute_external_closure_gpu(*in, mapping, external_edges);
        ClosurePairs closureA_pairs;
        closureA_pairs.reserve(internal_pairsA.size() + external_pairsA.size());
        closureA_pairs.insert(closureA_pairs.end(), internal_pairsA.begin(), internal_pairsA.end());
        closureA_pairs.insert(closureA_pairs.end(), external_pairsA.begin(), external_pairsA.end());
        t1 = Clock::now();
        auto tA1 = t1;
        std::cout << "Algorithm A convert + external closure: "
                  << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n";
        std::cout << "Algorithm A closure total pairs (including external): "
                  << closureA_pairs.size() << "\n";
        std::cout << "Algorithm A total time: "
                  << (common_time + std::chrono::duration<double, std::milli>(tA1 - tA0).count()) << " ms\n";

        // 9. Algorithm B: bitset matrix, initial + iterations
        t0 = Clock::now();
        auto tB0 = t0;
        BitsetMatrixDevice closureB_dev = allocate_bitset_matrix_device(index_size);
        t1 = Clock::now();
        std::cout << "Step allocate_bitset_matrix_device (B): "
                  << std::chrono::duration<double, std::milli>(t1 - t0).count()
                  << " ms\n";

        changed = false;

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
        ClosurePairs internal_pairsB = convert_internal_closure_to_pairs(closureB_dev, mapping);
        ClosurePairs external_pairsB = compute_external_closure_gpu(closureB_dev, mapping, external_edges);
        ClosurePairs closureB_pairs;
        closureB_pairs.reserve(internal_pairsB.size() + external_pairsB.size());
        closureB_pairs.insert(closureB_pairs.end(), internal_pairsB.begin(), internal_pairsB.end());
        closureB_pairs.insert(closureB_pairs.end(), external_pairsB.begin(), external_pairsB.end());
        t1 = Clock::now();
        auto tB1 = t1;
        std::cout << "Algorithm B convert + external closure: "
                  << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n";
        std::cout << "Algorithm B closure total pairs (including external): "
                  << closureB_pairs.size() << "\n";
        std::cout << "Algorithm B total time: "
                  << (common_time + std::chrono::duration<double, std::milli>(tB1 - tB0).count()) << " ms\n";

        // 10. Algorithm C: host serialized form of Algorithm A
        t0 = Clock::now();
        auto tC0 = t0;
        AdjMap conn0 = build_adjacency_from_edges(edges);
        t1 = Clock::now();
        std::cout << "Algorithm C build_adjacency_from_edges: "
                  << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n";
        t0 = Clock::now();
        auto conn_tc = compute_transitive_closure_serial(std::move(conn0));
        t1 = Clock::now();
        std::cout << "Algorithm C compute_transitive_closure_serial: "
                  << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n";
        t0 = Clock::now();
        ClosurePairs closureC_pairs = flatten_closure(conn_tc);
        t1 = Clock::now();
        auto tC1 = t1;
        std::cout << "Algorithm C flatten_closure to pairs: "
                  << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n";
        std::cout << "Algorithm C map-of-sets closure size: "
                  << closureC_pairs.size() << "\n";
        std::cout << "Algorithm C total time: "
                  << std::chrono::duration<double, std::milli>(tC1 - tC0).count() << " ms\n";

        // 11. Compare Algorithm A vs B vs C (sort + compare)
        ClosurePairs sortedA = closureA_pairs;
        ClosurePairs sortedB = closureB_pairs;
        ClosurePairs sortedC = closureC_pairs;
        std::sort(sortedA.begin(), sortedA.end());
        std::sort(sortedB.begin(), sortedB.end());
        std::sort(sortedC.begin(), sortedC.end());

        std::cout << "Algorithm A vs B results equal? "
                  << ((sortedA == sortedB) ? "YES" : "NO") << "\n";
        std::cout << "Algorithm A vs C results equal? "
                  << ((sortedA == sortedC) ? "YES" : "NO") << "\n";

        // 12. Cleanup device resources
        free_csr_device(graph_dev);
        free_bitset_matrix_device(closureA_in);
        free_bitset_matrix_device(closureA_out);
        free_bitset_matrix_device(closureB_dev);

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
