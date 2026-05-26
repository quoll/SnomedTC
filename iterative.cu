#include "iterative.cuh"

/******************************
 *  CUDA kernels (Algorithm B)
 ******************************/

// Fills closure with direct edges from the CSR and marks each source node in
// d_frontier so the first iteration knows which rows have content to propagate.
__global__ void algoB_initial_kernel(const int* __restrict__ row_offsets,
                                     const int* __restrict__ col_indices,
                                     int index_size, int words_per_row,
                                     unsigned int* __restrict__ closure,
                                     unsigned int* __restrict__ d_frontier) {
    int src = blockIdx.x * blockDim.x + threadIdx.x;
    if (src >= index_size) return;

    int row_start = row_offsets[src];
    int row_end   = row_offsets[src + 1];

    unsigned int* row = closure + static_cast<std::size_t>(src) * words_per_row;

    for (int e = row_start; e < row_end; ++e) {
        int dst = col_indices[e];
        row[dst / kBitsPerWord] |= 1u << (dst % kBitsPerWord);
    }

    if (row_start < row_end)
        atomicOr(&d_frontier[src >> 5], 1u << (src & 0x1F));
}

// Algorithm B iterative step with frontier bitset.
//
// Thread 0 checks whether any neighbour of u is in d_frontier (changed last
// iteration). If none are, u's closure cannot gain new bits and the block
// exits without touching device memory. Otherwise each thread ORs in the
// closure rows of active neighbours only, and marks u in d_next_frontier if
// anything changed.
//
// The frontier bitset is ~58 KB for SNOMED-CT and lives in L2 cache, so
// the per-neighbour bit checks are cheap and do not add visible latency.
__global__ void algoB_iter_kernel(const int* __restrict__ row_offsets,
                                  const int* __restrict__ col_indices,
                                  int index_size, int words_per_row,
                                  unsigned int* __restrict__ closure,
                                  const unsigned int* __restrict__ d_frontier,
                                  unsigned int* __restrict__ d_next_frontier,
                                  int* __restrict__ d_changed) {
    int u = blockIdx.x;
    if (u >= index_size) return;

    int row_start = row_offsets[u];
    int row_end   = row_offsets[u + 1];

    // Thread 0 scans the neighbour list for any active neighbour.
    __shared__ bool s_any_active;
    if (threadIdx.x == 0) {
        s_any_active = false;
        for (int e = row_start; e < row_end; ++e) {
            int v = col_indices[e];
            if (d_frontier[v >> 5] & (1u << (v & 0x1F))) {
                s_any_active = true;
                break;
            }
        }
    }
    __syncthreads();
    if (!s_any_active) return;

    // OR in closure[v,w] for active neighbours v only.
    bool local_changed = false;
    const std::size_t row_offset = static_cast<std::size_t>(u) * words_per_row;

    for (int w = threadIdx.x; w < words_per_row; w += blockDim.x) {
        unsigned int word = closure[row_offset + w];
        unsigned int acc  = word;

        for (int e = row_start; e < row_end; ++e) {
            int v = col_indices[e];
            if (!(d_frontier[v >> 5] & (1u << (v & 0x1F)))) continue;
            acc |= closure[static_cast<std::size_t>(v) * words_per_row + w];
        }

        if (acc != word) {
            closure[row_offset + w] = acc;
            local_changed = true;
        }
    }

    // Block-level reduction of the changed flag.
    __shared__ int block_changed;
    if (threadIdx.x == 0) block_changed = 0;
    __syncthreads();
    if (local_changed) atomicOr(&block_changed, 1);
    __syncthreads();

    if (threadIdx.x == 0 && block_changed) {
        atomicOr(d_changed, 1);
        atomicOr(&d_next_frontier[u >> 5], 1u << (u & 0x1F));
    }
}

/*************************************
 *  Host wrappers for algorithm B
 *************************************/

FrontierDevice run_algoB_initial(const CSRDevice &graph_dev,
                                 BitsetMatrixDevice &closure_dev) {
    int index_size = graph_dev.num_rows;
    int words      = (index_size + 31) / 32;

    FrontierDevice frontier;
    frontier.index_size = index_size;
    frontier.words      = words;

    check_cuda(cudaMalloc(&frontier.current, words * sizeof(unsigned int)),
               "cudaMalloc frontier.current");
    check_cuda(cudaMalloc(&frontier.next,    words * sizeof(unsigned int)),
               "cudaMalloc frontier.next");
    check_cuda(cudaMemset(frontier.current, 0, words * sizeof(unsigned int)),
               "cudaMemset frontier.current");
    check_cuda(cudaMemset(frontier.next,    0, words * sizeof(unsigned int)),
               "cudaMemset frontier.next");

    if (index_size == 0) return frontier;

    dim3 block(128);
    dim3 grid((index_size + block.x - 1) / block.x);

    algoB_initial_kernel<<<grid, block>>>(
        graph_dev.d_row_offsets, graph_dev.d_col_indices,
        index_size, static_cast<int>(closure_dev.words_per_row),
        closure_dev.data, frontier.current);
    check_cuda(cudaDeviceSynchronize(), "algoB_initial_kernel");

    return frontier;
}

bool run_algoB_iterations(const CSRDevice &graph_dev,
                          BitsetMatrixDevice &closure_dev,
                          FrontierDevice &frontier) {
    if (graph_dev.num_rows == 0) return false;

    check_cuda(cudaMemset(frontier.next, 0, frontier.words * sizeof(unsigned int)),
               "cudaMemset frontier.next");

    int *d_changed = nullptr;
    check_cuda(cudaMalloc(&d_changed, sizeof(int)), "cudaMalloc d_changed");
    int zero = 0;
    check_cuda(cudaMemcpy(d_changed, &zero, sizeof(int), cudaMemcpyHostToDevice),
               "cudaMemcpy d_changed init (B)");

    dim3 block(256);
    dim3 grid(graph_dev.num_rows);

    algoB_iter_kernel<<<grid, block>>>(
        graph_dev.d_row_offsets, graph_dev.d_col_indices,
        graph_dev.num_rows, static_cast<int>(closure_dev.words_per_row),
        closure_dev.data, frontier.current, frontier.next, d_changed);
    check_cuda(cudaDeviceSynchronize(), "algoB_iter_kernel");

    int h_changed = 0;
    check_cuda(cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost),
               "cudaMemcpy d_changed back (B)");
    cudaFree(d_changed);

    std::swap(frontier.current, frontier.next);

    return (h_changed != 0);
}

void free_frontier_device(FrontierDevice &frontier) {
    cudaFree(frontier.current);
    cudaFree(frontier.next);
    frontier.current    = nullptr;
    frontier.next       = nullptr;
    frontier.words      = 0;
    frontier.index_size = 0;
}
