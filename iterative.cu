#include "iterative.cuh"

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
