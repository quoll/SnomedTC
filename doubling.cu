#include "doubling.cuh"

#include <stdexcept>

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
