#include "resulttx.cuh"

#include <unordered_map>

/***********************************************
 *  Private helpers
 ***********************************************/

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

__global__ void external_count_kernel(const int* __restrict__ ext_row_offsets,
                                      const int* __restrict__ ext_dst_indices,
                                      int num_srcs,
                                      const unsigned int* __restrict__ closure_in,
                                      int index_size, int words_per_row,
                                      unsigned int* __restrict__ counts) {
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

            const std::size_t d_offset = static_cast<std::size_t>(d_idx) * words_per_row;
            acc |= closure_in[d_offset + w];
        }

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
__global__ void external_emit_kernel(const int* __restrict__ ext_row_offsets,
                                     const int* __restrict__ ext_dst_indices,
                                     int num_srcs,
                                     const unsigned int* __restrict__ closure_in,
                                     int index_size, int words_per_row,
                                     const std::int64_t* __restrict__ index_to_id,
                                     int* __restrict__ row_cursors,
                                     std::int64_t* __restrict__ out_dests) {
    int s = blockIdx.x;
    if (s >= num_srcs) return;

    int start = ext_row_offsets[s];
    int end   = ext_row_offsets[s + 1];

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

            const std::size_t d_offset = static_cast<std::size_t>(d_idx) * words_per_row;
            acc |= closure_in[d_offset + w];
        }

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

/***********************************************
 *  Conversion from bitset -> (src,dst) pairs
 ***********************************************/

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

// Connects all of the external (path-terminating) edges to the rest of the graph, on the GPU.
// Return the result to the host. This means:
// 1. converting the external edges to CSR form
// 2. uploading those external edges and the SNOMED-id mapping to the GPU
// 3. determining the row sizes needed for the final results
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

    // 3. Count how many pairs we will emit per external source
    unsigned int* d_counts = nullptr;
    check_cuda(cudaMalloc(&d_counts, num_srcs * sizeof(unsigned int)),
               "cudaMalloc d_counts");
    check_cuda(cudaMemset(d_counts, 0, num_srcs * sizeof(unsigned int)),
               "cudaMemset d_counts");

    dim3 block(256);
    dim3 grid(num_srcs);

    external_count_kernel<<<grid, block>>>(
        ext_csr_dev.d_row_offsets,
        ext_csr_dev.d_col_indices,
        num_srcs,
        closure_dev.data,
        index_size,
        words_per_row,
        d_counts
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

    // 6. Emit destination ids
    external_emit_kernel<<<grid, block>>>(
        ext_csr_dev.d_row_offsets,
        ext_csr_dev.d_col_indices,
        num_srcs,
        closure_dev.data,
        index_size,
        words_per_row,
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
