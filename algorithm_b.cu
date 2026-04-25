#include "iterative.cuh"
#include "resulttx.cuh"
#include "graph_util.h"

#include <fstream>
#include <string>

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "Usage: algorithm_b <input_snomed_file> <output_file>\n";
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

        // 5. Algorithm B: bitset matrix, initial + iterations
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

        // 6. Convert internal closure to pairs, then compute external closure on GPU
        t0 = Clock::now();
        ClosurePairs internal_pairs = convert_internal_closure_to_pairs(closureB_dev, mapping);
        ClosurePairs external_pairs = compute_external_closure_gpu(closureB_dev, mapping, external_edges);
        ClosurePairs closureB_pairs;
        closureB_pairs.reserve(internal_pairs.size() + external_pairs.size());
        closureB_pairs.insert(closureB_pairs.end(), internal_pairs.begin(), internal_pairs.end());
        closureB_pairs.insert(closureB_pairs.end(), external_pairs.begin(), external_pairs.end());
        t1 = Clock::now();
        auto tB1 = t1;
        std::cout << "Algorithm B convert + external closure: "
                  << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n";
        std::cout << "Algorithm B closure total pairs (including external): "
                  << closureB_pairs.size() << "\n";
        std::cout << "Algorithm B total time: "
                  << (common_time + std::chrono::duration<double, std::milli>(tB1 - tB0).count()) << " ms\n";

        // 7. Cleanup device resources
        free_csr_device(graph_dev);
        free_bitset_matrix_device(closureB_dev);

        // 8. Write results
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
