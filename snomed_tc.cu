#include "doubling.cuh"
#include "iterative.cuh"
#include "resulttx.cuh"
#include "serial.h"
#include "graph_util.h"

#include <fstream>
#include <string>

// Algorithm A
// run_algoA_initial and run_algoA_iterations are provided by doubling.cu

// Algorithm B
// run_algoB_initial and run_algoB_iterations are provided by iterative.cu

// Algorithm A-serial and Algorithm B-serial
// build_adjacency_from_edges, compute_a_ser_closure, compute_b_ser_closure, flatten_closure
// are provided by serial.cpp


int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "Usage: snomed_tc <input_snomed_file> <output_file>\n";
        return 1;
    }

    const std::string input_path  = argv[1];
    const std::string output_path = argv[2];

    try {
        // 0. Initialise GPU (not included in algorithm timings)
        init_gpu();

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

        // 8. Retrieve all closure pairs (internal + external)
        t0 = Clock::now();
        ClosurePairs closureA_pairs = retrieve_results(*in, mapping, external_edges);
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
        ClosurePairs closureB_pairs = retrieve_results(closureB_dev, mapping, external_edges);
        t1 = Clock::now();
        auto tB1 = t1;
        std::cout << "Algorithm B convert + external closure: "
                  << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n";
        std::cout << "Algorithm B closure total pairs (including external): "
                  << closureB_pairs.size() << "\n";
        std::cout << "Algorithm B total time: "
                  << (common_time + std::chrono::duration<double, std::milli>(tB1 - tB0).count()) << " ms\n";

        // 10. Algorithm A-serial: CPU doubling
        t0 = Clock::now();
        auto tASer0 = t0;
        AdjMap connASer = build_adjacency_from_edges(edges);
        t1 = Clock::now();
        std::cout << "Algorithm A-serial build_adjacency_from_edges: "
                  << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n";
        t0 = Clock::now();
        AdjMap closureASer = compute_a_ser_closure(connASer);
        t1 = Clock::now();
        std::cout << "Algorithm A-serial compute_a_ser_closure: "
                  << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n";
        t0 = Clock::now();
        ClosurePairs closureASer_pairs = flatten_closure(closureASer);
        t1 = Clock::now();
        auto tASer1 = t1;
        std::cout << "Algorithm A-serial flatten_closure to pairs: "
                  << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n";
        std::cout << "Algorithm A-serial closure size: "
                  << closureASer_pairs.size() << "\n";
        std::cout << "Algorithm A-serial total time: "
                  << std::chrono::duration<double, std::milli>(tASer1 - tASer0).count() << " ms\n";

        // 11. Algorithm B-serial: CPU BFS
        t0 = Clock::now();
        auto tBSer0 = t0;
        AdjMap connBSer = build_adjacency_from_edges(edges);
        t1 = Clock::now();
        std::cout << "Algorithm B-serial build_adjacency_from_edges: "
                  << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n";
        t0 = Clock::now();
        AdjMap closureBSer = compute_b_ser_closure(connBSer);
        t1 = Clock::now();
        std::cout << "Algorithm B-serial compute_b_ser_closure: "
                  << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n";
        t0 = Clock::now();
        ClosurePairs closureBSer_pairs = flatten_closure(closureBSer);
        t1 = Clock::now();
        auto tBSer1 = t1;
        std::cout << "Algorithm B-serial flatten_closure to pairs: "
                  << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n";
        std::cout << "Algorithm B-serial closure size: "
                  << closureBSer_pairs.size() << "\n";
        std::cout << "Algorithm B-serial total time: "
                  << std::chrono::duration<double, std::milli>(tBSer1 - tBSer0).count() << " ms\n";

        // 12. Compare all four algorithms (sort + compare)
        ClosurePairs sortedA    = closureA_pairs;
        ClosurePairs sortedB    = closureB_pairs;
        ClosurePairs sortedASer = closureASer_pairs;
        ClosurePairs sortedBSer = closureBSer_pairs;
        std::sort(sortedA.begin(),    sortedA.end());
        std::sort(sortedB.begin(),    sortedB.end());
        std::sort(sortedASer.begin(), sortedASer.end());
        std::sort(sortedBSer.begin(), sortedBSer.end());

        std::cout << "Algorithm A    vs B        results equal? "
                  << ((sortedA == sortedB)    ? "YES" : "NO") << "\n";
        std::cout << "Algorithm A    vs A-serial results equal? "
                  << ((sortedA == sortedASer) ? "YES" : "NO") << "\n";
        std::cout << "Algorithm A    vs B-serial results equal? "
                  << ((sortedA == sortedBSer) ? "YES" : "NO") << "\n";

        // 13. Cleanup device resources
        free_csr_device(graph_dev);
        free_bitset_matrix_device(closureA_in);
        free_bitset_matrix_device(closureA_out);
        free_bitset_matrix_device(closureB_dev);

        // 14. Write results
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
