#include "serial.h"
#include "graph_util.h"

#include <fstream>
#include <stdexcept>
#include <string>

/*****************
 *  Main program
 *****************/

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "Usage: algorithm_c <input_snomed_file> <output_file>\n";
        return 1;
    }

    const std::string input_path  = argv[1];
    const std::string output_path = argv[2];

    try {
        // 1. Load data
        auto t0 = Clock::now();
        auto edges = load_isA_edges(input_path);
        auto t1 = Clock::now();
        auto step0 = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "Step load_isA_edges: " << step0 << " ms\n";

        // Algorithm C: host serialized form of Algorithm A
        t0 = Clock::now();
        AdjMap conn0 = build_adjacency_from_edges(edges);
        t1 = Clock::now();
        auto step1 = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "Algorithm C build_adjacency_from_edges: " << step1 << " ms\n";

        t0 = Clock::now();
        auto conn_tc = compute_transitive_closure_serial(std::move(conn0));
        t1 = Clock::now();
        auto step2 = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "Algorithm C compute_transitive_closure_serial: " << step2 << " ms\n";

        t0 = Clock::now();
        ClosurePairs closureC_pairs = flatten_closure(conn_tc);
        t1 = Clock::now();
        auto step3 = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "Algorithm C flatten_closure to pairs: " << step3 << " ms\n";
        std::cout << "Algorithm C map-of-sets closure size: " << closureC_pairs.size() << "\n";
        std::cout << "Algorithm C total time: " << (step1 + step2 + step3) << " ms\n";

        // Write results
        t0 = Clock::now();
        std::ofstream file_out(output_path);
        for (const auto &p : closureC_pairs) {
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
