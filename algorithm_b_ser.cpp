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
        std::cerr << "Usage: algorithm_b_ser <input_snomed_file> <output_file>\n";
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
                  << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n";

        // 2. Build adjacency map
        t0 = Clock::now();
        auto tBSer0 = t0;
        AdjMap conn = build_adjacency_from_edges(edges);
        t1 = Clock::now();
        std::cout << "Algorithm B-serial build_adjacency_from_edges: "
                  << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n";

        // 3. Compute transitive closure
        t0 = Clock::now();
        AdjMap closure = compute_b_ser_closure(conn);
        t1 = Clock::now();
        std::cout << "Algorithm B-serial compute_b_ser_closure: "
                  << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n";

        // 4. Flatten to pairs
        t0 = Clock::now();
        ClosurePairs pairs = flatten_closure(closure);
        t1 = Clock::now();
        auto tBSer1 = t1;
        std::cout << "Algorithm B-serial flatten_closure to pairs: "
                  << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n";
        std::cout << "Algorithm B-serial closure size: " << pairs.size() << "\n";
        std::cout << "Algorithm B-serial total time: "
                  << std::chrono::duration<double, std::milli>(tBSer1 - tBSer0).count() << " ms\n";

        // 5. Write results
        t0 = Clock::now();
        std::ofstream file_out(output_path);
        for (const auto &p : pairs) {
            file_out << p.first << '\t' << p.second << '\n';
        }
        t1 = Clock::now();
        std::cout << "Writing output file: "
                  << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n";

    } catch (const std::exception &ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
