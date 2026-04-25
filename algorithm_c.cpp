#include "serial.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

// ----- Parsing -----

static std::vector<std::string> split_tab(const std::string &line) {
    std::vector<std::string> fields;
    std::string field;
    std::stringstream ss(line);
    while (std::getline(ss, field, '\t')) {
        fields.push_back(field);
    }
    return fields;
}

static ColumnIndices parse_header(const std::string &header_line) {
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

static std::vector<Edge> load_isA_edges(const std::string &input_path) {
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
