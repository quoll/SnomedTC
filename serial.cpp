#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// Note: comments and printed results includes unicode characters: ⋃∈∉

// SNOMED 'isA' relationship typeId.
constexpr const char* kIsATypeId = "116680003";

struct ColumnIndices {
    int source_idx = -1;
    int dest_idx   = -1;
    int type_idx   = -1;
    int active_idx = -1;
};

using Edge = std::pair<std::int64_t, std::int64_t>;  // (src_id, dst_id)
using ClosurePairs = std::vector<Edge>;

using AdjMap = std::unordered_map<std::int64_t, std::unordered_set<std::int64_t>>; // (src_id -> set(dst_id))

// small timing helper
using Clock = std::chrono::steady_clock;

// ----- Parsing -----

std::vector<std::string> split_tab(const std::string &line) {
    std::vector<std::string> fields;
    std::string field;
    std::stringstream ss(line);
    while (std::getline(ss, field, '\t')) {
        fields.push_back(field);
    }
    return fields;
}

ColumnIndices parse_header(const std::string &header_line) {
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

std::vector<Edge> load_isA_edges(const std::string &input_path) {
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
