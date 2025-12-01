#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <chrono>

// SNOMED 'isA' relationship typeId.
constexpr const char* IS_A_TYPE_ID = "116680003";

struct ColumnIndices {
    int source_idx = -1;
    int dest_idx   = -1;
    int type_idx   = -1;
};

struct Edge {
    std::int64_t src; // sourceId
    std::int64_t dst; // destinationId
};

struct CSRGraph {
    int n = 0;
    std::vector<int> row_offsets;  // size n+1
    std::vector<int> col_indices;  // size = #internal_edges
};

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
        }
    }

    if (idx.source_idx < 0 || idx.dest_idx < 0 || idx.type_idx < 0) {
        throw std::runtime_error("Failed to locate sourceId/destinationId/typeId in header");
    }

    return idx;
}

// Load all isA edges as (src, dst) pairs.
std::vector<Edge> load_isA_edges(const std::string &input_path) {
    std::ifstream in(input_path);
    if (!in) {
        throw std::runtime_error("Failed to open input file: " + input_path);
    }

    std::string line;
    if (!std::getline(in, line)) {
        throw std::runtime_error("Input file is empty: " + input_path);
    }

    ColumnIndices idx = parse_header(line);
    int max_idx = std::max({idx.source_idx, idx.dest_idx, idx.type_idx});

    std::vector<Edge> edges;
    std::size_t total_rows = 0;

    while (std::getline(in, line)) {
        if (line.empty()) continue;
        ++total_rows;

        auto fields = split_tab(line);
        if (max_idx >= static_cast<int>(fields.size())) {
            continue;  // malformed row; skip
        }

        if (fields[idx.type_idx] == IS_A_TYPE_ID) {
            std::int64_t src = std::stoll(fields[idx.source_idx]);
            std::int64_t dst = std::stoll(fields[idx.dest_idx]);
            edges.push_back(Edge{src, dst});
        }
    }

    std::cerr << "Total data rows (excluding header): " << total_rows << "\n";
    std::cerr << "Loaded isA edges (src,dst pairs): " << edges.size() << "\n";

    return edges;
}

// Build canonical dest_ids via sort+unique.
std::vector<std::int64_t> build_dest_ids_sort_unique(const std::vector<Edge> &edges) {
    std::vector<std::int64_t> dest_ids;
    dest_ids.reserve(edges.size());
    for (const auto &e : edges) {
        dest_ids.push_back(e.dst);
    }
    std::sort(dest_ids.begin(), dest_ids.end());
    dest_ids.erase(std::unique(dest_ids.begin(), dest_ids.end()), dest_ids.end());
    return dest_ids;
}

// Binary-search mapping: destId -> t_idx
inline int dest_id_to_tidx_binary(const std::vector<std::int64_t> &dest_ids,
                                  std::int64_t id) {
    auto it = std::lower_bound(dest_ids.begin(), dest_ids.end(), id);
    if (it == dest_ids.end() || *it != id) return -1;
    return static_cast<int>(it - dest_ids.begin());
}

// Hashmap mapping: destId -> t_idx
using IdToIdxMap = std::unordered_map<std::int64_t, int>;

IdToIdxMap build_id_to_idx_map(const std::vector<std::int64_t> &dest_ids) {
    IdToIdxMap map;
    map.reserve(dest_ids.size() * 2);
    for (std::size_t i = 0; i < dest_ids.size(); ++i) {
        map.emplace(dest_ids[i], static_cast<int>(i));
    }
    return map;
}

inline int dest_id_to_tidx_hash(const IdToIdxMap &map, std::int64_t id) {
    auto it = map.find(id);
    if (it == map.end()) return -1;
    return it->second;
}

// Generic CSR builder using a mapping functor destId -> t_idx.
template <typename Mapper>
CSRGraph build_csr_internal_with_mapper(
    const std::vector<Edge> &edges,
    const std::vector<std::int64_t> &dest_ids,
    Mapper &&mapper,
    std::size_t &num_internal_edges,
    std::size_t &num_external_edges)
{
    const int Nt = static_cast<int>(dest_ids.size());
    std::vector<std::pair<int,int>> internal_edges;
    internal_edges.reserve(edges.size());

    num_external_edges = 0;

    for (const auto &e : edges) {
        int t_src = mapper(e.src);
        int t_dst = mapper(e.dst);  // should be >=0 if dst ∈ dest_ids

        if (t_dst < 0) {
            ++num_external_edges;
            continue;
        }

        if (t_src >= 0) {
            internal_edges.emplace_back(t_src, t_dst);
        } else {
            ++num_external_edges;
        }
    }

    num_internal_edges = internal_edges.size();

    CSRGraph g;
    g.n = Nt;
    g.row_offsets.assign(Nt + 1, 0);
    g.col_indices.resize(num_internal_edges);

    // Count out-degree
    for (const auto &p : internal_edges) {
        int t_src = p.first;
        ++g.row_offsets[t_src + 1];
    }

    // Prefix sum
    for (int i = 0; i < Nt; ++i) {
        g.row_offsets[i + 1] += g.row_offsets[i];
    }

    // Cursor for filling
    std::vector<int> cursor = g.row_offsets;

    for (const auto &p : internal_edges) {
        int t_src = p.first;
        int t_dst = p.second;
        int pos = cursor[t_src]++;
        g.col_indices[pos] = t_dst;
    }

    // Sort neighbors per row for determinism
    for (int u = 0; u < Nt; ++u) {
        int begin = g.row_offsets[u];
        int end   = g.row_offsets[u + 1];
        std::sort(g.col_indices.begin() + begin, g.col_indices.begin() + end);
    }

    return g;
}

// Timing helper
template <typename F>
auto time_block(const std::string &label, F &&f) {
    using namespace std::chrono;
    auto t0 = high_resolution_clock::now();
    auto result = f();
    auto t1 = high_resolution_clock::now();
    double ms = duration_cast<microseconds>(t1 - t0).count() / 1000.0;
    std::cout << label << ": " << ms << " ms\n";
    return std::make_pair(std::move(result), ms);
}

bool csr_equal(const CSRGraph &a, const CSRGraph &b) {
    return a.n == b.n &&
           a.row_offsets == b.row_offsets &&
           a.col_indices == b.col_indices;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <input_snomed_file>\n";
        return 1;
    }

    const std::string input_path = argv[1];

    try {
        // 1. Load edges
        auto edges = load_isA_edges(input_path);

        // 2. Build canonical dest_ids once (sort+unique)
        auto [dest_ids, t_build_ids] =
            time_block("Build dest_ids (sort+unique)", [&]() {
                return build_dest_ids_sort_unique(edges);
            });
        std::cout << "Unique destinationIds: " << dest_ids.size() << "\n";

        // 3. Build CSR using binary-search mapping
        std::size_t internal_bin = 0, external_bin = 0;
        auto [csr_bin, t_csr_bin] =
            time_block("Build CSR using binary-search mapping", [&]() {
                return build_csr_internal_with_mapper(
                    edges,
                    dest_ids,
                    [&](std::int64_t id) { return dest_id_to_tidx_binary(dest_ids, id); },
                    internal_bin,
                    external_bin
                );
            });
        std::cout << "  [binary] internal_edges: " << internal_bin
                  << ", external_edges: " << external_bin << "\n";

        // 4. Build hashmap from dest_ids
        auto [id_map, t_build_map] =
            time_block("Build destId->idx hashmap", [&]() {
                return build_id_to_idx_map(dest_ids);
            });

        // 5. Build CSR using hashmap mapping (with pre-built map)
        std::size_t internal_hash = 0, external_hash = 0;
        auto [csr_hash, t_csr_hash] =
            time_block("Build CSR using hashmap mapping", [&]() {
                return build_csr_internal_with_mapper(
                    edges,
                    dest_ids,
                    [&](std::int64_t id) { return dest_id_to_tidx_hash(id_map, id); },
                    internal_hash,
                    external_hash
                );
            });
        std::cout << "  [hash]   internal_edges: " << internal_hash
                  << ", external_edges: " << external_hash << "\n";

        // 6. Combined times for fair comparison
        double total_binary = t_build_ids + t_csr_bin;
        double total_hash   = t_build_ids + t_build_map + t_csr_hash;
        std::cout << "Total (sort+unique + CSR-binary): " << total_binary << " ms\n";
        std::cout << "Total (sort+unique + build-map + CSR-hash): "
                  << total_hash << " ms\n";

        // 7. Verify CSR equality
        if (!csr_equal(csr_bin, csr_hash)) {
            std::cerr << "ERROR: CSR graphs differ between binary and hashmap mapping!\n";
        } else {
            std::cout << "CSR graphs are identical (same indexing, same edges).\n";
        }

    } catch (const std::exception &ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}