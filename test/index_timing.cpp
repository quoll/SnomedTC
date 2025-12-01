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

#include <omp.h>  // OpenMP

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

// Shared: load all isA edges as (src, dst) pairs.
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

/***************
 *  METHODS
 **************/

// Method 1: serial sort+unique
std::vector<std::int64_t> dest_ids_sort_unique_serial(const std::vector<Edge> &edges) {
    std::vector<std::int64_t> v;
    v.reserve(edges.size());
    for (const auto &e : edges) v.push_back(e.dst);

    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    return v;
}

// Method 2: OpenMP-parallel chunked sort + merge
std::vector<std::int64_t> dest_ids_sort_unique_omp(const std::vector<Edge> &edges) {
    std::vector<std::int64_t> v(edges.size());
    for (std::size_t i = 0; i < edges.size(); ++i) v[i] = edges[i].dst;

    const int n = static_cast<int>(v.size());
    if (n == 0) return v;

    int num_threads = omp_get_max_threads();
    int chunk_size  = (n + num_threads - 1) / num_threads;

    // Sort each chunk in parallel
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < num_threads; ++t) {
        int begin = t * chunk_size;
        int end   = std::min(begin + chunk_size, n);
        if (begin < end) {
            std::sort(v.begin() + begin, v.begin() + end);
        }
    }

    // Serial merging of sorted chunks (multi-pass)
    int current = chunk_size;
    while (current < n) {
        for (int begin = 0; begin < n; begin += 2 * current) {
            int mid = std::min(begin + current, n);
            int end = std::min(begin + 2 * current, n);
            if (mid < end) {
                std::inplace_merge(v.begin() + begin,
                                   v.begin() + mid,
                                   v.begin() + end);
            }
        }
        current *= 2;
    }

    v.erase(std::unique(v.begin(), v.end()), v.end());
    return v;
}

// Method 3: Hashmap
std::vector<std::int64_t> dest_ids_hashmap(const std::vector<Edge> &edges) {
    std::unordered_map<std::int64_t, std::uint32_t> map;
    map.reserve(edges.size() * 2);

    uint32_t next_idx = 0;
    for (const auto &e : edges) {
        auto it = map.find(e.dst);
        if (it == map.end()) {
            map.emplace(e.dst, next_idx++);
        }
    }

    std::vector<std::int64_t> ids(map.size());
    for (const auto &kv : map) {
        ids[kv.second] = kv.first;
    }

    std::sort(ids.begin(), ids.end());
    return ids;
}

template <typename F>
std::pair<std::vector<std::int64_t>, double> time_method(const std::string &name, F &&f) {
    using namespace std::chrono;
    auto t0 = high_resolution_clock::now();
    auto result = f();
    auto t1 = high_resolution_clock::now();
    double ms =
        duration_cast<microseconds>(t1 - t0).count() / 1000.0;
    std::cout << name << ": " << result.size() << " unique dest_ids in "
              << ms << " ms\n";
    return {std::move(result), ms};
}

bool same_set(const std::vector<std::int64_t> &a,
              const std::vector<std::int64_t> &b) {
    return a.size() == b.size() &&
           std::equal(a.begin(), a.end(), b.begin());
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <input_snomed_file>\n";
        return 1;
    }

    try {
        auto edges = load_isA_edges(argv[1]);

        auto [v1, t1] = time_method("Serial sort+unique",
            [&]() { return dest_ids_sort_unique_serial(edges); });

        auto [v2, t2] = time_method("OMP parallel chunked sort+merge+unique",
            [&]() { return dest_ids_sort_unique_omp(edges); });

        auto [v3, t3] = time_method("Hashmap (unordered_map) + sort",
            [&]() { return dest_ids_hashmap(edges); });

        bool ok = same_set(v1, v2) && same_set(v1, v3);

        if (ok)
            std::cout << "All methods produced identical dest_id sets.\n";
        else
            std::cout << "WARNING: mismatch between methods.\n";

    } catch (std::exception &e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }

    return 0;
}