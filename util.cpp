#include "util.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

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

// ----- DestMapping -----

DestMapping build_dest_mapping(const std::vector<Edge> &edges) {
    DestMapping mapping;
    mapping.id_to_index.reserve(edges.size() * 2);

    int next_index = 0;
    for (const auto &e : edges) {
        auto [it, inserted] = mapping.id_to_index.try_emplace(e.second, next_index);
        if (inserted) {
            ++next_index;
        }
    }

    mapping.index_to_id.resize(next_index);
    for (const auto &kv : mapping.id_to_index) {
        mapping.index_to_id[kv.second] = kv.first;
    }

    return mapping;
}

// ----- CSR + external edges -----

CSRGraph build_csr_internal(const std::vector<Edge> &edges, const DestMapping &mapping,
                            std::size_t &num_internal_edges, std::size_t &num_external_edges,
                            std::vector<Edge> &external_edges_out) {
    const auto &index_to_id = mapping.index_to_id;
    const auto &id_to_index = mapping.id_to_index;
    const int index_size = static_cast<int>(index_to_id.size());

    std::vector<std::pair<int,int>> internal_edges;
    internal_edges.reserve(edges.size());

    num_external_edges = 0;
    external_edges_out.clear();

    auto map_id = [&](std::int64_t id) -> int {
        auto it = id_to_index.find(id);
        return (it == id_to_index.end()) ? -1 : it->second;
    };

    for (const auto &e : edges) {
        int src_index = map_id(e.first);
        int dst_index = map_id(e.second);

        if (dst_index < 0) {
            // In principle this shouldn't happen, but keep stats honest.
            ++num_external_edges;
            continue;
        }

        if (src_index >= 0) {
            // Internal edge: both ends in T.
            internal_edges.emplace_back(src_index, dst_index);
        } else {
            // External source, internal destination.
            ++num_external_edges;
            external_edges_out.push_back(e);
        }
    }

    num_internal_edges = internal_edges.size();

    CSRGraph graph;
    graph.num_rows = index_size;
    graph.row_offsets.assign(index_size + 1, 0);
    graph.col_indices.resize(num_internal_edges);

    // Count out-degree
    for (const auto &p : internal_edges) {
        ++graph.row_offsets[p.first + 1];
    }

    // Prefix sum
    for (int i = 0; i < index_size; ++i) {
        graph.row_offsets[i + 1] += graph.row_offsets[i];
    }

    // Fill adjacency
    std::vector<int> cursor = graph.row_offsets;
    for (const auto &p : internal_edges) {
        int src_index = p.first;
        int dst_index = p.second;
        int pos = cursor[src_index]++;
        graph.col_indices[pos] = dst_index;
    }

    // Sort neighbors in each row for determinism
    for (int u = 0; u < index_size; ++u) {
        int begin = graph.row_offsets[u];
        int end   = graph.row_offsets[u + 1];
        std::sort(graph.col_indices.begin() + begin,
                  graph.col_indices.begin() + end);
    }

    return graph;
}
