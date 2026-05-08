#pragma once
#include "common.cuh"
#include <unordered_set>

using AdjMap = std::unordered_map<std::int64_t, std::unordered_set<std::int64_t>>; // (src_id -> set(dst_id))

AdjMap build_adjacency_from_edges(const std::vector<Edge> &edges);
AdjMap compute_a_ser_closure(const AdjMap &conn);
AdjMap compute_b_ser_closure(const AdjMap &conn);
ClosurePairs flatten_closure(const AdjMap &conn);
