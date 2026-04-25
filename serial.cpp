#include "serial.h"

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
AdjMap compute_transitive_closure_serial(AdjMap conn, int max_iterations) {
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
