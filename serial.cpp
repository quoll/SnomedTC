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

/****************************************************************
 *  Algorithm A-serial: doubling transitive closure
 *  conn is read-only. Each iteration copies prev into next and
 *  adds edges reachable by composing prev with itself, doubling
 *  the covered path length. The old prev is freed each iteration.
 ****************************************************************/
AdjMap compute_a_ser_closure(const AdjMap &conn) {
    AdjMap prev = conn;

    int iter_count = 0;

    while (true) {
        auto ti0 = Clock::now();

        AdjMap next = prev;  // next starts as a full copy; old prev freed below
        bool any_new = false;

        for (const auto &[s, tset] : prev) {
            for (const auto t : tset) {
                auto prev_it = prev.find(t);
                if (prev_it == prev.end()) continue;

                for (const auto u : prev_it->second) {
                    auto [_, inserted] = next[s].insert(u);
                    if (inserted) any_new = true;
                }
            }
        }

        auto ti1 = Clock::now();
        ++iter_count;
        std::cout << "Algorithm A-serial iteration " << iter_count << " took "
                  << std::chrono::duration<double, std::milli>(ti1 - ti0).count()
                  << " ms, changed=" << (any_new ? "true" : "false") << "\n";

        if (!any_new) break;

        prev = std::move(next);  // free old prev, adopt new one
    }

    std::cout << "Algorithm A-serial iterations until fixed point: " << iter_count << "\n";

    return prev;
}

/****************************************************************
 *  Algorithm B-serial: BFS transitive closure
 *  conn is read-only throughout. Each iteration extends known
 *  paths by one hop through conn. The frontier holds only the
 *  edges discovered in the previous iteration and is freed when
 *  replaced by the next frontier.
 ****************************************************************/
AdjMap compute_b_ser_closure(const AdjMap &conn) {
    AdjMap closure = conn;   // accumulates all edges found so far
    AdjMap frontier = conn;  // edges discovered in the last iteration

    int iter_count = 0;

    while (true) {
        auto ti0 = Clock::now();

        AdjMap next;

        for (const auto &[s, tset] : frontier) {
            auto &s_closure = closure[s];
            for (const auto t : tset) {
                auto conn_it = conn.find(t);
                if (conn_it == conn.end()) continue;

                for (const auto u : conn_it->second) {
                    if (s_closure.find(u) == s_closure.end()) {
                        next[s].insert(u);
                    }
                }
            }
        }

        // Merge next into closure
        for (auto &[s, uset] : next) {
            closure[s].insert(uset.begin(), uset.end());
        }

        auto ti1 = Clock::now();
        ++iter_count;
        bool any_new = !next.empty();
        std::cout << "Algorithm B-serial iteration " << iter_count << " took "
                  << std::chrono::duration<double, std::milli>(ti1 - ti0).count()
                  << " ms, changed=" << (any_new ? "true" : "false") << "\n";

        if (!any_new) break;

        frontier = std::move(next);  // free old frontier, adopt new one
    }

    std::cout << "Algorithm B-serial iterations until fixed point: " << iter_count << "\n";

    return closure;
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
