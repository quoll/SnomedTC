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

        AdjMap next = prev;  // next starts as a full copy; all keys pre-exist
        bool any_new = false;

        std::vector<std::int64_t> sources;
        sources.reserve(prev.size());
        for (const auto &kv : prev) sources.push_back(kv.first);

        int n = static_cast<int>(sources.size());
        #pragma omp parallel for schedule(dynamic) reduction(||:any_new)
        for (int i = 0; i < n; ++i) {
            const auto s = sources[i];
            const auto &tset = prev.at(s);
            auto &s_next = next.at(s);  // safe: next = prev, so s is a pre-existing key
            for (const auto t : tset) {
                auto prev_it = prev.find(t);
                if (prev_it == prev.end()) continue;
                for (const auto u : prev_it->second) {
                    auto [_, inserted] = s_next.insert(u);
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
        // Pre-populate next with empty entries so parallel threads can call
        // next.at(s) without structural map modification during the loop.
        for (const auto &kv : frontier) next[kv.first];

        std::vector<std::int64_t> sources;
        sources.reserve(frontier.size());
        for (const auto &kv : frontier) sources.push_back(kv.first);

        bool any_new = false;

        int n = static_cast<int>(sources.size());
        #pragma omp parallel for schedule(dynamic) reduction(||:any_new)
        for (int i = 0; i < n; ++i) {
            const auto s = sources[i];
            const auto &tset = frontier.at(s);
            const auto &s_closure = closure.at(s);  // read-only; key always exists
            auto &s_next = next.at(s);               // pre-existing key; unique per thread
            for (const auto t : tset) {
                auto conn_it = conn.find(t);
                if (conn_it == conn.end()) continue;
                for (const auto u : conn_it->second) {
                    if (s_closure.find(u) == s_closure.end()) {
                        s_next.insert(u);
                        any_new = true;
                    }
                }
            }
        }

        // Merge next into closure (serial; next and closure are disjoint per key)
        for (auto &kv : next) {
            if (!kv.second.empty())
                closure[kv.first].insert(kv.second.begin(), kv.second.end());
        }

        auto ti1 = Clock::now();
        ++iter_count;
        std::cout << "Algorithm B-serial iteration " << iter_count << " took "
                  << std::chrono::duration<double, std::milli>(ti1 - ti0).count()
                  << " ms, changed=" << (any_new ? "true" : "false") << "\n";

        if (!any_new) break;

        // Drop empty entries before adopting next as the new frontier
        for (auto it = next.begin(); it != next.end(); ) {
            if (it->second.empty()) it = next.erase(it);
            else ++it;
        }

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
