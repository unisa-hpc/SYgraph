#include <sycl/sycl.hpp>

#include <memory>

#include <sygraph/frontier/frontier.hpp>
#include <sygraph/graph/graph.hpp>
#include <sygraph/operators/advance/advance.hpp>
#include <sygraph/operators/filter/filter.hpp>
#include <sygraph/operators/for/for.hpp>
#ifdef ENABLE_PROFILING
#include <sygraph/utils/profiler.hpp>
#endif
#include <sygraph/sync/atomics.hpp>


namespace sygraph {
inline namespace v0 {
namespace algorithms {
namespace detail {

template<typename GraphType>
struct SSSPInstance {
  using vertex_t = typename GraphType::vertex_t;
  using edge_t = typename GraphType::edge_t;
  using weight_t = typename GraphType::weight_t;

  GraphType& G;
  vertex_t source;
  weight_t* distances;
  vertex_t* parents;
  int* visited;

  SSSPInstance(GraphType& G, vertex_t source) : G(G), source(source) {
    sycl::queue& queue = G.getQueue();
    size_t size = G.getVertexCount();

    distances = memory::detail::memoryAlloc<weight_t, memory::space::shared>(size, queue);
    queue.fill(distances, static_cast<weight_t>(size + 1), size).wait();
    queue.fill(distances + source, static_cast<weight_t>(0), 1).wait();

    parents = memory::detail::memoryAlloc<vertex_t, memory::space::shared>(size, queue);
    queue.fill(parents, static_cast<vertex_t>(-1), size).wait();

    visited = memory::detail::memoryAlloc<int, memory::space::shared>(size, queue);
    queue.fill(visited, -1, size).wait();
  }

  size_t getVisitedVertices() const {
    size_t vertex_count = G.getVertexCount();
    size_t visited_nodes = 0;
    for (size_t i = 0; i < G.getVertexCount(); i++) {
      if (distances[i] != static_cast<edge_t>(vertex_count + 1)) { visited_nodes++; }
    }
    return visited_nodes;
  }

  size_t getVisitedEdges() const {
    size_t vertex_count = G.getVertexCount();
    size_t visited_edges = 0;
    for (size_t i = 0; i < G.getVertexCount(); i++) {
      if (distances[i] != static_cast<edge_t>(vertex_count + 1)) { visited_edges += G.getDegree(i); }
    }
    return visited_edges;
  }

  ~SSSPInstance() {
    sycl::queue& queue = G.getQueue();
    sycl::free(distances, queue);
    sycl::free(parents, queue);
    sycl::free(visited, queue);
  }
};
} // namespace detail


template<typename GraphType>
class SSSP {
  using vertex_t = typename GraphType::vertex_t;
  using edge_t = typename GraphType::edge_t;
  using weight_t = typename GraphType::weight_t;

public:
  SSSP(GraphType& g) : _g(g){};


  void init(vertex_t& source) {
    _instance = std::make_unique<detail::SSSPInstance<GraphType>>(_g, source);
    _instance->distances[source] = 0;
  }


  void reset() { _instance.reset(); }


  template<bool EnableProfiling = false>
  void run() {
    if (!_instance) { throw std::runtime_error("SSSP instance not initialized"); }

    auto& G = _instance->G;
    auto& source = _instance->source;
    auto& distances = _instance->distances;
    auto& parents = _instance->parents;
    auto& visited = _instance->visited;

    sycl::queue& queue = G.getQueue();

    using load_balance_t = sygraph::operators::load_balancer;
    using direction_t = sygraph::operators::direction;
    using frontier_view_t = sygraph::frontier::frontier_view;
    using frontier_impl_t = sygraph::frontier::frontier_type;

    auto in_frontier = sygraph::frontier::makeFrontier<frontier_view_t::vertex, frontier_impl_t::hierachic_bitmap>(queue, G);
    auto out_frontier = sygraph::frontier::makeFrontier<frontier_view_t::vertex, frontier_impl_t::hierachic_bitmap>(queue, G);

    size_t size = G.getVertexCount();

    int iter = 0;
    in_frontier.insert(source);

    while (!in_frontier.empty()) {
      auto e1 = sygraph::operators::advance::vertex<load_balance_t::workgroup_mapped>(
          G, in_frontier, out_frontier, [=](auto src, auto dst, auto edge, auto weight) -> bool {
            weight_t source_distance = sygraph::sync::load(&distances[src]);
            weight_t distance_to_neighbor = source_distance + weight;

            // Check if the destination node has been claimed as someone's child
            weight_t recover_distance = sygraph::sync::load(&distances[dst]);
            recover_distance = sygraph::sync::min(&(distances[dst]), &distance_to_neighbor);

            return (distance_to_neighbor < recover_distance);
          });
      e1.wait();

      auto e2 = sygraph::operators::filter::external(G, out_frontier, in_frontier, [=](auto vertex) -> bool {
        if (visited[vertex] == iter) return false;
        visited[vertex] = iter;
        return true;
      });
      e2.wait();

#ifdef ENABLE_PROFILING
      sygraph::Profiler::addEvent(e1, "advance");
      sygraph::Profiler::addEvent(e2, "filter");
#endif

      out_frontier.clear();
      iter++;
    }
#ifdef ENABLE_PROFILING
    sygraph::Profiler::addVisitedEdges(_instance->getVisitedEdges());
#endif
  }

  weight_t getDistance(size_t vertex) const { return _instance->distances[vertex]; }

  vertex_t getParents(size_t vertex) const {
    throw std::runtime_error("Not implemented");
    return _instance->parents[vertex];
  }

private:
  GraphType& _g;
  std::unique_ptr<detail::SSSPInstance<GraphType>> _instance;
};

} // namespace algorithms
} // namespace v0
} // namespace sygraph