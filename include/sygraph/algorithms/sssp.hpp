#include <sycl/sycl.hpp>

#include <memory>

#include <sygraph/graph/graph.hpp>
#include <sygraph/frontier/frontier.hpp>
#include <sygraph/operators/advance/advance.hpp>
#include <sygraph/operators/for/for.hpp>
#include <sygraph/operators/filter/filter.hpp>
#ifdef ENABLE_PROFILING
#include <sygraph/utils/profiler.hpp>
#endif
#include <sygraph/sync/atomics.hpp>


namespace sygraph {
inline namespace v0 {
namespace algorithms {
namespace detail {

template <typename GraphType>
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
    sycl::queue& queue = G.get_queue();
    size_t size = G.get_vertex_count();

    distances = sycl::malloc_shared<edge_t>(size, queue);
    queue.fill(distances, static_cast<weight_t>(size + 1), size).wait();
    distances[source] = static_cast<edge_t>(0);

    parents = sycl::malloc_shared<vertex_t>(size, queue);
    queue.fill(parents, static_cast<vertex_t>(-1), size).wait();

    visited = sycl::malloc_shared<int>(size, queue);
    queue.fill(visited, -1, size).wait();
  }

  const size_t get_visited_vertices() const {
    size_t vertex_count = G.get_vertex_count();
    size_t visited_nodes = 0;
    for (size_t i = 0; i < G.get_vertex_count(); i++) {
      if (distances[i] != static_cast<edge_t>(vertex_count + 1)) {
        visited_nodes++;
      }
    }
    return visited_nodes;
  }

  const size_t get_visited_edges() const {
    size_t vertex_count = G.get_vertex_count();
    size_t visited_edges = 0;
    for (size_t i = 0; i < G.get_vertex_count(); i++) {
      if (distances[i] != static_cast<edge_t>(vertex_count + 1)) {
        visited_edges += G.get_degree(i);
      }
    }
    return visited_edges;
  }

  ~SSSPInstance() {
    sycl::queue& queue = G.get_queue();
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
  SSSP(GraphType& g) : _g(g) {};

  
  void init(vertex_t& source) {
    _instance = std::make_unique<detail::SSSPInstance<GraphType>>(_g, source);
    _instance->distances[source] = 0;
  }

  
  void reset() {
    _instance.reset();
  }

  
  template <bool enable_profiling = false> 
  void run() {
    if (!_instance) {
      throw std::runtime_error("SSSP instance not initialized");
    }

    auto& G = _instance->G;
    auto& source = _instance->source;
    auto& distances = _instance->distances;
    auto& parents = _instance->parents;
    auto& visited = _instance->visited;

    sycl::queue& queue = G.get_queue();

    using load_balance_t = sygraph::operators::LoadBalancer;
    using direction_t = sygraph::operators::Direction;
    using frontier_view_t = sygraph::frontier::FrontierView;
    using frontier_impl_t = sygraph::frontier::FrontierType;

    auto inFrontier = sygraph::frontier::make_frontier<frontier_view_t::vertex, frontier_impl_t::bitmap>(queue, G);
    auto outFrontier = sygraph::frontier::make_frontier<frontier_view_t::vertex, frontier_impl_t::bitmap>(queue, G);

    size_t size = G.get_vertex_count();

    int iter = 0;
    inFrontier.insert(source);

    while (!inFrontier.empty()) {
      auto e1 = sygraph::operators::advance::vertex<load_balance_t::workgroup_mapped>(G, inFrontier, outFrontier, [=](auto src, auto dst, auto edge, auto weight) -> bool {
        weight_t source_distance = sygraph::sync::load(&distances[src]);
        weight_t distance_to_neighbor = source_distance + weight;

        // Check if the destination node has been claimed as someone's child
        weight_t recover_distance = sygraph::sync::load(&distances[dst]);
        recover_distance = sygraph::sync::min(&(distances[dst]), &distance_to_neighbor);

        return (distance_to_neighbor < recover_distance);
      });
      e1.wait();

      auto e2 = sygraph::operators::filter::inplace(G, outFrontier, [=](auto vertex) -> bool {
        if (visited[vertex] == iter)
          return false;
        visited[vertex] = iter;
        return true;
      });
      e2.wait();

#ifdef ENABLE_PROFILING
      sygraph::profiler::add_event(e1, "advance");
      sygraph::profiler::add_event(e2, "filter");
#endif
      
      sygraph::frontier::swap(inFrontier, outFrontier);
      outFrontier.clear();
      iter++;
    }
#ifdef ENABLE_PROFILING
    sygraph::profiler::add_visited_edges(_instance->get_visited_edges());
#endif
  }

  const weight_t get_distance(size_t vertex) const {
    return _instance->distances[vertex];
  }

  const vertex_t get_parents(size_t vertex) const {
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