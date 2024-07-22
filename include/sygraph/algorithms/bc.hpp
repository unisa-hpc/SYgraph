#include <sycl/sycl.hpp>

#include <sygraph/frontier/frontier.hpp>
#include <sygraph/graph/graph.hpp>
#include <sygraph/operators/advance/advance.hpp>
#include <sygraph/operators/for/for.hpp>
#include <sygraph/sync/atomics.hpp>

#ifdef ENABLE_PROFILING
#include <sygraph/utils/profiler.hpp>
#endif
#include <memory>

namespace sygraph {
inline namespace v0 {
namespace algorithms {
namespace detail {

template<typename T>
void printFrontier(T& f, std::string prefix = "") {
  using type_t = typename T::type_t;
  auto size = f.getBitmapSize() * f.getBitmapRange();
  std::cout << prefix;
  for (int i = size - 1; i >= 0; --i) { std::cout << (f.check(static_cast<type_t>(i)) ? "1" : "0"); }
  std::cout << " [" << f.getDeviceFrontier().getData()[0] << "]" << std::endl;
  std::cout << std::endl;
}

template<typename GraphType>
struct BCInstance {
  using vertex_t = typename GraphType::vertex_t;
  using edge_t = typename GraphType::edge_t;
  using weight_t = typename GraphType::weight_t;

  const vertex_t invalid = std::numeric_limits<vertex_t>::max();

  GraphType& G;
  vertex_t source;

  vertex_t* labels;
  weight_t* deltas;
  weight_t* sigmas;
  weight_t* bc_values;

  BCInstance(GraphType& G, const vertex_t source) : G(G), source(source) {
    sycl::queue& queue = G.getQueue();
    size_t size = G.getVertexCount();

    labels = sygraph::memory::detail::memoryAlloc<vertex_t, memory::space::shared>(size, queue);
    deltas = sygraph::memory::detail::memoryAlloc<weight_t, memory::space::device>(size, queue);
    sigmas = sygraph::memory::detail::memoryAlloc<weight_t, memory::space::device>(size, queue);
    bc_values = sygraph::memory::detail::memoryAlloc<weight_t, memory::space::device>(size, queue);

    queue.fill(labels, static_cast<vertex_t>(this->invalid), size);
    queue.memset(deltas, 0, size);
    queue.memset(sigmas, 0, size);
    queue.memset(bc_values, 0, size);
    queue.wait_and_throw();

    queue.memset(&sigmas[source], static_cast<weight_t>(1), 1);
    labels[source] = 0;
    queue.wait_and_throw();
  }

  ~BCInstance() {
    sycl::queue& queue = G.getQueue();
    sycl::free(labels, queue);
    sycl::free(deltas, queue);
    sycl::free(sigmas, queue);
    sycl::free(bc_values, queue);
  }
};
} // namespace detail

template<typename GraphType>
class BC {
  using vertex_t = typename GraphType::vertex_t;
  using edge_t = typename GraphType::edge_t;
  using weight_t = typename GraphType::weight_t;

public:
  BC(GraphType& g) : _g(g){};

  void init(const vertex_t source) { _instance = std::make_unique<detail::BCInstance<GraphType>>(_g, source); }

  void reset() { _instance.reset(); }

  void run() {
    if (!_instance) { throw std::runtime_error("BC instance not initialized"); }

    auto& G = _instance->G;
    sycl::queue& queue = G.getQueue();
    size_t size = G.getVertexCount();
    auto source = _instance->source;

    auto in_frontier
        = sygraph::frontier::makeFrontier<sygraph::frontier::frontier_view::vertex, sygraph::frontier::frontier_type::hierachic_bitmap>(queue, G);
    auto out_frontier
        = sygraph::frontier::makeFrontier<sygraph::frontier::frontier_view::vertex, sygraph::frontier::frontier_type::hierachic_bitmap>(queue, G);

    in_frontier.insert(source);

    vertex_t invalid = _instance->invalid;
    vertex_t* labels = _instance->labels;
    weight_t* deltas = _instance->deltas;
    weight_t* sigmas = _instance->sigmas;
    weight_t* bc_values = _instance->bc_values;

    using frontier_state_t = typename decltype(in_frontier)::frontier_state_type;
    std::vector<frontier_state_t> frontiers_states;

    for (int i = 0; i < size; ++i) { std::cout << labels[i] << " "; }
    std::cout << std::endl;

    while (!isConverged()) {
      if (_forward) {
        while (!in_frontier.empty() && _depth < 50) { // remove _depth < 20
          std::cout << "Forward" << std::endl;
          auto e = sygraph::operators::advance::frontier<sygraph::operators::load_balancer::workgroup_mapped,
                                                         sygraph::frontier::frontier_view::vertex,
                                                         sygraph::frontier::frontier_view::vertex>(
              G, in_frontier, out_frontier, [=](auto src, auto dst, auto edge, auto weight) -> bool {
                vertex_t new_label = labels[src] + 1;
                vertex_t old_label = invalid;

                old_label = sygraph::sync::cas(&labels[dst], old_label, new_label);

                if ((old_label != invalid) && (old_label != new_label)) { return false; }

                sygraph::sync::atomicFetchAdd(sigmas + dst, sigmas[src]);
                return old_label == invalid;
              });
          e.wait_and_throw();

#ifdef ENABLE_PROFILING
          sygraph::Profiler::addEvent(e, "BC::Forward");
#endif

          detail::printFrontier(out_frontier);
          _depth++;
          _search_depth++;
          frontiers_states.push_back(out_frontier.saveState());
          sygraph::frontier::swap(out_frontier, in_frontier);
        }
        for (int i = 0; i < size; ++i) { std::cout << labels[i] << " "; }
        std::cout << std::endl;
        _forward = false;
        std::cout << "out of while" << std::endl;
      } else if (_backward) {
        std::cout << "Backward" << std::endl;
        _forward = false;
        auto op = [=](auto src, auto dst, auto edge, auto weight) -> bool {
          if (src == source) { return false; }

          auto s_label = labels[src];
          auto d_label = labels[dst];
          if (s_label + 1 != d_label) { return false; }

          auto update = sigmas[src] / sigmas[dst] * (1 + deltas[dst]);
          sygraph::sync::atomicFetchAdd(deltas + src, update);
          sygraph::sync::atomicFetchAdd(bc_values + src, update);

          return false;
        };

        while (true) {
          in_frontier.loadState(frontiers_states.back());
          frontiers_states.pop_back();

          sygraph::operators::advance::frontier<sygraph::operators::load_balancer::workgroup_mapped,
                                                sygraph::frontier::frontier_view::vertex,
                                                sygraph::frontier::frontier_view::none>(G, in_frontier, out_frontier, std::forward<decltype(op)>(op));
          _depth--;
          _search_depth++;
          if (isBackwardConverged()) { break; }
        }
      }
    }

    using load_balance_t = sygraph::operators::load_balancer;
    using direction_t = sygraph::operators::direction;
    using frontier_view_t = sygraph::frontier::frontier_view;
    using frontier_impl_t = sygraph::frontier::frontier_type;
  }

protected:
  bool isBackwardConverged() {
    if (_depth == 0) {
      _backward = false;
      return true;
    }

    return false;
  }

  virtual bool isConverged() { return !_forward && !_backward; }

  bool _forward = true;
  bool _backward = true;

  size_t _depth = 0;
  size_t _search_depth = 1;
  const size_t _max_depth = 5;

private:
  GraphType& _g;
  std::unique_ptr<sygraph::algorithms::detail::BCInstance<GraphType>> _instance;
};

} // namespace algorithms
} // namespace v0
} // namespace sygraph
