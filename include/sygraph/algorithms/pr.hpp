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
struct PRInstance {
  using vertex_t = typename GraphType::vertex_t;
  using edge_t = typename GraphType::edge_t;
  using weight_t = typename GraphType::weight_t;

  GraphType& G;

  weight_t* plast;
  weight_t* iweights;
  float alpha = 0.85;

  BCInstance(GraphType& G) : G(G) {
    sycl::queue& queue = G.getQueue();
    size_t size = G.getVertexCount();

    plast = memory::detail::memoryAlloc<weight_t, memory::space::shared>(size, queue);
    iweights = memory::detail::memoryAlloc<weight_t, memory::space::shared>(size, queue);
    rank = memory::detail::memoryAlloc<weight_t, memory::space::shared>(size, queue);
  }

  ~BCInstance() {
    sycl::queue& queue = G.getQueue();
    memory::detail::memoryFree(plast, queue);
    memory::detail::memoryFree(iweights, queue);
  }
};
} // namespace detail

template<typename GraphType>
class PR {
  using vertex_t = typename GraphType::vertex_t;
  using edge_t = typename GraphType::edge_t;
  using weight_t = typename GraphType::weight_t;

public:
  PR(GraphType& g) : _g(g) {};

  void init() { _instance = std::make_unique<detail::BCInstance<GraphType>>(_g); }

  void reset() { _instance.reset(); }

  void run() {
    if (!_instance) { throw std::runtime_error("PR instance not initialized"); }


    auto& G = _instance->G;
    sycl::queue& queue = G.getQueue();
    size_t size = G.getVertexCount();
    auto source = _instance->source;

    auto plast = _instance->plast;
    auto iweights = _instance->iweights;
    auto alpha = _instance->alpha;
    auto rank = _instance->rank;

    using load_balance_t = sygraph::operators::load_balancer;
    using direction_t = sygraph::operators::direction;
    using frontier_view_t = sygraph::frontier::frontier_view;
    using frontier_impl_t = sygraph::frontier::frontier_type;

    auto in_frontier
        = sygraph::frontier::makeFrontier<sygraph::frontier::frontier_view::vertex, sygraph::frontier::frontier_type::hierachic_bitmap>(queue, G);
    auto out_frontier
        = sygraph::frontier::makeFrontier<sygraph::frontier::frontier_view::vertex, sygraph::frontier::frontier_type::hierachic_bitmap>(queue, G);


    while (true) {
      sygraph::operators::advance::vertices<load_balance_t::workgroup_mapped, frontier_view_t::none>(
          G, in_frontier, out_frontier, [&](auto src, auto dst, auto edge, auto weight) -> bool {
            weight_t update = plast[src] * iweights[src] * weight;
            sygraph::sync::atomicFetchAdd(plast + dst, update);
            return false;
          });
    }
  }

private:
  GraphType& _g;
  std::unique_ptr<sygraph::algorithms::detail::PRInstance<GraphType>> _instance;
};

} // namespace algorithms
} // namespace v0
} // namespace sygraph
