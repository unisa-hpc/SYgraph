
#pragma once
#include <sycl/sycl.hpp>

#include <sygraph/frontier/frontier.hpp>
#include <sygraph/graph/graph.hpp>
#include <sygraph/operators/advance/advance.hpp>

#ifdef ENABLE_PROFILING
#include <sygraph/utils/profiler.hpp>
#endif
#include <memory>

namespace sygraph {
inline namespace v0 {
namespace algorithms {
namespace detail {


template<typename GraphType>
struct PRInstance {
  using vertex_t = typename GraphType::vertex_t;
  using edge_t = typename GraphType::edge_t;
  using weight_t = typename GraphType::weight_t;

  GraphType& G;

  weight_t* rank;
  weight_t* plast;
  weight_t* iweights;
  weight_t alpha = 0.85;
  weight_t tol = 1e-6;

  PRInstance(GraphType& G) : G(G) {
    sycl::queue& queue = G.getQueue();
    size_t size = G.getVertexCount();

    plast = memory::detail::memoryAlloc<weight_t, memory::space::shared>(size, queue);
    iweights = memory::detail::memoryAlloc<weight_t, memory::space::shared>(size, queue);
    rank = memory::detail::memoryAlloc<weight_t, memory::space::shared>(size, queue);
  }

  ~PRInstance() {
    sycl::queue& queue = G.getQueue();
    sycl::free(plast, queue);
    sycl::free(iweights, queue);
    sycl::free(rank, queue);
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

  void init() { _instance = std::make_unique<detail::PRInstance<GraphType>>(_g); }

  void reset() { _instance.reset(); }

  void run() {
    if (!_instance) { throw std::runtime_error("PR instance not initialized"); }


    auto& G = _instance->G;
    sycl::queue& queue = G.getQueue();
    size_t size = G.getVertexCount();

    auto plast = _instance->plast;
    auto iweights = _instance->iweights;
    auto alpha = _instance->alpha;
    auto rank = _instance->rank;

    using load_balance_t = sygraph::operators::load_balancer;
    using direction_t = sygraph::operators::direction;
    using frontier_view_t = sygraph::frontier::frontier_view;
    using frontier_impl_t = sygraph::frontier::frontier_type;

    auto g_dev = G.getDeviceGraph();

    auto in_frontier
        = sygraph::frontier::makeFrontier<sygraph::frontier::frontier_view::vertex, sygraph::frontier::frontier_type::hierachic_bitmap>(queue, G);
    auto out_frontier
        = sygraph::frontier::makeFrontier<sygraph::frontier::frontier_view::vertex, sygraph::frontier::frontier_type::hierachic_bitmap>(queue, G);

    queue.fill(plast, 0, size);
    queue.fill(rank, 1.0 / size, size);

    auto e_init = queue.submit([&](sycl::handler& cgh) {
      cgh.parallel_for<class pr_init>(sycl::range<1>{size}, [=](sycl::id<1> idx) {
        auto v = idx[0];
        iweights[v] = g_dev.getDegree(v);
      });
    });
    e_init.wait();

    bool converged = false;
    size_t iteration = 0;
    while (iteration < 5) {
      queue.copy(plast, rank, size);
      queue.fill(plast, (1 - alpha) / size, size);
      queue.wait();

      auto e1 = sygraph::operators::advance::vertices<load_balance_t::workgroup_mapped, frontier_view_t::none>(
          G, out_frontier, [=](auto src, auto dst, auto edge, auto weight) -> bool {
            weight_t update = plast[src] * iweights[src] * weight;
            sygraph::sync::atomicFetchAdd(plast + dst, update);
            return false;
          });
      e1.wait();

#ifdef ENABLE_PROFILING
      sygraph::Profiler::addEvent(e1, "rank_update");
#endif

      iteration++;
    }
  }

private:
  GraphType& _g;
  std::unique_ptr<sygraph::algorithms::detail::PRInstance<GraphType>> _instance;
};

} // namespace algorithms
} // namespace v0
} // namespace sygraph
