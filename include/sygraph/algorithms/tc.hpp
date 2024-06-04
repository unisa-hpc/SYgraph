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
struct TCInstance {
  using vertex_t = typename GraphType::vertex_t;
  using edge_t = typename GraphType::edge_t;
  using weight_t = typename GraphType::weight_t;

  GraphType& G;

  TCInstance(GraphType& G) : G(G) {
    
  }

  ~TCInstance() {
    sycl::queue& queue = G.getQueue();s
  }
};
} // namespace detail


template<typename GraphType> 
class TC {
  using vertex_t = typename GraphType::vertex_t;
  using edge_t = typename GraphType::edge_t;
  using weight_t = typename GraphType::weight_t;

public:
  TC(GraphType& g) : _g(g) {};

  
  void init(vertex_t& source) {
    _instance = std::make_unique<detail::TCInstance<GraphType>>(_g, source);
  }

  
  void reset() {
    _instance.reset();
  }

  
  template <bool enable_profiling = false> 
  void run() {
    if (!_instance) {
      throw std::runtime_error("TC instance not initialized");
    }

    auto& G = _instance->G;

    sycl::queue& queue = G.getQueue();

    using load_balance_t = sygraph::operators::LoadBalancer;
    using direction_t = sygraph::operators::Direction;
    using frontier_view_t = sygraph::frontier::FrontierView;
    using frontier_impl_t = sygraph::frontier::FrontierType;

    auto inFrontier = sygraph::frontier::makeFrontier<frontier_view_t::vertex, frontier_impl_t::bitmap>(queue, G);
    auto outFrontier = sygraph::frontier::makeFrontier<frontier_view_t::vertex, frontier_impl_t::bitmap>(queue, G);

    size_t size = G.getVertexCount();

    int iter = 0;
    inFrontier.insert(source);

    while (!inFrontier.empty()) {
      

#ifdef ENABLE_PROFILING

#endif
      iter++;
    }
  }

private:
  GraphType& _g;
  std::unique_ptr<detail::TCInstance<GraphType>> _instance;
};

} // namespace algorithms
} // namespace v0
} // namespace sygraph