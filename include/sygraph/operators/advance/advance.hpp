#pragma once

#include <memory>
#include <sycl/sycl.hpp>

#include <sygraph/graph/graph.hpp>
#include <sygraph/operators/config.hpp>
#include <sygraph/operators/advance/workitem_mapped.hpp>
#include <sygraph/frontier/frontier.hpp>
#include <sygraph/frontier/frontier_settings.hpp>
#include <sygraph/sycl/event.hpp>
#include <sygraph/utils/vector.hpp>

namespace sygraph {
inline namespace v0 {
namespace operators {

namespace advance {

template<typename graph_t,
         typename in_frontier_t,
         typename out_frontier_t>
class AdvanceFunctor {
  bool operator()() {
  };
};

template <sygraph::operators::LoadBalancer lb,
          typename graph_t,
          typename in_frontier_t,
          typename out_frontier_t,
          typename lambda_t>
sygraph::event push(graph_t& graph, in_frontier_t& in, out_frontier_t& out, lambda_t&& functor) {
  if constexpr (lb == sygraph::operators::LoadBalancer::workitem_mapped) {
    return sygraph::operators::advance::workitem_mapped::push(graph, in, out, std::forward<lambda_t>(functor));
  } else {
    throw std::runtime_error("Load balancer not implemented");
  }
}

} // namespace advance
} // namespace operators
} // namespace v0
} // namespace sygraph