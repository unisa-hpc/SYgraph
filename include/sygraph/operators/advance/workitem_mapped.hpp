#pragma once

#include <sycl/sycl.hpp>
#include <memory>

#include <sygraph/graph/graph.hpp>
#include <sygraph/operators/config.hpp>

namespace sygraph {
inline namespace v0 {
namespace operators {

namespace advance {

namespace detail {

/**
 * @todo Implement the logic for advancing the frontier.
   * The logic for implementing this class shouldn't take into account any modification of any data structure.
   * That can be done in the user code.
   * This function should only implmement the advance logic.
   * Also, the lambda should be called with the correct arguments (so maybe specifying a signature for the lambda would be a good idea).
   * The lambda also, should return a boolean value indicating if the node/edge should be added to the out frontier (maybe an auxiliar data structure might be needed).
   * How can I do that in order to not lose neither performance nor flexibility?
*/
template <typename graph_t,
          typename in_frontier_t,
          typename out_frontier_t,
          typename lambda_t>
sygraph::event push(graph_t& graph, const in_frontier_t& in, out_frontier_t& out, lambda_t&& functor) {
  
  sycl::queue& q = graph.get_queue();

  auto active_elements = in.get_active_elements();

  using type_t = typename in_frontier_t::type_t;
  sycl::buffer<type_t, 1> active_elements_buffer(active_elements.data(), sycl::range<1>(active_elements.size()));

  return {q.submit([&](sycl::handler& cgh) {
    sycl::accessor active_elements_acc(active_elements_buffer, cgh, sycl::read_only);
    auto inDevFrontier = in.get_device_frontier();
    auto outDevFrontier = out.get_device_frontier();
    auto graphDev = graph.get_device_graph();

    cgh.parallel_for<class advance_kernel>(sycl::range<1>(active_elements.size()), [=](sycl::id<1> idx) {
      auto element = active_elements_acc[idx];
      auto start = graphDev.begin(element);
      auto end = graphDev.end(element);

      for (auto i = start; i != end; ++i) {
        auto neighbour = *i;
        if (functor(element, neighbour)) {
          outDevFrontier.insert(neighbour);
        }
      }
    });
  })};
}

} // namespace detail  
} // namespace advance
} // namespace operators
} // namespace v0
} // namespace sygraph