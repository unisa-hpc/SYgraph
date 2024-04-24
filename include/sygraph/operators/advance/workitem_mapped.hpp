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

  using type_t = typename in_frontier_t::type_t;
  size_t active_elements_size = in.get_num_active_elements();
  type_t* active_elements = sycl::malloc_shared<type_t>(active_elements_size, q);
  in.get_active_elements(active_elements);

  sygraph::event ret {q.submit([&](sycl::handler& cgh) {
    auto inDevFrontier = in.get_device_frontier();
    auto outDevFrontier = out.get_device_frontier();
    auto graphDev = graph.get_device_graph();

    cgh.parallel_for<class advance_kernel>(sycl::range<1>(active_elements_size), [=](sycl::id<1> idx) {
      auto element = active_elements[idx];
      auto start = graphDev.begin(element);
      auto end = graphDev.end(element);

      // each work item takes care of all the neighbours of the vertex he is responsible for
      for (auto i = start; i != end; ++i) {
        auto neighbour = *i;
        if (functor(element, neighbour)) {
          outDevFrontier.insert(neighbour);
        }
      }
    });
  })};

  sycl::free(active_elements, q);
  return ret;
}

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
sygraph::event pull(graph_t& graph, const in_frontier_t& in, out_frontier_t& out, lambda_t&& functor) {
  
  sycl::queue& q = graph.get_queue();

  using type_t = typename in_frontier_t::type_t;
  size_t inactive_elements_size = in.get_num_elems() - in.get_num_active_elements();
  type_t* inactive_elements = sycl::malloc_shared<type_t>(inactive_elements_size, q);
  in.get_active_elements(inactive_elements, false);
  std::cout << "Inactive elements size: " << inactive_elements_size << "\n";

  sygraph::event ret {q.submit([&](sycl::handler& cgh) {
    auto inDevFrontier = in.get_device_frontier();
    auto outDevFrontier = out.get_device_frontier();
    auto graphDev = graph.get_device_graph();

    cgh.parallel_for<class advance_kernel>(sycl::range<1>(inactive_elements_size), [=](sycl::id<1> idx) {
      auto element = inactive_elements[idx];
      auto start = graphDev.begin(element);
      auto end = graphDev.end(element);

      // each work item takes care of all the neighbours of the vertex he is responsible for
      for (auto i = start; i != end; ++i) {
        auto neighbour = *i;
        if (inDevFrontier.check(neighbour) && functor(neighbour, element)) {
          outDevFrontier.insert(element);
          break;
        }
      }
    });
  })};

  sycl::free(inactive_elements, q);
  return ret;
}

} // namespace detail  
} // namespace advance
} // namespace operators
} // namespace v0
} // namespace sygraph