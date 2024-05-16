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

namespace filter {

namespace v0 {
template <typename graph_t,
          typename frontier_t,
          typename lambda_t>
sygraph::event inplace(graph_t& graph, frontier_t& frontier, lambda_t&& functor) {
  auto q = graph.get_queue();

  using type_t = typename frontier_t::type_t;
  // size_t active_elements_size = frontier.get_num_active_elements();
  size_t active_elements_size = types::detail::MAX_ACTIVE_ELEMS_SIZE;
  type_t* active_elements;
  if (!frontier.self_allocated()) {
    active_elements = sycl::malloc_shared<type_t>(active_elements_size, q);
  }
  frontier.get_active_elements(active_elements, active_elements_size);
  auto outDev = frontier.get_device_frontier();

  sygraph::event e = q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<class inplace_filter_kernel>(sycl::range<1>{active_elements_size}, [=](sycl::id<1> idx) {
      auto element = active_elements[idx];
      if (!functor(element)) {
        outDev.remove(element);
      }
    });
  });

  if (!frontier.self_allocated()) {
    sycl::free(active_elements, q);
  }

  return e;
}
}

namespace v0 { 
template <typename graph_t,
          typename frontier_t,
          typename lambda_t>
sygraph::event external(graph_t& graph, frontier_t& in, frontier_t& out, lambda_t&& functor) {
  auto q = graph.get_queue();
  out.clear();

  using type_t = typename frontier_t::type_t;
  // size_t active_elements_size = in.get_num_active_elements();
  size_t active_elements_size = types::detail::MAX_ACTIVE_ELEMS_SIZE;
  type_t* active_elements;
  if (!in.self_allocated()) {
    active_elements = sycl::malloc_shared<type_t>(active_elements_size, q);
  }
  in.get_active_elements(active_elements, active_elements_size);

  sygraph::event e = q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<class external_filter_kernel>(sycl::range<1>{active_elements_size}, [=](sycl::id<1> idx) {
      auto element = active_elements[idx];
      if (functor(element)) {
        out.insert(element);
      }
    });
  });

  if (!in.self_allocated()) {
    sycl::free(active_elements, q);
  }

  return e;
}
}

inline namespace v1 {
template <typename graph_t,
          typename frontier_t,
          typename lambda_t>
sygraph::event inplace(graph_t& graph, frontier_t& frontier, lambda_t&& functor) {
  auto q = graph.get_queue();

  using type_t = typename frontier_t::type_t;
  size_t num_nodes = graph.get_vertex_count();

  sygraph::event e = q.submit([&](sycl::handler& cgh) {
    auto outDev = frontier.get_device_frontier();

    cgh.parallel_for<class inplace_filter_kernel>(sycl::range<1>{num_nodes}, [=](sycl::id<1> idx) {
      type_t element = idx[0];
      if (outDev.check(element) && !functor(element)) {
        outDev.remove(element);
      }
    });
  });

  return e;
}
}

inline namespace v1 { 
template <typename graph_t,
          typename frontier_t,
          typename lambda_t>
sygraph::event external(graph_t& graph, frontier_t& in, frontier_t& out, lambda_t&& functor) {
  auto q = graph.get_queue();
  out.clear();

  using type_t = typename frontier_t::type_t;
  size_t num_nodes = graph.get_vertex_count();

  auto outDev = out.get_device_frontier();
  auto inDev = in.get_device_frontier();

  sygraph::event e = q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<class external_filter_kernel>(sycl::range<1>{num_nodes}, [=](sycl::id<1> idx) {
      type_t element = idx[0];
      if (inDev.check(element) && functor(element)) {
        out.insert(element);
      }
    });
  });

  return e;
}
}


} // namespace advance
} // namespace operators
} // namespace v0
} // namespace sygraph