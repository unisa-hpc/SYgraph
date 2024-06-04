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
namespace detail {

template <typename graph_t,
          typename T,
          typename sygraph::frontier::FrontierView FrontierView,
          typename lambda_t>
sygraph::event inplace(graph_t& graph, 
                              const sygraph::frontier::Frontier<T, FrontierView, sygraph::frontier::FrontierType::bitmap>& frontier, 
                              lambda_t&& functor) {
  auto q = graph.getQueue();

  using type_t = T;
  size_t num_nodes = graph.getVertexCount();

  sygraph::event e = q.submit([&](sycl::handler& cgh) {
    auto outDev = frontier.getDeviceFrontier();

    cgh.parallel_for<class inplace_filter_kernel>(sycl::range<1>{num_nodes}, [=](sycl::id<1> idx) {
      type_t element = idx[0];
      if (outDev.check(element) && !functor(element)) {
        outDev.remove(element);
      }
    });
  });

  return e;
}

template <typename graph_t,
          typename T,
          typename sygraph::frontier::FrontierView FrontierView,
          typename lambda_t>
sygraph::event external(graph_t& graph, 
                              const sygraph::frontier::Frontier<T, FrontierView, sygraph::frontier::FrontierType::bitmap>& in, 
                              const sygraph::frontier::Frontier<T, FrontierView, sygraph::frontier::FrontierType::bitmap>& out, 
                              lambda_t&& functor) {
  auto q = graph.getQueue();
  out.clear();

  using type_t = T;
  size_t num_nodes = graph.getVertexCount();

  auto outDev = out.getDeviceFrontier();
  auto inDev = in.getDeviceFrontier();

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

} // namespace detail
} // namespace advance
} // namespace operators
} // namespace v0
} // namespace sygraph