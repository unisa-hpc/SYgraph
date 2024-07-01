#pragma once

#include <memory>
#include <sycl/sycl.hpp>

#include <sygraph/frontier/frontier.hpp>
#include <sygraph/frontier/frontier_settings.hpp>
#include <sygraph/graph/graph.hpp>
#include <sygraph/operators/advance/workitem_mapped.hpp>
#include <sygraph/operators/config.hpp>
#include <sygraph/sycl/event.hpp>
#include <sygraph/utils/vector.hpp>

namespace sygraph {
inline namespace v0 {
namespace operators {

namespace filter {
namespace detail {

template<graph::detail::GraphConcept GraphT, typename T, sygraph::frontier::frontier_view FrontierView, typename LambdaT>
sygraph::Event
inplace(GraphT& graph, const sygraph::frontier::Frontier<T, FrontierView, sygraph::frontier::frontier_type::bitmap>& frontier, LambdaT&& functor) {
  auto q = graph.getQueue();

  using type_t = T;
  size_t num_nodes = graph.getVertexCount();

  sygraph::Event e = q.submit([&](sycl::handler& cgh) {
    auto out_dev = frontier.getDeviceFrontier();

    cgh.parallel_for<class inplace_filter_kernel>(sycl::range<1>{num_nodes}, [=](sycl::id<1> idx) {
      type_t element = idx[0];
      if (out_dev.check(element) && !functor(element)) { out_dev.remove(element); }
    });
  });

  return e;
}

template<graph::detail::GraphConcept GraphT, typename T, sygraph::frontier::frontier_view FrontierView, typename LambdaT>
sygraph::Event external(GraphT& graph,
                        const sygraph::frontier::Frontier<T, FrontierView, sygraph::frontier::frontier_type::bitmap>& in,
                        const sygraph::frontier::Frontier<T, FrontierView, sygraph::frontier::frontier_type::bitmap>& out,
                        LambdaT&& functor) {
  auto q = graph.getQueue();
  out.clear();

  using type_t = T;
  size_t num_nodes = graph.getVertexCount();

  auto out_dev = out.getDeviceFrontier();
  auto in_dev = in.getDeviceFrontier();

  sygraph::Event e = q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<class external_filter_kernel>(sycl::range<1>{num_nodes}, [=](sycl::id<1> idx) {
      type_t element = idx[0];
      if (in_dev.check(element) && functor(element)) { out_dev.insert(element); }
    });
  });

  return e;
}

} // namespace detail
} // namespace filter
} // namespace operators
} // namespace v0
} // namespace sygraph