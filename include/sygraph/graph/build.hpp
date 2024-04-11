#pragma once

#include <sycl/sycl.hpp>

#include <sygraph/graph/graph.hpp>
#include <sygraph/graph/graph_impl.hpp>
#include <sygraph/graph/formats/graph_csr.hpp>
#include <sygraph/graph/properties.hpp>
#include <sygraph/formats/csr.hpp>
#include <sygraph/utils/memory.hpp>

namespace sygraph {
inline namespace v0 {
namespace graph {
namespace build {

template <memory::space space, typename index_t, typename offset_t, typename value_t>
auto from_csr(sycl::queue& q, sygraph::formats::CSR<index_t, offset_t, value_t> csr, graph::Properties properties = graph::Properties()) {
  using csr_t = graph::detail::graph_impl_csr_t<space, index_t, offset_t, value_t>;
  using graph_type_t = graph::Graph<index_t, offset_t, value_t, csr_t>;

  graph_type_t g(properties);
  g.template set<sygraph::formats::CSR<index_t, offset_t, value_t>>(csr);
  return g;
};

} // namespace build
} // namespace graph
} // namespace v0
} // namespace sygraph