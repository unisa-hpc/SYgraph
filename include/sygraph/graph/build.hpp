#pragma once

#include <sycl/sycl.hpp>

#include <sygraph/graph/graph.hpp>
#include <sygraph/graph/impls/graph_csr.hpp>
#include <sygraph/graph/properties.hpp>
#include <sygraph/formats/csr.hpp>
#include <sygraph/utils/memory.hpp>

namespace sygraph {
inline namespace v0 {
namespace graph {
namespace build {

template <memory::space space, typename index_t, typename offset_t, typename value_t>
auto from_csr(sycl::queue& q, sygraph::formats::CSR<value_t, index_t, offset_t> csr, graph::Properties properties = graph::Properties()) {
  using csr_t = graph::detail::graph_csr_t<space, index_t, offset_t, value_t>;
  using graph_t = detail::graph_csr_t<space, index_t, offset_t, value_t>;
  return graph_t {q, csr, properties};
};

} // namespace build
} // namespace graph
} // namespace v0
} // namespace sygraph