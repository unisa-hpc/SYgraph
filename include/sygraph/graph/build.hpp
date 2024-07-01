#pragma once

#include <sycl/sycl.hpp>

#include <sygraph/formats/csr.hpp>
#include <sygraph/graph/graph.hpp>
#include <sygraph/graph/impls/graph_csr.hpp>
#include <sygraph/graph/properties.hpp>
#include <sygraph/utils/memory.hpp>

namespace sygraph {
inline namespace v0 {
namespace graph {
namespace build {

template<memory::space Space, typename IndexT, typename OffsetT, typename ValueT>
auto fromCSR(sycl::queue& q, sygraph::formats::CSR<ValueT, IndexT, OffsetT> csr, graph::Properties properties = graph::Properties()) {
  using csr_t = graph::detail::GraphCSR<Space, IndexT, OffsetT, ValueT>;
  using GraphT = detail::GraphCSR<Space, IndexT, OffsetT, ValueT>;
  return GraphT{q, csr, properties};
};

} // namespace build
} // namespace graph
} // namespace v0
} // namespace sygraph