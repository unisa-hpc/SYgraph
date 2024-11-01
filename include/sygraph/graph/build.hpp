#pragma once

#include <sycl/sycl.hpp>

#include <sygraph/formats/csr.hpp>
#include <sygraph/graph/graph.hpp>
#include <sygraph/graph/impls/graph_csr.hpp>
#include <sygraph/graph/properties.hpp>
#include <sygraph/utils/memory.hpp>

namespace sygraph {
namespace graph {
namespace build {

/**
 * @brief Constructs a graph from a CSR (Compressed Sparse Row) format.
 *
 * @tparam ValueT The type of the values in the CSR format.
 * @tparam IndexT The type of the indices in the CSR format.
 * @tparam OffsetT The type of the offsets in the CSR format.
 * @tparam Space The memory space where the graph will be allocated.
 * @param q The SYCL queue to be used for graph operations.
 * @param csr The CSR format representation of the graph.
 * @param properties Optional properties for the graph.
 * @return A graph constructed from the given CSR format.
 */
template<memory::space Space, typename IndexT, typename OffsetT, typename ValueT>
auto fromCSR(sycl::queue& q, sygraph::formats::CSR<ValueT, IndexT, OffsetT> csr, graph::Properties properties = graph::Properties()) {
  using GraphT = detail::GraphCSR<Space, IndexT, OffsetT, ValueT>;
  return GraphT{q, csr, properties};
};

} // namespace build
} // namespace graph
} // namespace sygraph