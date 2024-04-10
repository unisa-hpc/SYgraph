#pragma once

#include <sycl/sycl.hpp>
#include <memory>

#include <sygraph/formats/csr.hpp>
#include <sygraph/formats/csc.hpp>
#include <sygraph/formats/coo.hpp>

namespace sygraph {
inline namespace v0 {
namespace graph {
namespace detail {

class graph_impl {
public:
  // TODO: Retrieve the edges of a node, the neighbors of a node, etc.
  // TODO: [Optional] Add methods for graph manipulation: add_node, add_edge, remove_node, remove_edge, etc.

};

} // namespace detail
} // namespace graph
} // namespace v0
} // namespace sygraph