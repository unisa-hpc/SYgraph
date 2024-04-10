#pragma once

#include <sycl/sycl.hpp>
#include <memory>

#include <sygraph/graph/graph_impl.hpp>
#include <sygraph/graph/properties.hpp>


namespace sygraph {
inline namespace v0 {
namespace graph {

enum class GraphRepr {
  csr,
  csc,
  coo,
};

class Graph {
public:
  template <typename index_t,
            typename offset_t,
            typename value_t>
  Graph(sygraph::formats::CSR<index_t, offset_t, value_t>& csr) {
    impl = std::make_unique<detail::graph_csr_t>(csr);
  }

  template <typename index_t,
            typename offset_t,
            typename value_t>
  Graph(sygraph::formats::CSC<index_t, offset_t, value_t>& csc) {
    throw std::runtime_error("Not implemented yet");
  }

  template <typename index_t,
            typename offset_t,
            typename value_t>
  Graph(sygraph::formats::COO<index_t, offset_t, value_t>& coo) {
    throw std::runtime_error("Not implemented yet");
  }

  Graph() = delete;
  
  ~Graph() = default;

private:

  std::unique_ptr<detail::graph_impl> impl;
};

} // namespace graph
} // namespace v0
} // namespace sygraph