#pragma once

#include <iostream>
#include <fstream>
#include <sstream>

#include <sycl/sycl.hpp>

#include <sygraph/formats/coo.hpp>
#include <sygraph/utils/types.hpp>

namespace sygraph {
inline namespace v0 {
namespace io {
namespace coo {

/**
 * Converts a COO (Coordinate List) representation of a graph to CSR (Compressed Sparse Row) format.
 * 
 * @tparam value_t The value type of the graph.
 * @tparam index_t The index type of the graph.
 * @tparam offset_t The offset type of the graph.
 * 
 * @param iss The input stream containing the COO representation of the graph.
 * @return The CSR representation of the graph.
 * 
 * @throws std::runtime_error if the conversion is not implemented.
 */
template <typename value_t, typename index_t, typename offset_t = types::offset_t>
sygraph::formats::COO<value_t, index_t, offset_t> from_coo(std::istream& iss) {
  // Initialize row pointers for CSR format
  size_t num_edges;
  iss >> num_edges;

  std::vector<index_t> coo_row_indices(num_edges);
  std::vector<index_t> coo_col_indices(num_edges);
  std::vector<value_t> coo_values(num_edges);

  for (int i = 0; i < num_edges; ++i) {
    iss >> coo_row_indices[i] >> coo_col_indices[i] >> coo_values[i];    
  }

  return sygraph::formats::COO<value_t, index_t, offset_t>(coo_row_indices, coo_col_indices, coo_values);
}

} // namespace csr
} // namespace io
} // namespace v0
} // namespace sygraph
