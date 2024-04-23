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
 * Retrieves the COO (Coordinate List) representation of a graph.
 * 
 * @tparam value_t The value type of the graph.
 * @tparam index_t The index type of the graph.
 * @tparam offset_t The offset type of the graph.
 * 
 * @param iss The input stream containing the COO representation of the graph.
 * @return The COO representation of the graph.
 */
template <typename value_t, typename index_t, typename offset_t = types::offset_t>
sygraph::formats::COO<value_t, index_t, offset_t> from_coo(std::istream& iss) {
  // Initialize row pointers for CSR format

  // skip all comments that starts with %
  while (iss.peek() == '%') {
    iss.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }

  size_t n_nodes1, n_nodes2, num_edges;
  iss >> n_nodes1 >> n_nodes2 >> num_edges;

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
