#pragma once

#include <fstream>
#include <iostream>
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
 * @param directed Whether the graph is directed or not. If it's set to true, the graph is considered undirected and so all the values will be
 * inserted twice.
 * @return The COO representation of the graph.
 */
template<typename value_t, typename index_t, typename offset_t = types::offset_t>
sygraph::formats::COO<value_t, index_t, offset_t> fromCOO(std::istream& iss, bool undirected = false) {
  // Initialize row pointers for CSR format

  // skip all comments that starts with %
  while (iss.peek() == '%') { iss.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); }

  std::string line;
  if (std::getline(iss, line).eof()) { throw std::runtime_error("Error: could not read the first line of the file."); }
  size_t n_nodes1, n_nodes2, num_edges;
  std::istringstream{line} >> n_nodes1 >> n_nodes2 >> num_edges;

  num_edges *= undirected ? 2 : 1;

  std::vector<index_t> coo_row_indices(num_edges);
  std::vector<index_t> coo_col_indices(num_edges);
  std::vector<value_t> coo_values(num_edges);

  size_t i = 0;
  while (std::getline(iss, line)) {
    std::istringstream liss{line};
    index_t u, v;
    value_t w;
    liss >> u >> v;
    if (!(liss >> w)) w = 1; // if no weight is provided, set it to 1 (default weight)
    coo_row_indices[i] = u;
    coo_col_indices[i] = v;
    coo_values[i] = w;
    ++i;
    if (undirected) {
      coo_row_indices[i] = v;
      coo_col_indices[i] = u;
      coo_values[i] = w;
      ++i;
    }
  }

  return sygraph::formats::COO<value_t, index_t, offset_t>(coo_row_indices, coo_col_indices, coo_values);
}

} // namespace coo
} // namespace io
} // namespace v0
} // namespace sygraph
