/*
 * Copyright (c) 2025 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <fstream>
#include <iostream>
#include <sstream>

#include <sycl/sycl.hpp>

#include <sygraph/formats/coo.hpp>
#include <sygraph/utils/types.hpp>

namespace sygraph {
namespace io {
namespace coo {

/**
 * Retrieves the COO (Coordinate List) representation of a graph.
 *
 * @tparam ValueT The value type of the graph.
 * @tparam IndexT The index type of the graph.
 * @tparam OffsetT The offset type of the graph.
 *
 * @param iss The input stream containing the COO representation of the graph.
 * @param directed Whether the graph is directed or not. If it's set to true, the graph is considered undirected and so all the values will be
 * inserted twice.
 * @return The COO representation of the graph.
 */
template<typename ValueT, typename IndexT, typename OffsetT = types::offset_t>
sygraph::formats::COO<ValueT, IndexT, OffsetT> fromCOO(std::istream& iss, bool undirected = false) {
  // Initialize row pointers for CSR format

  // skip all comments that starts with %
  while (iss.peek() == '%') { iss.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); }

  std::string line;
  if (std::getline(iss, line).eof()) { throw std::runtime_error("Error: could not read the first line of the file."); }
  size_t n_nodes1;
  size_t n_nodes2;
  size_t num_edges;
  std::istringstream{line} >> n_nodes1 >> n_nodes2 >> num_edges;

  num_edges *= undirected ? 2 : 1;

  std::vector<IndexT> coo_row_indices(num_edges);
  std::vector<IndexT> coo_col_indices(num_edges);
  std::vector<ValueT> coo_values(num_edges);

  size_t i = 0;
  while (std::getline(iss, line)) {
    std::istringstream liss{line};
    IndexT u;
    IndexT v;
    ValueT w;
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

  return sygraph::formats::COO<ValueT, IndexT, OffsetT>(coo_row_indices, coo_col_indices, coo_values);
}

} // namespace coo
} // namespace io
} // namespace sygraph
