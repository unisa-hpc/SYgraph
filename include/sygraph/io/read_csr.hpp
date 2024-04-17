#include <iostream>
#include <fstream>
#include <sstream>

#include <sycl/sycl.hpp>

#include <sygraph/formats/csr.hpp>

namespace sygraph {
inline namespace v0 {
namespace io {
namespace csr {

/**
 * @brief Converts a matrix in CSR format to a CSR object.
 * 
 * This function reads a matrix in CSR format from a given input stream and converts it into a CSR object.
 * The CSR object contains the row offsets, column indices, and non-zero values of the matrix.
 * 
 * @tparam value_t The value type of the matrix elements.
 * @tparam index_t The index type used for column indices.
 * @tparam offset_t The offset type used for row offsets.
 * @param iss The input stream containing the matrix in CSR format.
 * @return The CSR object representing the matrix.
 */
template <typename value_t, typename index_t, typename offset_t>
sygraph::formats::CSR<value_t, index_t, offset_t> from_matrix(std::istream& iss) {
  size_t n_rows = 0;
  size_t n_nonzeros = 0;
  std::vector<offset_t> row_offsets;
  std::vector<index_t> column_indices;
  std::vector<value_t> nnz_values;

  // Read number of rows
  iss >> n_rows;

  row_offsets.push_back(0);

  // Read adjacency matrix
  for (int i = 0; i < n_rows; ++i) {
    for (int j = 0; j < n_rows; ++j) {
      value_t value;
      iss >> value;
      if (value != static_cast<value_t>(0)) {
        nnz_values.push_back(value);
        column_indices.push_back(j);
      }
    }
    row_offsets.push_back(nnz_values.size());
  }

  return sygraph::formats::CSR<value_t, index_t, offset_t>(row_offsets, column_indices, nnz_values);
}

/**
 * @brief Converts a matrix in CSR format from a file to a CSR object.
 * 
 * This function reads a matrix in CSR format from a given file and converts it into a CSR object.
 * The CSR object contains the row offsets, column indices, and non-zero values of the matrix.
 * 
 * @tparam value_t The value type of the matrix elements.
 * @tparam index_t The index type used for column indices.
 * @tparam offset_t The offset type used for row offsets.
 * @param fname The name of the file containing the matrix in CSR format.
 * @return The CSR object representing the matrix.
 * @throws std::runtime_error if the file fails to open.
 */
template <typename value_t, typename index_t, typename offset_t>
sygraph::formats::CSR<value_t, index_t, offset_t> from_matrix(const std::string& fname) {
  std::ifstream ifs(fname);
  if (!ifs.is_open()) {
    throw std::runtime_error("Failed to open file: " + fname);
  }
  return from_matrix<value_t, index_t, offset_t>(ifs);
}

} // namespace csr
} // namespace io
} // namespace v0
} // namespace sygraph
