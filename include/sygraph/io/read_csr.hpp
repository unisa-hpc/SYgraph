#include <iostream>
#include <fstream>
#include <sstream>

#include <sycl/sycl.hpp>

#include <sygraph/formats/csr.hpp>
#include <sygraph/formats/coo.hpp>

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
sygraph::formats::CSR<value_t, index_t, offset_t> from_csr(std::istream& iss) {
  size_t n_rows = 0;
  size_t n_nonzeros = 0;
  std::vector<offset_t> row_offsets;
  std::vector<index_t> column_indices;
  std::vector<value_t> nnz_values;

  // Read number of rows
  iss >> n_rows;

  row_offsets.push_back(0);

  // Read row offsets
  for (int i = 0; i < n_rows + 1; ++i) {
    offset_t offset;
    iss >> offset;
    row_offsets.push_back(offset);
  }

  // Read column indices
  for (int i = 0; i < row_offsets.back(); ++i) {
    index_t index;
    iss >> index;
    column_indices.push_back(index);
  }

  // Read non-zero values
  for (int i = 0; i < row_offsets.back(); ++i) {
    value_t value;
    iss >> value;
    nnz_values.push_back(value);
  }

  return sygraph::formats::CSR<value_t, index_t, offset_t>(row_offsets, column_indices, nnz_values);
}

template <typename value_t, typename index_t, typename offset_t>
sygraph::formats::CSR<value_t, index_t, offset_t> from_coo(const sygraph::formats::COO<value_t, index_t, offset_t>& coo) {
  auto coo_row_indices = coo.get_row_indices();
  auto coo_column_indices = coo.get_column_indices();
  auto coo_values = coo.get_values();
  auto size = coo.get_size();
  auto n_nodes = std::max(*std::max_element(coo_row_indices.begin(), coo_row_indices.end()),
                          *std::max_element(coo_column_indices.begin(), coo_column_indices.end())) + 1;

  std::vector<offset_t> csr_row_offsets(n_nodes + 1);
  std::vector<index_t> csr_column_indices(size);
  std::vector<value_t> csr_values(size);

  // Count the number of nonzeros in each row
  for (index_t i = 0; i < size; i++) {
    csr_row_offsets[coo_row_indices[i]]++;
  }

  // Compute the prefix sum
  offset_t sum = 0;
  for (index_t i = 0; i < n_nodes; i++) {
    offset_t temp = csr_row_offsets[i];
    csr_row_offsets[i] = sum;
    sum += temp;
  }
  csr_row_offsets[n_nodes] = sum;

  // Fill the CSR matrix
  for (index_t i = 0; i < size; i++) {
    index_t row = coo_row_indices[i];
    offset_t offset = csr_row_offsets[row];
    csr_column_indices[offset] = coo_column_indices[i];
    csr_values[offset] = coo_values[i];
    csr_row_offsets[row]++;
  }

  // Shift the row offsets
  for (index_t i = n_nodes - 1; i > 0; i--) {
    csr_row_offsets[i] = csr_row_offsets[i - 1];
  }
  csr_row_offsets[0] = 0;

  return {csr_row_offsets, csr_column_indices, csr_values};
}

template <typename value_t, typename index_t, typename offset_t>
void to_binary(const sygraph::formats::CSR<value_t, index_t, offset_t>& csr, std::ostream& oss) {
  if (!oss) {
    throw std::runtime_error("Failed to write binary CSR matrix");
  }

  auto& row_offsets = csr.get_row_offsets();
  auto& column_indices = csr.get_column_indices();
  auto& values = csr.get_values();

  size_t num_rows = row_offsets.size();
  size_t num_nonzero = column_indices.size();
  std::cout << "WRITING: num_rows: " << num_rows << " num_nonzero: " << num_nonzero << "\n";

  oss.write(reinterpret_cast<const char*>(&num_rows), sizeof(size_t));
  oss.write(reinterpret_cast<const char*>(&num_nonzero), sizeof(size_t));

  oss.write(reinterpret_cast<const char*>(&row_offsets[0]), row_offsets.size() * sizeof(offset_t));
  oss.write(reinterpret_cast<const char*>(&column_indices[0]), column_indices.size() * sizeof(index_t));
  oss.write(reinterpret_cast<const char*>(&values[0]), values.size() * sizeof(value_t));
}

template <typename value_t, typename index_t, typename offset_t>
sygraph::formats::CSR<value_t, index_t, offset_t> from_binary(std::istream& iss) {
  if (!iss) {
    throw std::runtime_error("Failed to read binary CSR matrix");
  }

  size_t num_rows;
  size_t num_nonzero;

  iss.read(reinterpret_cast<char*>(&num_rows), sizeof(size_t));
  iss.read(reinterpret_cast<char*>(&num_nonzero), sizeof(size_t));
  std::cout << "READING: num_rows: " << num_rows << " num_nonzero: " << num_nonzero << "\n";

  std::vector<offset_t> row_ptr(num_rows);
  std::vector<index_t> col_indices(num_nonzero);
  std::vector<value_t> values(num_nonzero);

  iss.read(reinterpret_cast<char*>(&row_ptr[0]), row_ptr.size() * sizeof(offset_t));
  iss.read(reinterpret_cast<char*>(&col_indices[0]), col_indices.size() * sizeof(index_t));
  iss.read(reinterpret_cast<char*>(&values[0]), values.size() * sizeof(value_t));

  return {row_ptr, col_indices, values};
}
} // namespace csr
} // namespace io
} // namespace v0
} // namespace sygraph
