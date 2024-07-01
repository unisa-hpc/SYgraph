#include <fstream>
#include <iostream>
#include <sstream>

#include <sycl/sycl.hpp>

#include <sygraph/formats/coo.hpp>
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
 * @tparam ValueT The value type of the matrix elements.
 * @tparam IndexT The index type used for column indices.
 * @tparam OffsetT The offset type used for row offsets.
 * @param iss The input stream containing the matrix in CSR format.
 * @return The CSR object representing the matrix.
 */
template<typename ValueT, typename IndexT, typename OffsetT>
sygraph::formats::CSR<ValueT, IndexT, OffsetT> fromMatrix(std::istream& iss) {
  size_t n_rows = 0;
  size_t n_nonzeros = 0;
  std::vector<OffsetT> row_offsets;
  std::vector<IndexT> column_indices;
  std::vector<ValueT> nnz_values;

  // Read number of rows
  iss >> n_rows;

  row_offsets.push_back(0);

  // Read adjacency matrix
  for (int i = 0; i < n_rows; ++i) {
    for (int j = 0; j < n_rows; ++j) {
      ValueT value;
      iss >> value;
      if (value != static_cast<ValueT>(0)) {
        nnz_values.push_back(value);
        column_indices.push_back(j);
      }
    }
    row_offsets.push_back(nnz_values.size());
  }

  return sygraph::formats::CSR<ValueT, IndexT, OffsetT>(row_offsets, column_indices, nnz_values);
}

/**
 * @brief Converts a matrix in CSR format from a file to a CSR object.
 *
 * This function reads a matrix in CSR format from a given file and converts it into a CSR object.
 * The CSR object contains the row offsets, column indices, and non-zero values of the matrix.
 *
 * @tparam ValueT The value type of the matrix elements.
 * @tparam IndexT The index type used for column indices.
 * @tparam OffsetT The offset type used for row offsets.
 * @param fname The name of the file containing the matrix in CSR format.
 * @return The CSR object representing the matrix.
 * @throws std::runtime_error if the file fails to open.
 */
template<typename ValueT, typename IndexT, typename OffsetT>
sygraph::formats::CSR<ValueT, IndexT, OffsetT> fromCSR(std::istream& iss) {
  size_t n_rows = 0;
  size_t n_nonzeros = 0;
  std::vector<OffsetT> row_offsets;
  std::vector<IndexT> column_indices;
  std::vector<ValueT> nnz_values;

  // Read number of rows
  iss >> n_rows;

  row_offsets.push_back(0);

  // Read row offsets
  for (int i = 0; i < n_rows + 1; ++i) {
    OffsetT offset;
    iss >> offset;
    row_offsets.push_back(offset);
  }

  // Read column indices
  for (int i = 0; i < row_offsets.back(); ++i) {
    IndexT index;
    iss >> index;
    column_indices.push_back(index);
  }

  // Read non-zero values
  for (int i = 0; i < row_offsets.back(); ++i) {
    ValueT value;
    iss >> value;
    nnz_values.push_back(value);
  }

  return sygraph::formats::CSR<ValueT, IndexT, OffsetT>(row_offsets, column_indices, nnz_values);
}

template<typename ValueT, typename IndexT, typename OffsetT>
sygraph::formats::CSR<ValueT, IndexT, OffsetT> fromCOO(const sygraph::formats::COO<ValueT, IndexT, OffsetT>& coo) {
  auto coo_row_indices = coo.getRowIndices();
  auto coo_column_indices = coo.getColumnIndices();
  auto coo_values = coo.getValues();
  auto size = coo.getSize();
  auto n_nodes = std::max(*std::max_element(coo_row_indices.begin(), coo_row_indices.end()),
                          *std::max_element(coo_column_indices.begin(), coo_column_indices.end()))
                 + 1;

  std::vector<OffsetT> csr_row_offsets(n_nodes + 1);
  std::vector<IndexT> csr_column_indices(size);
  std::vector<ValueT> csr_values(size);

  // Count the number of nonzeros in each row
  for (IndexT i = 0; i < size; i++) { csr_row_offsets[coo_row_indices[i]]++; }

  // Compute the prefix sum
  OffsetT sum = 0;
  for (IndexT i = 0; i < n_nodes; i++) {
    OffsetT temp = csr_row_offsets[i];
    csr_row_offsets[i] = sum;
    sum += temp;
  }
  csr_row_offsets[n_nodes] = sum;

  // Fill the CSR matrix
  for (IndexT i = 0; i < size; i++) {
    IndexT row = coo_row_indices[i];
    OffsetT offset = csr_row_offsets[row];
    csr_column_indices[offset] = coo_column_indices[i];
    csr_values[offset] = coo_values[i];
    csr_row_offsets[row]++;
  }

  // Shift the row offsets
  for (IndexT i = n_nodes - 1; i > 0; i--) { csr_row_offsets[i] = csr_row_offsets[i - 1]; }
  csr_row_offsets[0] = 0;

  return {csr_row_offsets, csr_column_indices, csr_values};
}

template<typename ValueT, typename IndexT, typename OffsetT>
void toBinary(const sygraph::formats::CSR<ValueT, IndexT, OffsetT>& csr, std::ostream& oss) {
  if (!oss) { throw std::runtime_error("Failed to write binary CSR matrix"); }

  auto& row_offsets = csr.getRowOffsets();
  auto& column_indices = csr.getColumnIndices();
  auto& values = csr.getValues();

  size_t num_rows = row_offsets.size();
  size_t num_nonzero = column_indices.size();

  oss.write(reinterpret_cast<const char*>(&num_rows), sizeof(size_t));
  oss.write(reinterpret_cast<const char*>(&num_nonzero), sizeof(size_t));

  oss.write(reinterpret_cast<const char*>(&row_offsets[0]), row_offsets.size() * sizeof(OffsetT));
  oss.write(reinterpret_cast<const char*>(&column_indices[0]), column_indices.size() * sizeof(IndexT));
  oss.write(reinterpret_cast<const char*>(&values[0]), values.size() * sizeof(ValueT));
}

template<typename ValueT, typename IndexT, typename OffsetT>
sygraph::formats::CSR<ValueT, IndexT, OffsetT> fromBinary(std::istream& iss) {
  if (!iss) { throw std::runtime_error("Failed to read binary CSR matrix"); }

  size_t num_rows;
  size_t num_nonzero;

  iss.read(reinterpret_cast<char*>(&num_rows), sizeof(size_t));
  iss.read(reinterpret_cast<char*>(&num_nonzero), sizeof(size_t));

  std::vector<OffsetT> row_ptr(num_rows);
  std::vector<IndexT> col_indices(num_nonzero);
  std::vector<ValueT> values(num_nonzero);

  iss.read(reinterpret_cast<char*>(&row_ptr[0]), row_ptr.size() * sizeof(OffsetT));
  iss.read(reinterpret_cast<char*>(&col_indices[0]), col_indices.size() * sizeof(IndexT));
  iss.read(reinterpret_cast<char*>(&values[0]), values.size() * sizeof(ValueT));

  return {row_ptr, col_indices, values};
}
} // namespace csr
} // namespace io
} // namespace v0
} // namespace sygraph
