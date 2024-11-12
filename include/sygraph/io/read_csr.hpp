#include <fstream>
#include <iostream>
#include <sstream>

#include <sycl/sycl.hpp>

#include <sygraph/formats/coo.hpp>
#include <sygraph/formats/csr.hpp>
#include <sygraph/io/matrix_market.hpp>

namespace sygraph {
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
 * @brief Reads a Matrix Market file in coordinate format and converts it to a CSR matrix.
 *
 * This function parses a Matrix Market file from an input stream and converts it to a CSR (Compressed Sparse Row)
 * matrix format. The function only supports `coordinate` format and `general` or `symmetric` symmetry.
 * If the matrix is symmetric, both (row, col) and (col, row) entries are stored for each non-diagonal entry.
 *
 * @tparam ValueT Type of the non-zero values in the matrix.
 * @tparam IndexT Type of the indices (default is int).
 * @tparam OffsetT Type of the offsets (default is int).
 *
 * @param iss Input stream containing the Matrix Market data.
 * @return CSR<ValueT, IndexT, OffsetT> An instance of the CSR class containing the matrix data in CSR format.
 *
 * @throws std::runtime_error if the Matrix Market format or symmetry type is unsupported.

 * @note The function expects the input to be in Matrix Market coordinate format.
 *       It does not support array format or field types other than real numbers.
 */
template<typename ValueT, typename IndexT, typename OffsetT>
sygraph::formats::CSR<ValueT, IndexT, OffsetT> fromMM(std::istream& iss) {
  sygraph::io::detail::mm::Banner banner;

  size_t rows = 0, cols = 0, nnz = 0;
  std::vector<std::tuple<IndexT, IndexT, ValueT>> entries;

  // Read matrix dimensions and non-zero count
  std::string line;
  bool dimensions_read = false;
  bool banner_read = false;

  while (std::getline(iss, line)) {
    if (line[0] == '%') {
      if (!banner_read) {
        banner_read = true;
        banner.read(line);
        banner.validate<ValueT, IndexT, OffsetT>();
      }
      continue;
    }; // Skip comments
    std::istringstream line_stream(line);

    if (!dimensions_read) {
      line_stream >> rows >> cols >> nnz;
      dimensions_read = true;
    } else {
      IndexT row, col;
      ValueT value;
      if (banner.isPattern()) {
        line_stream >> row >> col;
        value = static_cast<ValueT>(1);
      } else {
        line_stream >> row >> col >> value;
      }

      entries.emplace_back(row - 1, col - 1, value);

      // For symmetric matrices, also add the transpose entry if not on the diagonal
      if (banner.isSymmetric()) { entries.emplace_back(col - 1, row - 1, value); }
    }
  }

  // Sort entries by row, then by column
  std::sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) {
    return std::get<0>(a) < std::get<0>(b) || (std::get<0>(a) == std::get<0>(b) && std::get<1>(a) < std::get<1>(b));
  });

  // Initialize CSR vectors
  std::vector<OffsetT> row_offsets(rows + 1, 0);
  std::vector<IndexT> column_indices(entries.size());
  std::vector<ValueT> nnz_values(entries.size());

  // Count non-zero elements per row for row_offsets
  for (const auto& entry : entries) { row_offsets[std::get<0>(entry) + 1]++; }

  // Accumulate counts to get row offsets
  for (IndexT i = 1; i <= rows; ++i) { row_offsets[i] += row_offsets[i - 1]; }

  // Fill in column indices and values arrays
  std::vector<OffsetT> row_position(rows, 0);
  for (const auto& entry : entries) {
    IndexT row = std::get<0>(entry);
    IndexT col = std::get<1>(entry);
    ValueT value = std::get<2>(entry);

    OffsetT pos = row_offsets[row] + row_position[row];
    column_indices[pos] = col;
    nnz_values[pos] = value;
    row_position[row]++;
  }

  return sygraph::formats::CSR<ValueT, IndexT, OffsetT>(row_offsets, column_indices, nnz_values);
}

/**
 * @brief Reads a Matrix Market file in coordinate format and converts it to a CSR matrix.
 *
 * This function parses a Matrix Market file from an input stream and converts it to a CSR (Compressed Sparse Row)
 * matrix format. The function only supports `coordinate` format and `general` or `symmetric` symmetry.
 * If the matrix is symmetric, both (row, col) and (col, row) entries are stored for each non-diagonal entry.
 *
 * @tparam ValueT Type of the non-zero values in the matrix.
 * @tparam IndexT Type of the indices (default is int).
 * @tparam OffsetT Type of the offsets (default is int).
 *
 * @param filename Name of the file containing the Matrix Market data.
 * @return CSR<ValueT, IndexT, OffsetT> An instance of the CSR class containing the matrix data in CSR format.
 *
 * @throws std::runtime_error if the Matrix Market format or symmetry type is unsupported.

 * @note The function expects the input to be in Matrix Market coordinate format.
 *       It does not support array format or field types other than real numbers.
 */
template<typename ValueT, typename IndexT, typename OffsetT>
sygraph::formats::CSR<ValueT, IndexT, OffsetT> fromMM(const std::string& filename) {
  std::ifstream file(filename);
  if (!file.is_open()) { throw std::runtime_error("Failed to open file: " + filename); }

  return fromMM<ValueT, IndexT, OffsetT>(file);
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

/**
 * @brief Converts a COO (Coordinate List) formatted sparse matrix to CSR (Compressed Sparse Row) format.
 *
 * @tparam ValueT The type of the values in the matrix.
 * @tparam IndexT The type of the indices in the matrix.
 * @tparam OffsetT The type of the offsets in the CSR format.
 * @param coo The input COO formatted sparse matrix.
 * @return A CSR formatted sparse matrix.
 *
 * This function takes a sparse matrix in COO format and converts it to CSR format.
 * The COO format stores the matrix as a list of (row, column, value) tuples,
 * while the CSR format uses three arrays: one for row offsets, one for column indices,
 * and one for values. The conversion involves counting the number of non-zero elements
 * in each row, computing prefix sums to determine row offsets, and then filling the
 * CSR arrays with the appropriate values and column indices.
 */
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

/**
 * @brief Serializes a CSR (Compressed Sparse Row) matrix to a binary stream.
 *
 * This function writes the CSR matrix data to the provided output stream in binary format.
 * The CSR matrix is represented by its row offsets, column indices, and values arrays.
 *
 * @tparam ValueT The type of the values in the CSR matrix.
 * @tparam IndexT The type of the column indices in the CSR matrix.
 * @tparam OffsetT The type of the row offsets in the CSR matrix.
 * @param csr The CSR matrix to be serialized.
 * @param oss The output stream to which the CSR matrix will be written.
 *
 * @throws std::runtime_error If the output stream is not in a good state.
 */
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

/**
 * @brief Reads a CSR (Compressed Sparse Row) matrix from a binary input stream.
 *
 * This function reads the number of rows and non-zero elements from the binary
 * input stream, followed by the row pointers, column indices, and values arrays.
 * It then constructs and returns a CSR matrix using these arrays.
 *
 * @tparam ValueT The type of the values in the CSR matrix.
 * @tparam IndexT The type of the column indices in the CSR matrix.
 * @tparam OffsetT The type of the row pointers in the CSR matrix.
 * @param iss The input stream to read the binary data from.
 * @return A CSR matrix containing the data read from the input stream.
 * @throws std::runtime_error If the input stream is not valid or if reading fails.
 */
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
} // namespace sygraph
