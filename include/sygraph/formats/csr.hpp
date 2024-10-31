#pragma once
#include <memory>
#include <sycl/sycl.hpp>
#include <vector>

#include <sygraph/utils/memory.hpp>
#include <sygraph/utils/types.hpp>

namespace sygraph {
inline namespace v0 {
namespace formats {


/**
 * @class CSR
 * @brief Compressed Sparse Row (CSR) matrix format.
 *
 * This class represents a sparse matrix in CSR format, which is efficient for matrix-vector multiplication.
 *
 * @tparam ValueT Type of the non-zero values in the matrix.
 * @tparam IndexT Type of the indices (default is types::index_t).
 * @tparam OffsetT Type of the offsets (default is types::offset_t).
 *
 * The CSR format stores the matrix in three separate arrays:
 * - _row_offsets: Array of size (n_rows + 1) that stores the starting index of each row in the _column_indices and _nnz_values arrays.
 * - _column_indices: Array that stores the column indices of the non-zero values.
 * - _nnz_values: Array that stores the non-zero values of the matrix.
 */
template<typename ValueT, typename IndexT = types::index_t, typename OffsetT = types::offset_t>
class CSR {
public:
  /**
   * @brief Default constructor.
   *
   * Creates an empty CSR matrix.
   */
  CSR() = default;


  /**
   * @brief Constructs a CSR (Compressed Sparse Row) matrix.
   *
   * @param row_offsets A vector containing the row offsets.
   * @param column_indices A vector containing the column indices.
   * @param nnz_values A vector containing the non-zero values.
   */
  CSR(std::vector<OffsetT> row_offsets, std::vector<IndexT> column_indices, std::vector<ValueT> nnz_values)
      : _row_offsets(row_offsets), _column_indices(column_indices), _nnz_values(nnz_values) {}

  /**
   * @brief Constructor for the CSR (Compressed Sparse Row) class.
   *
   * This constructor initializes the CSR matrix with the given number of rows and non-zero elements.
   * It resizes the internal vectors to accommodate the specified number of rows and non-zero values.
   *
   * @param n_rows The number of rows in the matrix.
   * @param n_nonzeros The number of non-zero elements in the matrix.
   */
  CSR(IndexT n_rows, OffsetT n_nonzeros) {
    _row_offsets.resize(n_rows + 1);
    _column_indices.resize(n_nonzeros);
    _nnz_values.resize(n_nonzeros);
  }

  /**
   * @brief Default destructor for the CSR class.
   */
  ~CSR() = default;

  // Getters
  /**
   * @brief Get the size of the row offsets.
   *
   * This function returns the size of the row offsets array,
   * which is the number of rows in the CSR (Compressed Sparse Row) format matrix.
   *
   * @return IndexT The size of the row offsets array minus one.
   */
  IndexT getRowOffsetsSize() const { return _row_offsets.size() - 1; }

  /**
   * @brief Returns the number of non-zero elements in the CSR (Compressed Sparse Row) matrix.
   *
   * This function calculates the number of non-zero elements by returning the size of the
   * column_indices vector, which stores the column indices of the non-zero elements.
   *
   * @return OffsetT The number of non-zero elements in the CSR matrix.
   */
  OffsetT getNumNonzeros() const { return _column_indices.size(); }

  /**
   * @brief Retrieves the row offsets of the CSR (Compressed Sparse Row) format.
   *
   * This function returns a constant reference to the vector containing the row offsets.
   * The row offsets vector indicates the starting index of each row in the values array.
   *
   * @return const std::vector<OffsetT>& A constant reference to the vector of row offsets.
   */
  const std::vector<OffsetT>& getRowOffsets() const { return _row_offsets; }

  /**
   * @brief Retrieves the row offsets of the CSR (Compressed Sparse Row) format.
   *
   * This function returns a reference to the vector containing the row offsets.
   * The row offsets vector indicates the starting index of each row in the values array.
   *
   * @return std::vector<OffsetT>& A constant reference to the vector of row offsets.
   */
  std::vector<OffsetT>& getRowOffsets() { return _row_offsets; }

  /**
   * @brief Retrieves the column indices of the CSR (Compressed Sparse Row) format.
   *
   * @return A constant reference to a vector containing the column indices.
   */
  const std::vector<IndexT>& getColumnIndices() const { return _column_indices; }

  /**
   * @brief Retrieves the column indices of the CSR (Compressed Sparse Row) format.
   *
   * @return A reference to a vector containing the column indices.
   */
  std::vector<IndexT>& getColumnIndices() { return _column_indices; }

  /**
   * @brief Retrieves the non-zero values of the CSR (Compressed Sparse Row) format.
   *
   * @return A constant reference to a vector containing the non-zero values.
   */
  const std::vector<ValueT>& getValues() const { return _nnz_values; }

  /**
   * @brief Retrieves the non-zero values of the CSR (Compressed Sparse Row) format.
   *
   * @return A reference to a vector containing the non-zero values.
   */
  std::vector<ValueT>& getValues() { return _nnz_values; }

  // Setters

  /**
   * @brief Sets the row offsets of the CSR (Compressed Sparse Row) format.
   *
   * This function sets the row offsets of the CSR matrix to the specified vector of offsets.
   *
   * @param offsets A vector containing the row offsets.
   */
  void setRowOffsets(const std::vector<OffsetT>& offsets) { _row_offsets = offsets; }

  /**
   * @brief Sets the column indices of the CSR (Compressed Sparse Row) format.
   *
   * This function sets the column indices of the CSR matrix to the specified vector of indices.
   *
   * @param indices A vector containing the column indices.
   */
  void setColumnIndices(const std::vector<IndexT>& indices) { _column_indices = indices; }

  /**
   * @brief Sets the non-zero values of the CSR (Compressed Sparse Row) format.
   *
   * This function sets the non-zero values of the CSR matrix to the specified vector of values.
   *
   * @param values A vector containing the non-zero values.
   */
  void setNnzValues(const std::vector<ValueT>& values) { _nnz_values = values; }

private:
  std::vector<OffsetT> _row_offsets;
  std::vector<IndexT> _column_indices;
  std::vector<ValueT> _nnz_values;
};

} // namespace formats
} // namespace v0
} // namespace sygraph