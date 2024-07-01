#pragma once
#include <memory>
#include <sycl/sycl.hpp>
#include <vector>

#include <sygraph/utils/memory.hpp>
#include <sygraph/utils/types.hpp>

namespace sygraph {
inline namespace v0 {
namespace formats {


template<typename ValueT, typename IndexT = types::index_t, typename OffsetT = types::offset_t>
class CSR {
public:
  CSR() = default;

  CSR(std::vector<OffsetT> row_offsets, std::vector<IndexT> column_indices, std::vector<ValueT> nnz_values)
      : _row_offsets(row_offsets), _column_indices(column_indices), _nnz_values(nnz_values) {}

  CSR(IndexT n_rows, OffsetT n_nonzeros) {
    _row_offsets.resize(n_rows + 1);
    _column_indices.resize(n_nonzeros);
    _nnz_values.resize(n_nonzeros);
  }

  ~CSR() = default;

  // Getters
  IndexT getRowOffsetsSize() const { return _row_offsets.size() - 1; }

  OffsetT getNumNonzeros() const { return _column_indices.size(); }

  const std::vector<OffsetT>& getRowOffsets() const { return _row_offsets; }
  std::vector<OffsetT>& getRowOffsets() { return _row_offsets; }

  const std::vector<IndexT>& getColumnIndices() const { return _column_indices; }
  std::vector<IndexT>& getColumnIndices() { return _column_indices; }

  const std::vector<ValueT>& getValues() const { return _nnz_values; }
  std::vector<ValueT>& getValues() { return _nnz_values; }

  void setRowOffsets(const std::vector<OffsetT>& offsets) { _row_offsets = offsets; }

  void setColumnIndices(const std::vector<IndexT>& indices) { _column_indices = indices; }

  void setNnzValues(const std::vector<ValueT>& values) { _nnz_values = values; }

private:
  std::vector<OffsetT> _row_offsets;
  std::vector<IndexT> _column_indices;
  std::vector<ValueT> _nnz_values;
};

} // namespace formats
} // namespace v0
} // namespace sygraph