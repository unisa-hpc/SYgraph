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
class COO {
public:
  using value_t = ValueT;
  using index_t = IndexT;
  using offset_t = OffsetT;

  COO(std::vector<index_t> row_indices, std::vector<index_t> column_indices, std::vector<value_t> nnz_values)
      : _row_indices(row_indices), _column_indices(column_indices), _nnz_values(nnz_values) {}

  COO(size_t values) {
    _row_indices.reserve(values);
    _column_indices.reserve(values);
    _nnz_values.reserve(values);
  }

  ~COO() = default;

  // Getters
  const std::vector<offset_t>& getRowIndices() const { return _row_indices; }
  const std::vector<index_t>& getColumnIndices() const { return _column_indices; }
  const std::vector<value_t>& getValues() const { return _nnz_values; }
  std::vector<offset_t>& getRowIndices() { return _row_indices; }
  std::vector<index_t>& getColumnIndices() { return _column_indices; }
  std::vector<value_t>& getValues() { return _nnz_values; }
  size_t getSize() const { return _row_indices.size(); }


  // static methods


private:
  std::vector<index_t> _row_indices;
  std::vector<index_t> _column_indices;
  std::vector<value_t> _nnz_values;
};

} // namespace formats
} // namespace v0
} // namespace sygraph