#pragma once
#include <sycl/sycl.hpp>
#include <memory>
#include <vector>

#include <sygraph/utils/types.hpp>
#include <sygraph/utils/memory.hpp>

namespace sygraph {
inline namespace v0 {
namespace formats {


template <typename ValueT,
          typename IndexT = types::index_t,
          typename OffsetT = types::offset_t>
class COO {
public:
  using value_t = ValueT;
  using index_t = IndexT;
  using offset_t = OffsetT;

  COO(std::vector<index_t> row_indices,
      std::vector<index_t> column_indices,
      std::vector<value_t> nnz_values) : 
    row_indices(row_indices),
    column_indices(column_indices),
    nnz_values(nnz_values) { }

  COO(size_t values) {
    row_indices.reserve(values);
    column_indices.reserve(values);
    nnz_values.reserve(values);
  }

  ~COO() = default;

  //Getters
  const std::vector<offset_t>& get_row_indices() const { return row_indices; }
  const std::vector<index_t>& get_column_indices() const { return column_indices; }
  const std::vector<value_t>& get_values() const { return nnz_values; }
  std::vector<offset_t>& get_row_indices() { return row_indices; }
  std::vector<index_t>& get_column_indices() { return column_indices; }
  std::vector<value_t>& get_values() { return nnz_values; }
  const size_t get_size() const { return row_indices.size(); }


  // static methods


private:

  std::vector<index_t> row_indices;
  std::vector<index_t> column_indices;
  std::vector<value_t> nnz_values;  
};

} // namespace formats
} // namespace v0
} // namespace sygraph