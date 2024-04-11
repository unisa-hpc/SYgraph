#pragma once
#include <sycl/sycl.hpp>
#include <memory>
#include <vector>

#include <sygraph/utils/types.hpp>
#include <sygraph/utils/memory.hpp>

namespace sygraph {
inline namespace v0 {
namespace formats {


template <typename value_t,
          typename index_t = types::index_t, 
          typename offset_t = types::offset_t>
class CSR {
public:
  CSR(index_t n_rows, offset_t n_nonzeros)
    : n_rows(n_rows), n_nonzeros(n_nonzeros) {
    row_offsets.resize(n_rows + 1);
    column_indices.resize(n_nonzeros);
    nnz_values.resize(n_nonzeros);  
  }

  ~CSR() = default;


private:
  index_t n_rows;
  offset_t n_nonzeros;

  std::vector<offset_t> row_offsets;
  std::vector<index_t> column_indices;
  std::vector<value_t> nnz_values;  
};

} // namespace formats
} // namespace v0
} // namespace sygraph