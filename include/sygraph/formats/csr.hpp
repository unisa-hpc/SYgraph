#pragma once
#include <sycl/sycl.hpp>
#include <memory>
#include <vector>

namespace sygraph {
inline namespace v0 {
namespace formats {

template <typename index_t,
          typename offset_t,
          typename value_t>
class CSR {
private:
  index_t n_rows;
  index_t n_cols;
  offset_t n_nonzeros;

  offset_t* row_offsets;
  index_t* column_indices;
  value_t* nonzero_values;

  void build() {
    this->row_offsets = sycl::malloc_shared<offset_t>(n_rows + 1, q);
    this->column_indices = sycl::malloc_shared<index_t>(n_nonzeros, q);
    this->nonzero_values = sycl::malloc_shared<value_t>(n_nonzeros, q);
  }
  
public:
  CSR(sycl::queue& q, index_t n_rows, index_t n_cols, offset_t n_nonzeros)
    : q(q), n_rows(n_rows), n_cols(n_cols), n_nonzeros(n_nonzeros) {
    build();   
  }
  
  CSR(sycl::queue& q, index_t n_rows, offset_t n_nonzeros)
    : csr_t(q, n_rows, 0, n_nonzeros) {}
  
  CSR(sycl::queue& q) : csr_t(q, 0, 0, 0) {}

  CSR(const csr_t& other)
    : n_rows(other.n_rows), n_cols(other.n_cols), n_nonzeros(other.n_nonzeros) {
    build();
  }

  CSR(csr_t&& other) = default;

  ~CSR() {
    sycl::free(this->row_offsets, q);
    sycl::free(this->column_indices, q);
    sycl::free(this->nonzero_values, q);
  };

  offset_t* get_row_offsets() {
    return row_offsets;
  }

  index_t* get_column_indices() {
    return column_indices;
  }

  value_t* get_nonzero_values() {
    return nonzero_values;
  }

  index_t get_n_rows() const {
    return n_rows;
  }

  index_t get_n_cols() const {
    return n_cols;
  }

  offset_t get_n_nonzeros() const {
    return n_nonzeros;
  }
};

} // namespace formats
} // namespace v0
} // namespace sygraph