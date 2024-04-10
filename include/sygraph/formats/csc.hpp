#pragma once

#include <sycl/sycl.hpp>
#include <memory>

namespace sygraph {
inline namespace v0 {
namespace formats {

template <typename index_t,
          typename offset_t,
          typename value_t>
class CSC {
private:
  sycl::queue& q;
  index_t n_rows;
  index_t n_cols;
  offset_t n_nonzeros;

  index_t* row_indices;
  offset_t* column_offsets;
  value_t* nonzero_values;

  void build() {
    this->row_indices = sycl::malloc_shared<index_t>(n_nonzeros, q);
    this->column_offsets = sycl::malloc_shared<offset_t>(n_cols + 1, q);
    this->nonzero_values = sycl::malloc_shared<value_t>(n_nonzeros, q);
  }
public:
  CSC(sycl::queue& q, index_t n_rows, index_t n_cols, offset_t n_nonzeros)
    : q(q), n_rows(n_rows), n_cols(n_cols), n_nonzeros(n_nonzeros) {
    build();
  }

  CSC(sycl::queue& q, index_t n_cols, offset_t n_nonzeros)
    : csr_t(q, 0, n_cols, n_nonzeros) {}

  CSC(sycl::queue& q) : csc_t(q, 0, 0, 0) {}

  CSC(const csc_t& other)
    : q(q), n_rows(other.n_rows), n_cols(other.n_cols), n_nonzeros(other.n_nonzeros) {
    build();
  }

  CSC(csc_t&& other) = default;

  ~CSC() {
    sycl::free(this->row_indices, q);
    sycl::free(this->column_offsets, q);
    sycl::free(this->nonzero_values, q);
  };

  index_t* get_row_indices() {
    return row_indices;
  }

  offset_t* get_column_offsets() {
    return column_offsets;
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

  nz_size_t get_n_nonzeros() const {
    return n_nonzeros;
  }
};

} // namespace formats 
} // inline namespace v0
} // namespace sygraph