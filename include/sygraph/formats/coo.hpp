#pragma once

#include <sycl/sycl.hpp>
#include <memory>

namespace sygraph {
inline namespace v0 {
namespace formats {

template <typename index_t,
          typename nz_size_t,
          typename value_t>
class COO {
private:
  sycl::queue& q;

  index_t n_rows;
  index_t n_cols;
  nz_size_t n_nonzeros;

  index_t* row_indices;
  index_t* column_indices;
  value_t* nonzero_values;

  void build() {
    if (n_nonzeros == 0) {
      return;
    }
    this->row_indices = sycl::malloc_shared<index_t>(n_nonzeros, q);
    this->column_indices = sycl::malloc_shared<index_t>(n_nonzeros, q);
    this->nonzero_values = sycl::malloc_shared<value_t>(n_nonzeros, q);
  }
public:

  COO(sycl::queue& q, index_t n_rows, index_t n_cols, nz_size_t n_nonzeros)
    : q(q), n_rows(n_rows), n_cols(n_cols), n_nonzeros(n_nonzeros) {
    build();
  }

  COO(sycl::queue& q, nz_size_t n_nonzeros)
    : coo_t(q, 0, 0, n_nonzeros) {}

  COO(sycl::queue& q) : coo_t(q, 0, 0, 0) {}

  COO(const coo_t& other)
    : q(q), n_rows(other.n_rows), n_cols(other.n_cols), n_nonzeros(other.n_nonzeros) {
    build();
  }

  COO(coo_t&& other) = default;

  ~COO() {
    sycl::free(this->row_indices, q);
    sycl::free(this->column_indices, q);
    sycl::free(this->nonzero_values, q);
  };

  index_t* get_row_indices() {
    return row_indices;
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

  nz_size_t get_n_nonzeros() const {
    return n_nonzeros;
  }
};

} // namespace formats
} // inline namespace v0
} // namespace sygraph