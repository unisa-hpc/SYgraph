#pragma once

#include <sycl/sycl.hpp>
#include <memory>

namespace sygraph {
inline namespace v0 {
namespace formats {

template <typename index_t,
          typename nz_size_t,
          typename value_t>
class coo_t {
private:
  index_t n_rows;
  index_t n_cols;
  nz_size_t n_nonzeros;

  std::shared_ptr<sycl::buffer<index_t, 1>> row_indices;
  std::shared_ptr<sycl::buffer<index_t, 1>> column_indices;
  std::shared_ptr<sycl::buffer<value_t, 1>> nonzero_values;

  void build() {
    this->row_indices = std::make_shared<sycl::buffer<index_t, 1>>(sycl::range<1>(n_nonzeros));
    this->column_indices = std::make_shared<sycl::buffer<index_t, 1>>(sycl::range<1>(n_nonzeros));
    this->nonzero_values = std::make_shared<sycl::buffer<value_t, 1>>(sycl::range<1>(n_nonzeros));
  }
public:

  coo_t(index_t n_rows, index_t n_cols, nz_size_t n_nonzeros)
    : n_rows(n_rows), n_cols(n_cols), n_nonzeros(n_nonzeros) {
    build();
  }

  coo_t(nz_size_t n_nonzeros)
    : coo_t(0, 0, n_nonzeros) {}

  coo_t() : coo_t(0, 0, 0) {}

  coo_t(const coo_t& other)
    : n_rows(other.n_rows), n_cols(other.n_cols), n_nonzeros(other.n_nonzeros) {
    build();
  }

  coo_t(coo_t&& other) = default;

  ~coo_t() {};

  sycl::buffer<index_t, 1>& get_row_indices() {
    return *row_indices;
  }

  sycl::buffer<index_t, 1>& get_column_indices() {
    return *column_indices;
  }

  sycl::buffer<value_t, 1>& get_nonzero_values() {
    return *nonzero_values;
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