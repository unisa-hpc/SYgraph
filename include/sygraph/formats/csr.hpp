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
class csr_t {
private:
  index_t n_rows;
  index_t n_cols;
  offset_t n_nonzeros;

  std::shared_ptr<sycl::buffer<offset_t, 1>> row_offsets;
  std::shared_ptr<sycl::buffer<index_t, 1>> column_indices;
  std::shared_ptr<sycl::buffer<value_t, 1>> nonzero_values;

  void build() {
    this->row_offsets = std::make_shared<sycl::buffer<offset_t, 1>>(sycl::range<1>(n_rows + 1));
    this->column_indices = std::make_shared<sycl::buffer<index_t, 1>>(sycl::range<1>(n_nonzeros));
    this->nonzero_values = std::make_shared<sycl::buffer<value_t, 1>>(sycl::range<1>(n_nonzeros));
  }
  
public:
  csr_t(index_t n_rows, index_t n_cols, offset_t n_nonzeros)
    : n_rows(n_rows), n_cols(n_cols), n_nonzeros(n_nonzeros) {
    build();   
  }
  
  csr_t(index_t n_rows, offset_t n_nonzeros)
    : csr_t(n_rows, 0, n_nonzeros) {}
  
  csr_t() : csr_t(0, 0, 0) {}

  csr_t(const csr_t& other)
    : n_rows(other.n_rows), n_cols(other.n_cols), n_nonzeros(other.n_nonzeros) {
    build();
  }

  csr_t(csr_t&& other) = default;

  ~csr_t() {};

  sycl::buffer<offset_t, 1>& get_row_offsets() {
    return *row_offsets;
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

  offset_t get_n_nonzeros() const {
    return n_nonzeros;
  }
};

} // namespace formats
} // namespace v0
} // namespace sygraph