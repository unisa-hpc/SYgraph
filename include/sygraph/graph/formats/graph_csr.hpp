#pragma once

#include <sygraph/formats/csr.hpp>
#include <sygraph/utils/memory.hpp>

namespace sygraph {
inline namespace v0 {
namespace graph {
namespace detail {

template <memory::space space,
          typename index_t,
          typename offset_t,
          typename value_t>
class graph_impl_csr_t {

public:

  graph_impl_csr_t(sycl::queue& q, index_t n_rows, offset_t n_nonzeros)
    : n_rows(n_rows), n_nonzeros(n_nonzeros) {
    this->row_offsets = memory::detail::memory_alloc<offset_t, space>(n_rows + 1, q);
    this->column_indices = memory::detail::memory_alloc<index_t, space>(n_nonzeros, q);
    this->nnz_values = memory::detail::memory_alloc<value_t, space>(n_nonzeros, q);
  }

  ~graph_impl_csr_t() {
    sycl::free(row_offsets, q);
    sycl::free(column_indices, q);
    sycl::free(nnz_values, q);
  }

protected:
  template<typename CSR_T>
  void set(CSR_T& csr) {
    auto e1 = q.copy(csr.template row_offsets.data(), this->row_offsets, n_rows + 1);
    auto e2 = q.copy(csr.template column_indices.data(), this->column_indices, n_nonzeros);
    auto e3 = q.copy(csr.template nnz_values.data(), this->nnz_values, n_nonzeros);
    e1.wait();
    e2.wait();
    e3.wait();
  }

private:
  sycl::queue& q;

  index_t n_rows;
  offset_t n_nonzeros;

  index_t* column_indices;
  offset_t* row_offsets;
  value_t* nnz_values;
};

} // namespace detail
} // namespace graph
} // namespace v0
} // namespace sygraph
