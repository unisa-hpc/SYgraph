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
class graph_impl_csr_t : Graph<index_t, offset_t, value_t> {
public:
/* Constructors */
  graph_impl_csr_t(sycl::queue& q, formats::CSR<value_t, index_t, offset_t>& csr, Properties properties)
    : Graph<index_t, offset_t, value_t>(properties), q(q)
  {
    this->n_rows = csr.getNumRows();
    this->n_nonzeros = csr.getNumNonzeros();
    this->row_offsets = memory::detail::memory_alloc<offset_t, space>(n_rows + 1, q);
    this->column_indices = memory::detail::memory_alloc<index_t, space>(n_nonzeros, q);
    this->nnz_values = memory::detail::memory_alloc<value_t, space>(n_nonzeros, q);

    auto e1 = q.copy(csr.getRowOffsets().data(), this->row_offsets, n_rows + 1);
    auto e2 = q.copy(csr.getColumnIndices().data(), this->column_indices, n_nonzeros);
    auto e3 = q.copy(csr.getNnzValues().data(), this->nnz_values, n_nonzeros);
    e1.wait(); e2.wait(); e3.wait();
  }

  ~graph_impl_csr_t() {
    sycl::free(row_offsets, q);
    sycl::free(column_indices, q);
    sycl::free(nnz_values, q);
  }

/* Methods */

  index_t getNumRows() const  {
    return n_rows;
  }

  offset_t getNumVals() const {
    return n_nonzeros;
  }

  index_t* getColumnIndices() {
    return column_indices;
  }

  offset_t* getRowOffsets() {
    return row_offsets;
  }

  value_t* getNnzValues() {
    return nnz_values;
  }

  sycl::queue& getQueue() const {
    return q;
  }

private:
  sycl::queue& q;

  index_t n_rows;
  offset_t n_nonzeros;

  index_t* column_indices;
  offset_t* row_offsets;
  value_t* nnz_values;
};;
} // namespace detail
} // namespace graph
} // namespace v0
} // namespace sygraph
