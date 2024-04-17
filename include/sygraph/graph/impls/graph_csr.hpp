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
/**
 * @file graph_csr.hpp
 * @brief Contains the definition of the graph_csr_t class.
 */

/**
 * @class graph_csr_t
 * @brief Represents a graph in Compressed Sparse Row (CSR) format.
 * @tparam index_t The type used to represent indices of the graph.
 * @tparam offset_t The type used to represent offsets of the graph.
 * @tparam value_t The type used to represent values of the graph.
 */
class graph_csr_t : public Graph<index_t, offset_t, value_t> {
public:
  using vertex_t = index_t; ///< The type used to represent vertices of the graph.
  using edge_t = offset_t; ///< The type used to represent edges of the graph.
  using weight_t = offset_t; ///< The type used to represent weights of the graph.

  /**
   * @brief Constructs a graph_csr_t object.
   * @param q The SYCL queue to be used for memory operations.
   * @param csr The CSR format of the graph.
   * @param properties The properties of the graph.
   */
  graph_csr_t(sycl::queue& q, formats::CSR<value_t, index_t, offset_t>& csr, Properties properties)
    : Graph<index_t, offset_t, value_t>(properties), q(q)
  {
    this->n_rows = csr.get_row_offsets_size() - 1;
    this->n_nonzeros = csr.get_num_nonzeros();
    this->row_offsets = memory::detail::memory_alloc<offset_t, space>(n_rows + 1, q);
    this->column_indices = memory::detail::memory_alloc<index_t, space>(n_nonzeros, q);
    this->nnz_values = memory::detail::memory_alloc<value_t, space>(n_nonzeros, q);

    auto e1 = q.copy(csr.get_row_offsets().data(), this->row_offsets, n_rows + 1);
    auto e2 = q.copy(csr.get_column_indices().data(), this->column_indices, n_nonzeros);
    auto e3 = q.copy(csr.get_values().data(), this->nnz_values, n_nonzeros);
    e1.wait(); e2.wait(); e3.wait();
  }

  /**
   * @brief Destroys the graph_csr_t object and frees the allocated memory.
   */
  ~graph_csr_t() {
    sycl::free(row_offsets, q);
    sycl::free(column_indices, q);
    sycl::free(nnz_values, q);
  }

  /* Methods */

  /* Override superclass methods */

  /**
   * @brief Returns the number of vertices in the graph.
   * @return The number of vertices.
   */
  inline size_t get_vertex_count() const override {
    return n_rows;
  }

  /**
   * @brief Returns the number of edges in the graph.
   * @return The number of edges.
   */
  inline size_t get_edge_count() const override {
    return n_nonzeros;
  }

  /* Getters and Setters for CSR Graph */

  /**
   * @brief Returns the number of rows in the graph.
   * @return The number of rows.
   */
  const index_t get_offsets_size() const  {
    return n_rows;
  }

  /**
   * @brief Returns the number of non-zero values in the graph.
   * @return The number of non-zero values.
   */
  const offset_t get_values_size() const {
    return n_nonzeros;
  }

  /**
   * @brief Returns a pointer to the column indices of the graph.
   * @return A pointer to the column indices.
   */
  index_t* get_column_indices() {
    return column_indices;
  }

  /**
   * @brief Returns a constant pointer to the column indices of the graph.
   * @return A constant pointer to the column indices.
   */
  const index_t* get_column_indices() const {
    return column_indices;
  }

  /**
   * @brief Returns a pointer to the row offsets of the graph.
   * @return A pointer to the row offsets.
   */
  offset_t* get_row_offsets() {
    return row_offsets;
  }
  
  /**
   * @brief Returns a constant pointer to the row offsets of the graph.
   * @return A constant pointer to the row offsets.
   */
  const offset_t* get_row_offsets() const {
    return row_offsets;
  }

  /**
   * @brief Returns a pointer to the non-zero values of the graph.
   * @return A pointer to the non-zero values.
   */
  value_t* get_values() {
    return nnz_values;
  }

  /**
   * @brief Returns a constant pointer to the non-zero values of the graph.
   * @return A constant pointer to the non-zero values.
   */
  const value_t* get_values() const {
    return nnz_values;
  }

  /**
   * @brief Returns the SYCL queue associated with the graph.
   * @return The SYCL queue.
   */
  sycl::queue& get_queue() const {
    return q;
  }

private:
  sycl::queue& q; ///< The SYCL queue associated with the graph.

  index_t n_rows; ///< The number of rows in the graph.
  offset_t n_nonzeros; ///< The number of non-zero values in the graph.

  index_t* column_indices; ///< Pointer to the column indices of the graph.
  offset_t* row_offsets; ///< Pointer to the row offsets of the graph.
  value_t* nnz_values; ///< Pointer to the non-zero values of the graph.
};
} // namespace detail
} // namespace graph
} // namespace v0
} // namespace sygraph
