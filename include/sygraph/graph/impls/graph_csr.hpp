#pragma once

#include <sygraph/formats/csr.hpp>
#include <sygraph/utils/memory.hpp>

namespace sygraph {
inline namespace v0 {
namespace graph {
namespace detail {

template <typename index_t,
          typename offset_t,
          typename value_t>
class graph_csr_device_t {
  using vertex_t = index_t; ///< The type used to represent vertices of the graph.
  using edge_t = offset_t; ///< The type used to represent edges of the graph.
  using weight_t = value_t; ///< The type used to represent weights of the graph.
public:
  struct NeighbourIterator {
    NeighbourIterator(index_t* start_ptr, index_t* ptr) : start_ptr(start_ptr), ptr(ptr) {}

    SYCL_EXTERNAL inline index_t operator*() const {
      return *ptr;
    }

    SYCL_EXTERNAL inline NeighbourIterator& operator++() {
      ++ptr;
      return *this;
    }

    SYCL_EXTERNAL inline bool operator==(const NeighbourIterator& other) const {
      return ptr == other.ptr;
    }

    SYCL_EXTERNAL inline bool operator!=(const NeighbourIterator& other) const {
      return ptr != other.ptr;
    }

    SYCL_EXTERNAL inline edge_t get_index() const {
      return static_cast<edge_t>(ptr - start_ptr);
    }

    index_t* ptr;
    index_t* start_ptr;
  };

  /**
   * @brief Returns the number of vertices in the graph.
   * @return The number of vertices.
   */
  SYCL_EXTERNAL inline size_t get_vertex_count() const {
    return n_rows;
  }

  /**
   * @brief Returns the number of edges in the graph.
   * @return The number of edges.
   */
  SYCL_EXTERNAL inline size_t get_edge_count() const {
    return n_nonzeros;
  }

  /**
   * @brief Returns the number of neighbours of a vertex in the graph.
   * @param vertex The vertex.
   * @return The number of neighbours.
   */
  SYCL_EXTERNAL inline size_t get_neighbour_count(vertex_t vertex) const {
    return row_offsets[vertex + 1] - row_offsets[vertex];
  }

  /**
   * @brief Returns the index of the first neighbour of a vertex in the graph.
   * @param vertex The vertex.
   * @return The index of the first neighbour.
   */
  SYCL_EXTERNAL inline vertex_t get_first_neighbour_idx(vertex_t vertex) const {
    return row_offsets[vertex];
  }

  // getters
  SYCL_EXTERNAL index_t* get_column_indices() const {
    return column_indices;
  }

  SYCL_EXTERNAL offset_t* get_row_offsets() const {
    return row_offsets;
  }

  SYCL_EXTERNAL value_t* get_values() const {
    return nnz_values;
  }

  SYCL_EXTERNAL vertex_t get_source_vertex(edge_t edge) const {
    // binary search
    vertex_t low = 0;
    vertex_t high = n_rows - 1;
    while (low <= high) {
      vertex_t mid = low + (high - low) / 2;
      if (row_offsets[mid] <= edge && edge < row_offsets[mid + 1]) {
        return mid;
      } else if (row_offsets[mid] > edge) {
        high = mid - 1;
      } else {
        low = mid + 1;
      }
    }
    return n_rows;
  }

  SYCL_EXTERNAL vertex_t get_destination_vertex(edge_t edge) const {
    return column_indices[edge];
  }

  SYCL_EXTERNAL weight_t get_edge_weight(edge_t edge) const {
    return nnz_values[edge];
  }

  SYCL_EXTERNAL inline graph_csr_device_t::NeighbourIterator begin(vertex_t vertex) const {
    return NeighbourIterator(column_indices, column_indices + row_offsets[vertex]);
  }

  SYCL_EXTERNAL inline graph_csr_device_t::NeighbourIterator end(vertex_t vertex) const {
    return NeighbourIterator(column_indices, column_indices + row_offsets[vertex + 1]);
  }

  index_t n_rows; ///< The number of rows in the graph.
  offset_t n_nonzeros; ///< The number of non-zero values in the graph.

  index_t* column_indices; ///< Pointer to the column indices of the graph.
  offset_t* row_offsets; ///< Pointer to the row offsets of the graph.
  value_t* nnz_values; ///< Pointer to the non-zero values of the graph.
};

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

    index_t n_rows = csr.get_row_offsets_size();
    offset_t n_nonzeros = csr.get_num_nonzeros();
    index_t* row_offsets = memory::detail::memory_alloc<offset_t, space>(n_rows + 1, q);
    offset_t* column_indices = memory::detail::memory_alloc<index_t, space>(n_nonzeros, q);
    value_t* nnz_values = memory::detail::memory_alloc<value_t, space>(n_nonzeros, q);

    auto e1 = q.copy(csr.get_row_offsets().data(), row_offsets, n_rows + 1);
    auto e2 = q.copy(csr.get_column_indices().data(), column_indices, n_nonzeros);
    auto e3 = q.copy(csr.get_values().data(), nnz_values, n_nonzeros);
    e1.wait(); e2.wait(); e3.wait();

    this->device_graph = {
      n_rows, n_nonzeros, column_indices, row_offsets, nnz_values
    };
  }

  /**
   * @brief Destroys the graph_csr_t object and frees the allocated memory.
   */
  ~graph_csr_t() {
    sycl::free(device_graph.row_offsets, q);
    sycl::free(device_graph.column_indices, q);
    sycl::free(device_graph.nnz_values, q);
  }

  /* Methods */

  auto& get_device_graph() {
    return device_graph;
  }

  /* Override superclass methods */

  /**
   * @brief Returns the number of vertices in the graph.
   * @return The number of vertices.
   */
  inline size_t get_vertex_count() const override {
    return device_graph.get_vertex_count();
  }

  /**
   * @brief Returns the number of edges in the graph.
   * @return The number of edges.
   */
  inline size_t get_edge_count() const override {
    return device_graph.get_edge_count();
  }

  /**
   * @brief Returns the number of neighbours of a vertex in the graph.
   * @param vertex The vertex.
   * @return The number of neighbours.
   */
  inline size_t get_neighbour_count(vertex_t vertex) const override {
    return device_graph.get_neighbour_count(vertex);
  }

  /**
   * @brief Returns the index of the first neighbour of a vertex in the graph.
   * @param vertex The vertex.
   * @return The index of the first neighbour.
   */
  inline vertex_t get_first_neighbour_idx(vertex_t vertex) const override {
    return device_graph.get_first_neighbour_idx(vertex);
  }

  inline vertex_t get_source_vertex(edge_t edge) const override {
    return device_graph.get_source_vertex(edge);
  }

  inline vertex_t get_destination_vertex(edge_t edge) const override {
    return device_graph.get_destination_vertex(edge);
  }

  inline weight_t get_edge_weight(edge_t edge) const override {
    return device_graph.get_edge_weight(edge);
  }

  /* Getters and Setters for CSR Graph */

  /**
   * @brief Returns the number of rows in the graph.
   * @return The number of rows.
   */
  const index_t get_offsets_size() const  {
    return device_graph.n_rows;
  }

  /**
   * @brief Returns the number of non-zero values in the graph.
   * @return The number of non-zero values.
   */
  const offset_t get_values_size() const {
    return device_graph.n_nonzeros;
  }

  /**
   * @brief Returns a pointer to the column indices of the graph.
   * @return A pointer to the column indices.
   */
  index_t* get_column_indices() {
    return device_graph.column_indices;
  }

  /**
   * @brief Returns a constant pointer to the column indices of the graph.
   * @return A constant pointer to the column indices.
   */
  const index_t* get_column_indices() const {
    return device_graph.column_indices;
  }

  /**
   * @brief Returns a pointer to the row offsets of the graph.
   * @return A pointer to the row offsets.
   */
  offset_t* get_row_offsets() {
    return device_graph.row_offsets;
  }
  
  /**
   * @brief Returns a constant pointer to the row offsets of the graph.
   * @return A constant pointer to the row offsets.
   */
  const offset_t* get_row_offsets() const {
    return device_graph.row_offsets;
  }

  /**
   * @brief Returns a pointer to the non-zero values of the graph.
   * @return A pointer to the non-zero values.
   */
  value_t* get_values() {
    return device_graph.nnz_values;
  }

  /**
   * @brief Returns a constant pointer to the non-zero values of the graph.
   * @return A constant pointer to the non-zero values.
   */
  const value_t* get_values() const {
    return device_graph.nnz_values;
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

  graph_csr_device_t<index_t, offset_t, value_t> device_graph;
};
} // namespace detail
} // namespace graph
} // namespace v0
} // namespace sygraph
