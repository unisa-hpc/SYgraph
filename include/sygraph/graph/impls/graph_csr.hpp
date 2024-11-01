#pragma once

#include <sygraph/formats/csr.hpp>
#include <sygraph/utils/memory.hpp>

namespace sygraph {
namespace graph {
namespace detail {

template<typename IndexT, typename OffsetT, typename ValueT>
class GraphCSRDevice {
public:
  using vertex_t = IndexT; ///< The type used to represent vertices of the graph.
  using edge_t = OffsetT;  ///< The type used to represent edges of the graph.
  using weight_t = ValueT; ///< The type used to represent weights of the graph.
  struct NeighborIterator {
    NeighborIterator(IndexT* start_ptr, IndexT* ptr) : _start_ptr(start_ptr), _ptr(ptr) {}

    SYCL_EXTERNAL inline IndexT operator*() const { return *_ptr; }

    SYCL_EXTERNAL inline NeighborIterator& operator++() {
      ++_ptr;
      return *this;
    }

    SYCL_EXTERNAL inline NeighborIterator operator+(int n) const {
      NeighborIterator tmp = *this;
      tmp._ptr += n;
      return tmp;
    }

    SYCL_EXTERNAL inline bool operator==(const NeighborIterator& other) const { return _ptr == other._ptr; }

    SYCL_EXTERNAL inline bool operator!=(const NeighborIterator& other) const { return _ptr != other._ptr; }

    SYCL_EXTERNAL inline edge_t getIndex() const { return static_cast<edge_t>(_ptr - _start_ptr); }

    IndexT* _ptr;
    IndexT* _start_ptr;
  };

  /**
   * @brief Returns the number of vertices in the graph.
   * @return The number of vertices.
   */
  SYCL_EXTERNAL inline size_t getVertexCount() const { return _n_rows; }

  /**
   * @brief Returns the number of edges in the graph.
   * @return The number of edges.
   */
  SYCL_EXTERNAL inline size_t getEdgeCount() const { return _n_nonzeros; }

  /**
   * @brief Returns the number of neighbors of a vertex in the graph.
   * @param vertex The vertex.
   * @return The number of neighbors.
   */
  SYCL_EXTERNAL inline size_t getDegree(vertex_t vertex) const { return _row_offsets[vertex + 1] - _row_offsets[vertex]; }

  /**
   * @brief Returns the index of the first neighbor of a vertex in the graph.
   * @param vertex The vertex.
   * @return The index of the first neighbor.
   */
  SYCL_EXTERNAL inline vertex_t getFirstNeighbor(vertex_t vertex) const { return _row_offsets[vertex]; }

  // getters
  SYCL_EXTERNAL IndexT* getColumnIndices() const { return _column_indices; }

  SYCL_EXTERNAL OffsetT* getRowOffsets() const { return _row_offsets; }

  SYCL_EXTERNAL ValueT* getValues() const { return _nnz_values; }

  SYCL_EXTERNAL vertex_t getSourceVertex(edge_t edge) const {
    // binary search
    vertex_t low = 0;
    vertex_t high = _n_rows - 1;
    while (low <= high) {
      vertex_t mid = low + (high - low) / 2;
      if (_row_offsets[mid] <= edge && edge < _row_offsets[mid + 1]) {
        return mid;
      } else if (_row_offsets[mid] > edge) {
        high = mid - 1;
      } else {
        low = mid + 1;
      }
    }
    return _n_rows;
  }

  SYCL_EXTERNAL vertex_t getDestinationVertex(edge_t edge) const { return _column_indices[edge]; }

  SYCL_EXTERNAL weight_t getEdgeWeight(edge_t edge) const { return _nnz_values[edge]; }

  SYCL_EXTERNAL inline GraphCSRDevice::NeighborIterator begin(vertex_t vertex) const {
    return NeighborIterator(_column_indices, _column_indices + _row_offsets[vertex]);
  }

  SYCL_EXTERNAL inline GraphCSRDevice::NeighborIterator end(vertex_t vertex) const {
    return NeighborIterator(_column_indices, _column_indices + _row_offsets[vertex + 1]);
  }

  template<typename Func>
  SYCL_EXTERNAL inline size_t getIntersectionCount(const vertex_t& src, const vertex_t& dst, Func&& func) const {
    size_t count = 0;

    size_t degree_src = getDegree(src);
    size_t degree_dst = getDegree(dst);

    if (degree_src == 0 || degree_dst == 0) { return 0; }

    auto it_src = begin(src);
    auto it_dst = begin(dst);
    auto end_src = end(src);
    auto end_dst = end(dst);
    while (it_src != end_src && it_dst != end_dst) {
      if (*it_src == *it_dst) {
        func(*it_src);
        ++it_src;
        ++it_dst;
        ++count;
      } else if (*it_src < *it_dst) {
        ++it_src;
      } else {
        ++it_dst;
      }
    }
    return count;
  }

  IndexT _n_rows;      ///< The number of rows in the graph.
  OffsetT _n_nonzeros; ///< The number of non-zero values in the graph.

  IndexT* _column_indices; ///< Pointer to the column indices of the graph.
  OffsetT* _row_offsets;   ///< Pointer to the row offsets of the graph.
  ValueT* _nnz_values;     ///< Pointer to the non-zero values of the graph.
};

template<memory::space Space, typename IndexT, typename OffsetT, typename ValueT>
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
class GraphCSR : public Graph<IndexT, OffsetT, ValueT> {
public:
  using vertex_t = IndexT; ///< The type used to represent vertices of the graph.
  using edge_t = OffsetT;  ///< The type used to represent edges of the graph.
  using weight_t = ValueT; ///< The type used to represent weights of the graph.

  /**
   * @brief Constructs a graph_csr_t object.
   * @param q The SYCL queue to be used for memory operations.
   * @param csr The CSR format of the graph.
   * @param properties The properties of the graph.
   */
  GraphCSR(sycl::queue& q, formats::CSR<ValueT, IndexT, OffsetT>& csr, Properties properties)
      : Graph<IndexT, OffsetT, ValueT>(properties), _queue(q) {
    IndexT n_rows = csr.getRowOffsetsSize();
    OffsetT n_nonzeros = csr.getNumNonzeros();
    IndexT* row_offsets = memory::detail::memoryAlloc<IndexT, Space>(n_rows + 1, _queue);
    OffsetT* column_indices = memory::detail::memoryAlloc<OffsetT, Space>(n_nonzeros, _queue);
    ValueT* nnz_values = memory::detail::memoryAlloc<ValueT, Space>(n_nonzeros, _queue);

    auto e1 = _queue.copy(csr.getRowOffsets().data(), row_offsets, n_rows + 1);
    auto e2 = _queue.copy(csr.getColumnIndices().data(), column_indices, n_nonzeros);
    auto e3 = _queue.copy(csr.getValues().data(), nnz_values, n_nonzeros);
    e1.wait();
    e2.wait();
    e3.wait();

    this->_device_graph = {n_rows, n_nonzeros, column_indices, row_offsets, nnz_values};
  }

  /**
   * @brief Destroys the graph_csr_t object and frees the allocated memory.
   */
  ~GraphCSR() {
    sycl::free(_device_graph._row_offsets, _queue);
    sycl::free(_device_graph._column_indices, _queue);
    sycl::free(_device_graph._nnz_values, _queue);
  }

  /* Methods */

  auto& getDeviceGraph() { return _device_graph; }

  /* Override superclass methods */

  /**
   * @brief Returns the number of vertices in the graph.
   * @return The number of vertices.
   */
  size_t getVertexCount() const override { return _device_graph.getVertexCount(); }

  /**
   * @brief Returns the number of edges in the graph.
   * @return The number of edges.
   */
  size_t getEdgeCount() const override { return _device_graph.getEdgeCount(); }

  /**
   * @brief Returns the number of neighbors (out degree) of a vertex in the graph.
   * @param vertex The vertex.
   * @return The number of neighbors.
   */
  size_t getDegree(vertex_t vertex) const override { return _device_graph.getDegree(vertex); }

  /**
   * @brief Returns the index of the first neighbor of a vertex in the graph.
   * @param vertex The vertex.
   * @return The index of the first neighbor.
   */
  vertex_t getFirstNeighbor(vertex_t vertex) const override { return _device_graph.getFirstNeighbor(vertex); }

  vertex_t getSourceVertex(edge_t edge) const override { return _device_graph.getSourceVertex(edge); }

  vertex_t getDestinationVertex(edge_t edge) const override { return _device_graph.getDestinationVertex(edge); }

  weight_t getEdgeWeight(edge_t edge) const override { return _device_graph.getEdgeWeight(edge); }

  /* Getters and Setters for CSR Graph */

  /**
   * @brief Returns the number of rows in the graph.
   * @return The number of rows.
   */
  IndexT getOffsetsSize() const { return _device_graph.getVertexCount(); }

  /**
   * @brief Returns the number of non-zero values in the graph.
   * @return The number of non-zero values.
   */
  OffsetT getValuesSize() const { return _device_graph.getEdgeCount(); }

  /**
   * @brief Returns a pointer to the column indices of the graph.
   * @return A pointer to the column indices.
   */
  IndexT* getColumnIndices() { return _device_graph.getColumnIndices(); }

  /**
   * @brief Returns a constant pointer to the column indices of the graph.
   * @return A constant pointer to the column indices.
   */
  const IndexT* getColumnIndices() const { return _device_graph.getColumnIndices(); }

  /**
   * @brief Returns a pointer to the row offsets of the graph.
   * @return A pointer to the row offsets.
   */
  OffsetT* getRowOffsets() { return _device_graph.getRowOffsets(); }

  /**
   * @brief Returns a constant pointer to the row offsets of the graph.
   * @return A constant pointer to the row offsets.
   */
  const OffsetT* getRowOffsets() const { return _device_graph.getRowOffsets(); }

  /**
   * @brief Returns a pointer to the non-zero values of the graph.
   * @return A pointer to the non-zero values.
   */
  ValueT* getValues() { return _device_graph.getValues(); }

  /**
   * @brief Returns a constant pointer to the non-zero values of the graph.
   * @return A constant pointer to the non-zero values.
   */
  const ValueT* getValues() const { return _device_graph.getValues(); }


  /**
   * Returns the count of intersections between the source vertex and the destination vertex.
   *
   * @param src The source vertex.
   * @param dst The destination vertex.
   * @param func The function to be called for each intersection vertex.
   * @return The count of intersections.
   */
  const size_t getIntersectionCount(const vertex_t& src, const vertex_t& dst, std::function<void(vertex_t)> func) const {
    return _device_graph.getIntersectionCount(src, dst, func);
  }

  /**
   * @brief Returns the SYCL queue associated with the graph.
   * @return The SYCL queue.
   */
  sycl::queue& getQueue() const { return _queue; }

private:
  sycl::queue& _queue; ///< The SYCL queue associated with the graph.

  GraphCSRDevice<IndexT, OffsetT, ValueT> _device_graph;
};
} // namespace detail
} // namespace graph
} // namespace sygraph
