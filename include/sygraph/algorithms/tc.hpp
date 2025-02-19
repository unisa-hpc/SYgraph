/*
 * Copyright (c) 2025 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <sycl/sycl.hpp>

#include <memory>

#include <sygraph/graph/graph.hpp>
#ifdef ENABLE_PROFILING
#include <sygraph/utils/profiler.hpp>
#endif
#include <sygraph/sync/atomics.hpp>


namespace sygraph {
namespace algorithms {
namespace detail {

template<typename GraphType>
struct TCInstance {
  using vertex_t = typename GraphType::vertex_t;
  using edge_t = typename GraphType::edge_t;
  using weight_t = typename GraphType::weight_t;

  GraphType& G;

  TCInstance(GraphType& G) : G(G) {
    sycl::queue& queue = G.getQueue();
    size_t num_nodes = G.getVertexCount();
    triangles = memory::detail::memoryAlloc<uint32_t, memory::space::device>(num_nodes, queue);
    queue.fill(triangles, static_cast<uint32_t>(0), num_nodes).wait();
  }

  ~TCInstance() {
    sycl::queue& queue = G.getQueue();
    sycl::free(triangles, queue);
  }


  uint32_t* triangles;
};
} // namespace detail


template<typename GraphType>
class TC {
  using vertex_t = typename GraphType::vertex_t;
  using edge_t = typename GraphType::edge_t;
  using weight_t = typename GraphType::weight_t;

public:
  TC(GraphType& g) : _g(g) {};


  void init() { _instance = std::make_unique<detail::TCInstance<GraphType>>(_g); }


  void reset() { _instance.reset(); }


  template<bool EnableProfiling = false>
  void run() {
    if (!_instance) { throw std::runtime_error("TC instance not initialized"); }

    auto& G = _instance->G;
    auto& triangles = _instance->triangles;

    sycl::queue& queue = G.getQueue();

    size_t num_nodes = G.getVertexCount();

    auto e = queue.submit([&](sycl::handler& cgh) {
      auto g_dev = G.getDeviceGraph();
      size_t edges = G.getEdgeCount();

      cgh.parallel_for({edges}, [=](sycl::id<1> idx) {
        edge_t edge = idx[0];

        vertex_t src = g_dev.getSourceVertex(edge);
        vertex_t dst = g_dev.getDestinationVertex(edge);

        auto src_it = g_dev.begin(src);
        auto src_end = g_dev.end(src);
        auto dst_it = g_dev.begin(dst);
        auto dst_end = g_dev.end(dst);
        uint32_t src_degree = g_dev.getDegree(src);
        uint32_t dst_degree = g_dev.getDegree(dst);

        while (src_it != src_end && dst_it != dst_end) {
          if (*src_it < *dst_it) {
            ++src_it;
          } else if (*src_it > *dst_it) {
            ++dst_it;
          } else { // triangle found
            sygraph::sync::atomicFetchAdd<uint32_t>(triangles + src, 1);
            ++src_it;
            ++dst_it;
          }
        }
      });
    });
    e.wait();
#ifdef ENABLE_PROFILING
    sygraph::Profiler::addEvent(e, "TC");
#endif
  }

  size_t getNumTriangles(vertex_t v) const {
    if (!_instance) { throw std::runtime_error("TC instance not initialized"); }
    return _instance->triangles[v];
  }

  size_t getNumTriangles() const {
    if (!_instance) { throw std::runtime_error("TC instance not initialized"); }

    sycl::queue& queue = _g.getQueue();
    size_t num_nodes = _g.getVertexCount();
    auto& triangles = _instance->triangles;

    sycl::buffer<uint32_t, 1> sum_buff(sycl::range<1>(1));

    queue
        .submit([&](sycl::handler& cgh) {
          auto red = sycl::reduction(sum_buff, cgh, sycl::plus<uint32_t>());
          cgh.parallel_for(sycl::range{num_nodes}, red, [=](sycl::id<1> idx, auto& sum) { sum += triangles[idx]; });
        })
        .wait();

    sycl::host_accessor sum_acc(sum_buff, sycl::read_only);
    return sum_acc[0] / 3;
  }

private:
  GraphType& _g;
  std::unique_ptr<detail::TCInstance<GraphType>> _instance;
};

} // namespace algorithms
} // namespace sygraph