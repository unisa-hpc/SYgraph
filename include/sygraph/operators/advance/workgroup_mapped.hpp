/*
 * Copyright (c) 2025 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <sycl/sycl.hpp>

#include <sygraph/frontier/frontier.hpp>
#include <sygraph/graph/graph.hpp>
#include <sygraph/operators/config.hpp>
#include <sygraph/sycl/event.hpp>
#include <sygraph/utils/device.hpp>
#include <sygraph/utils/types.hpp>

namespace sygraph {
namespace operators {

namespace advance {

namespace detail {

template<sygraph::frontier::frontier_view IFW, sygraph::frontier::frontier_view OFW>
class workgroup_mapped_advance_kernel; // needed only for naming purposes

template<sygraph::frontier::frontier_view IFW, sygraph::frontier::frontier_view OFW, typename InFrontierDevT, typename OutFrontierDevT>
struct Context {
  size_t limit;
  InFrontierDevT in_dev_frontier;
  OutFrontierDevT out_dev_frontier;

  Context(size_t limit, InFrontierDevT in_dev_frontier, OutFrontierDevT out_dev_frontier)
      : limit(limit), in_dev_frontier(in_dev_frontier), out_dev_frontier(out_dev_frontier) {}

  SYCL_EXTERNAL inline size_t getAssignedElement(sycl::nd_item<1> item) const {
    if constexpr (IFW == sygraph::frontier::frontier_view::vertex) {
      const size_t bitmap_range = in_dev_frontier.getBitmapRange();
      const int* bitmap_offsets = in_dev_frontier.getOffsets();
      const size_t coarsening_factor = item.get_local_range(0) / bitmap_range;
      const size_t acutal_id_offset = (item.get_group_linear_id() * coarsening_factor) + (item.get_local_linear_id() / bitmap_range);
      const auto assigned_vertex = (bitmap_offsets[acutal_id_offset] * bitmap_range) + (item.get_local_linear_id() % bitmap_range);
      return assigned_vertex;
    } else if constexpr (IFW == sygraph::frontier::frontier_view::graph) {
      return item.get_global_linear_id();
    } else {
      return -1;
    }
  }

  SYCL_EXTERNAL inline bool check(sycl::nd_item<1> item, size_t vertex) const {
    if constexpr (IFW == sygraph::frontier::frontier_view::vertex) {
      return vertex < limit && in_dev_frontier.check(vertex);
    } else if constexpr (IFW == sygraph::frontier::frontier_view::graph) {
      return vertex < limit;
    } else {
      return -1;
    }
  }

  SYCL_EXTERNAL inline void insert(size_t vertex) const {
    if constexpr (OFW == sygraph::frontier::frontier_view::vertex) {
      out_dev_frontier.insert(vertex);
    } else if constexpr (OFW == sygraph::frontier::frontier_view::none) {
    }
  }
};

template<sygraph::frontier::frontier_view InFW,
         sygraph::frontier::frontier_view OutFW,
         typename T,
         typename ContextT,
         graph::detail::DeviceGraphConcept GraphDevT,
         typename LambdaT>
struct BitmapKernel {
  void operator()(sycl::nd_item<1> item) const {
    // 0. retrieve global and local ids
    const size_t gid = item.get_global_linear_id();
    const size_t lid = item.get_local_linear_id();
    const auto wgroup = item.get_group();
    const auto wgroup_id = item.get_group_linear_id();
    const size_t wgroup_size = wgroup.get_local_range(0);
    const auto sgroup = item.get_sub_group();
    const auto sgroup_id = sgroup.get_group_id();
    const size_t sgroup_size = sgroup.get_local_range()[0];
    const size_t llid = sgroup.get_local_linear_id();

    const auto assigned_vertex = context.getAssignedElement(item);

    // 1. load number of edges in local memory
    if (sgroup.leader()) { subgroup_reduce_tail[sgroup_id] = 0; }
    if (wgroup.leader()) { workgroup_reduce_tail[0] = 0; }

    sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::sub_group> sg_tail{subgroup_reduce_tail[sgroup_id]};
    sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group> wg_tail{workgroup_reduce_tail[0]};

    const uint32_t offset = sgroup_id * sgroup_size;
    if (context.check(item, assigned_vertex)) {
      uint32_t n_edges = graph_dev.getDegree(assigned_vertex);
      if (n_edges >= wgroup_size * wgroup_size) { // assign to the workgroup
        uint32_t loc = wg_tail.fetch_add(static_cast<uint32_t>(1));
        n_edges_wg[loc] = n_edges;
        workgroup_reduce[loc] = assigned_vertex;
        workgroup_ids[loc] = lid;
      } else if (n_edges >= sgroup_size) { // assign to the subgroup
        uint32_t loc = sg_tail.fetch_add(static_cast<uint32_t>(1));
        n_edges_sg[offset + loc] = n_edges;
        subgroup_reduce[offset + loc] = assigned_vertex;
        subgroup_ids[offset + loc] = lid;
      }
      visited[lid] = false;
    } else {
      visited[lid] = true;
    }

    sycl::group_barrier(wgroup);
    for (size_t i = 0; i < wg_tail.load(); i++) {
      auto vertex = workgroup_reduce[i];
      size_t n_edges = n_edges_wg[i];
      auto start = graph_dev.begin(vertex);

      for (auto j = lid; j < n_edges; j += wgroup_size) {
        auto n = start + j;
        auto edge = n.getIndex();
        auto weight = graph_dev.getEdgeWeight(edge);
        auto neighbor = *n;
        if (functor(vertex, neighbor, edge, weight)) { context.insert(neighbor); }
      }

      if (wgroup.leader()) { visited[workgroup_ids[i]] = true; }
    }

    sycl::group_barrier(sgroup);

    for (size_t i = 0; i < subgroup_reduce_tail[sgroup_id]; i++) { // active_elements_tail[subgroup_id] is always less or equal than subgroup_size
      size_t vertex_id = offset + i;
      auto vertex = subgroup_reduce[vertex_id];
      size_t n_edges = n_edges_sg[vertex_id];

      auto start = graph_dev.begin(vertex);

      for (auto j = llid; j < n_edges; j += sgroup_size) {
        auto n = start + j;
        auto edge = n.getIndex();
        auto weight = graph_dev.getEdgeWeight(edge);
        auto neighbor = *n;
        if (functor(vertex, neighbor, edge, weight)) { context.insert(neighbor); }
      }

      if (sgroup.leader()) { visited[subgroup_ids[vertex_id]] = true; }
    }
    sycl::group_barrier(sgroup);

    if (!visited[lid]) {
      auto vertex = assigned_vertex;
      auto start = graph_dev.begin(vertex);
      auto end = graph_dev.end(vertex);

      for (auto n = start; n != end; ++n) {
        auto edge = n.getIndex();
        auto weight = graph_dev.getEdgeWeight(edge);
        auto neighbor = *n;
        if (functor(vertex, neighbor, edge, weight)) { context.insert(neighbor); }
      }
    }
  }

  const ContextT context;
  const GraphDevT graph_dev;
  const sycl::local_accessor<uint32_t, 1> n_edges_wg;
  const sycl::local_accessor<uint32_t, 1> n_edges_sg;
  const sycl::local_accessor<bool, 1> visited;
  const sycl::local_accessor<T, 1> subgroup_reduce;
  const sycl::local_accessor<uint32_t, 1> subgroup_reduce_tail;
  const sycl::local_accessor<uint32_t, 1> subgroup_ids;
  const sycl::local_accessor<T, 1> workgroup_reduce;
  const sycl::local_accessor<uint32_t, 1> workgroup_reduce_tail;
  const sycl::local_accessor<uint32_t, 1> workgroup_ids;
  const LambdaT functor;
};


namespace workgroup_mapped {

template<sygraph::frontier::frontier_view InFW,
         sygraph::frontier::frontier_view OutFW,
         typename T,
         graph::detail::GraphConcept GraphT,
         typename InFrontierT,
         typename OutFrontierT,
         typename LambdaT>
sygraph::Event launchBitmapKernel(GraphT& graph, const InFrontierT& in, const OutFrontierT& out, LambdaT&& functor) {
  sycl::queue& q = graph.getQueue();

  size_t num_nodes = graph.getVertexCount();

  using element_t = std::conditional_t<InFW == sygraph::frontier::frontier_view::vertex, typename GraphT::vertex_t, typename GraphT::edge_t>;

  auto in_dev_frontier = in.getDeviceFrontier();
  auto out_dev_frontier = out.getDeviceFrontier();
  auto graph_dev = graph.getDeviceGraph();

  size_t coarsening_factor = types::detail::COMPUTE_UNIT_SIZE / sygraph::detail::device::getSubgroupSize(q);

  sycl::range<1> local_range;
  size_t global_size;
  if constexpr (InFW == sygraph::frontier::frontier_view::vertex) {
    size_t bitmap_range = in.getBitmapRange();
    size_t offsets_size = in.computeActiveFrontier();
    local_range = {bitmap_range * coarsening_factor};
    global_size = offsets_size * bitmap_range;
  } else if constexpr (InFW == sygraph::frontier::frontier_view::graph) {
    local_range = {types::detail::COMPUTE_UNIT_SIZE};
    global_size = num_nodes;
  } else {
    throw std::runtime_error("Invalid frontier view");
  }
  sycl::range<1> global_range{global_size > local_range[0] ? global_size + (local_range[0] - (global_size % local_range[0])) : local_range[0]};

  Context<InFW, OutFW, decltype(in_dev_frontier), decltype(out_dev_frontier)> context{num_nodes, in_dev_frontier, out_dev_frontier};
  using bitmap_kernel_t = BitmapKernel<InFW, OutFW, element_t, decltype(context), decltype(graph_dev), LambdaT>;

  const uint32_t max_num_subgroups = sygraph::detail::device::getMaxNumSubgroups(q);

  auto e = q.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<uint32_t, 1> n_edges_wg{local_range, cgh};
    sycl::local_accessor<uint32_t, 1> n_edges_sg{local_range, cgh};
    sycl::local_accessor<bool, 1> visited{local_range, cgh};
    sycl::local_accessor<element_t, 1> subgroup_reduce{local_range, cgh};
    sycl::local_accessor<uint32_t, 1> subgroup_reduce_tail{max_num_subgroups, cgh};
    sycl::local_accessor<uint32_t, 1> subgroup_ids{local_range, cgh};
    sycl::local_accessor<element_t, 1> workgroup_reduce{local_range, cgh};
    sycl::local_accessor<uint32_t, 1> workgroup_reduce_tail{1, cgh};
    sycl::local_accessor<uint32_t, 1> workgroup_ids{local_range, cgh};


    cgh.parallel_for<workgroup_mapped_advance_kernel<InFW, OutFW>>(sycl::nd_range<1>{global_range, local_range},
                                                                   bitmap_kernel_t{context,
                                                                                   graph_dev,
                                                                                   n_edges_wg,
                                                                                   n_edges_sg,
                                                                                   visited,
                                                                                   subgroup_reduce,
                                                                                   subgroup_reduce_tail,
                                                                                   subgroup_ids,
                                                                                   workgroup_reduce,
                                                                                   workgroup_reduce_tail,
                                                                                   workgroup_ids,
                                                                                   std::forward<LambdaT>(functor)});
  });
  return {e};
}

} // namespace workgroup_mapped
} // namespace detail
} // namespace advance
} // namespace operators
} // namespace sygraph