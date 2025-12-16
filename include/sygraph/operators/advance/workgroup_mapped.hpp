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
#include <sygraph/utils/kernel.hpp>
#include <sygraph/utils/types.hpp>

namespace sygraph {
namespace operators {

namespace advance {

namespace detail {


template<typename...>
inline constexpr bool dependent_false_v = false;

template<sygraph::operators::direction Direction, sygraph::frontier::frontier_view IFW, sygraph::frontier::frontier_view OFW>
class workgroup_mapped_advance_kernel; // needed only for naming purposes

// Per-workgroup bookkeeping shared across helper methods.
struct ContextState {
  size_t group_offset;
  const uint16_t coarsening_factor;
  const uint32_t offsets_size;
  const sycl::nd_item<1> item;
};

template<sygraph::frontier::frontier_view IFW,
         sygraph::frontier::frontier_view OFW,
         sygraph::operators::direction Direction,
         typename InFrontierDevT,
         typename OutFrontierDevT>
// Wraps the device frontiers and exposes helpers used inside the advance kernel.
struct Context {
  // Available vertex/edge count and frontier views used by the kernel.
  size_t limit;
  InFrontierDevT in_dev_frontier;
  OutFrontierDevT out_dev_frontier;

  // Build the execution context once per kernel launch.
  Context(size_t limit, InFrontierDevT in_dev_frontier, OutFrontierDevT out_dev_frontier)
      : limit(limit), in_dev_frontier(in_dev_frontier), out_dev_frontier(out_dev_frontier) {}

  // Initialize a ContextState for the calling work-group.
  SYCL_EXTERNAL inline ContextState init(sycl::nd_item<1>& item) const {
    return {
        item.get_group_linear_id(),
        static_cast<uint16_t>(item.get_local_range(0) / in_dev_frontier.getBitmapRange()),
        in_dev_frontier.getOffsetsSize()[0],
        item,
    };
  }

  // Determine whether the current work-group still owns unprocessed segments.
  SYCL_EXTERNAL inline bool needToProcess(ContextState& state) const { return (state.group_offset * state.coarsening_factor < state.offsets_size); }

  // Advance the state to the next chunk of bitmap offsets.
  SYCL_EXTERNAL inline void completeIteration(ContextState& state) const { state.group_offset += state.item.get_group_range(0); }

  // Compute which vertex/edge matches the current lane.
  SYCL_EXTERNAL inline size_t getAssignedElement(const ContextState& state) const {
    if constexpr (IFW == sygraph::frontier::frontier_view::vertex) {
      const uint16_t bitmap_range = in_dev_frontier.getBitmapRange();
      const uint32_t actual_id_offset = (state.group_offset * state.coarsening_factor) + (state.item.get_local_linear_id() / bitmap_range);
      const int* bitmap_offsets = in_dev_frontier.getOffsets();
      const auto assigned_vertex = (bitmap_offsets[actual_id_offset] * bitmap_range) + (state.item.get_local_linear_id() % bitmap_range);
      return assigned_vertex;
    } else if constexpr (IFW == sygraph::frontier::frontier_view::graph) {
      return state.item.get_global_linear_id();
    } else {
      return -1;
    }
  }

  // Check whether the vertex is inside the active frontier / graph.
  SYCL_EXTERNAL inline bool check(const ContextState& state, const uint32_t& vertex) const {
    if constexpr (IFW == sygraph::frontier::frontier_view::vertex) {
      return vertex < limit && ((Direction == sygraph::operators::direction::push) == in_dev_frontier.check(vertex));
    } else if constexpr (IFW == sygraph::frontier::frontier_view::graph) {
      return vertex < limit;
    } else {
      return -1;
    }
  }

  // Predicate used to skip invalid neighbors (only meaningful in pull mode).
  SYCL_EXTERNAL inline bool isValidNeighbor(const ContextState& state, const uint32_t& neighbor) const {
    if constexpr (Direction == sygraph::operators::direction::pull) {
      return in_dev_frontier.check(neighbor);
    } else {
      return true;
    }
  }

  // Add the produced vertex/neighbor to the destination frontier depending on direction.
  SYCL_EXTERNAL inline void insert(const ContextState& state, const uint32_t& vertex, const uint32_t& neighbor) const {
    if constexpr (OFW == sygraph::frontier::frontier_view::vertex) {
      if constexpr (Direction == sygraph::operators::direction::push)
        out_dev_frontier.insert(neighbor);
      else
        out_dev_frontier.insert(vertex);
    } else if constexpr (OFW == sygraph::frontier::frontier_view::none) {
    }
  }

  template<typename G>
  SYCL_EXTERNAL inline void accumulate_edges(sycl::nd_item<1> item, const G& graph_dev, const uint32_t& vertex) const {
    auto wgid = item.get_group_linear_id();
    sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device> ref{in_dev_frontier._edges_processed[wgid]};
    if (in_dev_frontier.check(vertex)) { ref += graph_dev.getDegree(vertex); }
  }
};

template<sygraph::frontier::frontier_view InFW,
         sygraph::frontier::frontier_view OutFW,
         typename T,
         typename ContextT,
         graph::detail::DeviceGraphConcept GraphDevT,
         typename LambdaT>
// Work-distribution kernel that maps bitmap entries to the most suitable execution granularity.
struct BitmapKernel {
  // Entry point invoked by the SYCL runtime for each work-item.
  void operator()(sycl::nd_item<1> item) const {
    const size_t lid = item.get_local_linear_id();
    const auto wgroup = item.get_group();
    const size_t wgroup_size = wgroup.get_local_range(0);
    const auto sgroup = item.get_sub_group();
    const auto sgroup_id = sgroup.get_group_id();
    const size_t sgroup_size = sgroup.get_local_range()[0];
    const size_t llid = sgroup.get_local_linear_id();

    auto state = context.init(item);

    sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::sub_group> sg_tail{subgroup_reduce_tail[sgroup_id]};
    sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group> wg_tail{workgroup_reduce_tail[0]};

    while (context.needToProcess(state)) {
      const auto assigned_vertex = context.getAssignedElement(state);

      context.accumulate_edges(item, graph_dev, assigned_vertex);

      // Reset local reductions so the group can classify work for this iteration.
      if (sgroup.leader()) { subgroup_reduce_tail[sgroup_id] = 0; }
      if (wgroup.leader()) { workgroup_reduce_tail[0] = 0; }


      const uint32_t offset = sgroup_id * sgroup_size;
      if (context.check(state, assigned_vertex)) {
        const uint32_t n_edges = graph_dev.getDegree(assigned_vertex);
        if (n_edges >= wgroup_size * wgroup_size) { // dispatch to the whole workgroup
          const uint32_t loc = wg_tail.fetch_add(1U);
          n_edges_wg[loc] = n_edges;
          workgroup_reduce[loc] = assigned_vertex;
          workgroup_ids[loc] = lid;
        } else if (n_edges >= sgroup_size) { // dispatch to a subgroup
          const uint32_t loc = sg_tail.fetch_add(1U);
          n_edges_sg[offset + loc] = n_edges;
          subgroup_reduce[offset + loc] = assigned_vertex;
          subgroup_ids[offset + loc] = lid;
        }
        visited[lid] = false;
      } else {
        visited[lid] = true;
      }

      // Process vertices assigned to the workgroup.
      sycl::group_barrier(wgroup);
      const auto wg_assigned = wg_tail.load();
      for (size_t i = 0; i < wg_assigned; i++) {
        const auto vertex = workgroup_reduce[i];
        const size_t n_edges = n_edges_wg[i];
        auto start = graph_dev.begin(vertex);

        for (auto j = lid; j < n_edges; j += wgroup_size) {
          auto n = start + j;
          const auto edge = n.getIndex();
          const auto weight = graph_dev.getEdgeWeight(edge);
          const auto neighbor = *n;
          if (context.isValidNeighbor(state, neighbor) && functor(vertex, neighbor, edge, weight)) { context.insert(state, vertex, neighbor); }
        }

        if (wgroup.leader()) { visited[workgroup_ids[i]] = true; }
      }

      // Process vertices assigned to subgroups.
      sycl::group_barrier(sgroup);
      const auto sg_assigned = sg_tail.load();
      for (size_t i = 0; i < sg_assigned; i++) {
        const size_t vertex_id = offset + i;
        const auto vertex = subgroup_reduce[vertex_id];
        const size_t n_edges = n_edges_sg[vertex_id];
        auto start = graph_dev.begin(vertex);

        for (auto j = llid; j < n_edges; j += sgroup_size) {
          auto n = start + j;
          const auto edge = n.getIndex();
          const auto weight = graph_dev.getEdgeWeight(edge);
          const auto neighbor = *n;
          if (context.isValidNeighbor(state, neighbor) && functor(vertex, neighbor, edge, weight)) { context.insert(state, vertex, neighbor); }
        }

        if (sgroup.leader()) { visited[subgroup_ids[vertex_id]] = true; }
      }
      sycl::group_barrier(sgroup);

      // Fallback: each lane processes its own very small vertex.
      if (!visited[lid]) {
        const auto vertex = assigned_vertex;
        auto start = graph_dev.begin(vertex);
        const auto end = graph_dev.end(vertex);

        for (auto n = start; n != end; ++n) {
          const auto edge = n.getIndex();
          const auto weight = graph_dev.getEdgeWeight(edge);
          const auto neighbor = *n;
          if (context.isValidNeighbor(state, neighbor) && functor(vertex, neighbor, edge, weight)) { context.insert(state, vertex, neighbor); }
        }
      }
      context.completeIteration(state);
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

// Determine the execution configuration for vertex/graph frontiers while keeping
// the heuristics that balance occupancy and available compute.
template<sygraph::frontier::frontier_view InFW, typename GraphT, typename InFrontierT>
inline sygraph::detail::kernel::LaunchConfig
buildLaunchConfig(GraphT& graph, const InFrontierT& in, bool pull_advance, int expected_size, size_t coarsening_factor, sycl::queue& q) {
  sygraph::detail::kernel::LaunchConfig config{};
  auto in_dev_frontier = in.getDeviceFrontier();
  if constexpr (InFW == sygraph::frontier::frontier_view::vertex) {
    const size_t bitmap_range = in.getBitmapRange();
    config.local = {bitmap_range * coarsening_factor};
    config.dependency = in.computeActiveFrontier(pull_advance);

    size_t requested_global = 0;
    if (expected_size > 0) {
      requested_global = static_cast<size_t>(expected_size);
    } else if (expected_size == frontier::size::infer_from_device) {
      requested_global = config.local[0] * (sygraph::detail::device::getNumComputeUnits(q) * coarsening_factor);
    } else if (expected_size == frontier::size::fetch_from_memory) {
      config.dependency.wait_and_throw();
      uint32_t active_size = 0;
      auto copy_e = q.copy(in_dev_frontier.getOffsetsSize(), &active_size, 1);
      copy_e.wait();
#ifdef ENABLE_PROFILING
      sygraph::Profiler::addEvent(copy_e, "frontier_size_fetch");
#endif
      requested_global = static_cast<size_t>(active_size) * bitmap_range;
    } else {
      throw std::runtime_error("Invalid expected_size value");
    }

    config.global = {sygraph::detail::kernel::ensureLocalMultiple(requested_global, config.local[0])};
  } else if constexpr (InFW == sygraph::frontier::frontier_view::graph) {
    config.local = {types::detail::COMPUTE_UNIT_SIZE};
    const size_t requested_global = graph.getVertexCount();
    config.global = {sygraph::detail::kernel::ensureLocalMultiple(requested_global, config.local[0])};
  } else {
    throw std::runtime_error("Invalid frontier view for compute operation.");
  }

  return config;
}

namespace workgroup_mapped {

template<sygraph::frontier::frontier_view InFW,
         sygraph::frontier::frontier_view OutFW,
         sygraph::operators::direction Direction,
         typename T,
         graph::detail::GraphConcept GraphT,
         typename InFrontierT,
         typename OutFrontierT,
         typename LambdaT>
// Launch the mapped advance kernel for the requested frontier/configuration.
sygraph::Event launchBitmapKernel(GraphT& graph, const InFrontierT& in, const OutFrontierT& out, LambdaT&& functor, int expected_size) {
  sycl::queue& q = graph.getQueue();

  size_t num_nodes = graph.getVertexCount();

  using element_t = std::conditional_t<InFW == sygraph::frontier::frontier_view::vertex, typename GraphT::vertex_t, typename GraphT::edge_t>;

  auto in_dev_frontier = in.getDeviceFrontier();
  auto out_dev_frontier = out.getDeviceFrontier();
  auto graph_dev = graph.getDeviceGraph();
  if constexpr (Direction == sygraph::operators::direction::pull) { graph_dev = graph.getInverseDeviceGraph(); }

  // Keep each workgroup busy by mapping multiple bitmap entries to it.
  const size_t coarsening_factor = types::detail::COMPUTE_UNIT_SIZE / sygraph::detail::device::getSubgroupSize(q);
  const bool pull_advance = (Direction == sygraph::operators::direction::pull);

  const auto launch_cfg = buildLaunchConfig<InFW>(graph, in, pull_advance, expected_size, coarsening_factor, q);
  const sycl::range<1>& local_range = launch_cfg.local;
  const sycl::range<1>& global_range = launch_cfg.global;
  const sycl::event& dependency = launch_cfg.dependency;

  Context<InFW, OutFW, Direction, decltype(in_dev_frontier), decltype(out_dev_frontier)> context{num_nodes, in_dev_frontier, out_dev_frontier};
  using bitmap_kernel_t = BitmapKernel<InFW, OutFW, element_t, decltype(context), decltype(graph_dev), LambdaT>;

  const uint32_t max_num_subgroups = sygraph::detail::device::getMaxNumSubgroups(q);

  auto e = q.submit([&](sycl::handler& cgh) {
    cgh.depends_on(dependency);
    // Local storage used to distribute vertices across workgroups/subgroups.
    sycl::local_accessor<uint32_t, 1> n_edges_wg{local_range, cgh};
    sycl::local_accessor<uint32_t, 1> n_edges_sg{local_range, cgh};
    sycl::local_accessor<bool, 1> visited{local_range, cgh};
    sycl::local_accessor<element_t, 1> subgroup_reduce{local_range, cgh};
    sycl::local_accessor<uint32_t, 1> subgroup_reduce_tail{max_num_subgroups, cgh};
    sycl::local_accessor<uint32_t, 1> subgroup_ids{local_range, cgh};
    sycl::local_accessor<element_t, 1> workgroup_reduce{local_range, cgh};
    sycl::local_accessor<uint32_t, 1> workgroup_reduce_tail{1, cgh};
    sycl::local_accessor<uint32_t, 1> workgroup_ids{local_range, cgh};


    cgh.parallel_for<workgroup_mapped_advance_kernel<Direction, InFW, OutFW>>(sycl::nd_range<1>{global_range, local_range},
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
