#pragma once

#include <sycl/sycl.hpp>

#include <sygraph/frontier/frontier.hpp>
#include <sygraph/graph/graph.hpp>
#include <sygraph/operators/config.hpp>
#include <sygraph/sycl/event.hpp>
#include <sygraph/utils/types.hpp>

namespace sygraph {
inline namespace v0 {
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


template<typename T, typename FrontierDevT, graph::detail::DeviceGraphConcept GraphDevT, typename LambdaT>
struct VectorKernel {
  void operator()(sycl::nd_item<1> item) const {
    // 0. retrieve global and local ids
    const size_t gid = item.get_global_linear_id();
    const size_t lid = item.get_local_linear_id();
    const size_t local_range = item.get_local_range(0);
    const auto group = item.get_group();
    const auto group_id = item.get_group_linear_id();
    const auto subgroup = item.get_sub_group();
    const auto subgroup_id = subgroup.get_group_id();
    const size_t subgroup_size = subgroup.get_local_range()[0];
    const size_t llid = subgroup.get_local_linear_id();

    uint32_t* global_tail = out_dev_frontier.getVectorTail();

    // 1. load number of edges in local memory
    sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group> pad_tail_ref{pad_tail[0]};
    sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::sub_group> active_elems_tail_ref{
        active_elements_local_tail[subgroup_id]};

    if (gid == 0) { global_tail[0] = 0; }
    if (group.leader()) { pad_tail_ref.store(0); }
    if (subgroup.leader()) { active_elems_tail_ref.store(0); }

    const uint32_t offset = subgroup_id * subgroup_size;
    if (gid < active_elements_size) {
      T element = active_elements[gid];
      uint32_t n_edges = graph_dev.getDegree(element);
      if (n_edges >= subgroup_size) {
        uint32_t loc = active_elems_tail_ref.fetch_add(1);
        active_elements_local[offset + loc] = element;
        n_edges_local[offset + loc] = n_edges;
        ids[offset + loc] = lid;
      }
      // active_elements_local[lid] = element;
      visited[lid] = false;
    } else {
      n_edges_local[lid] = 0;
      visited[lid] = true;
    }
    sycl::group_barrier(subgroup);

    // 2. process elements with less than local_range edges but more than one subgroup size edges
    for (uint32_t i = 0; i < active_elems_tail_ref.load(); i++) {
      size_t vertex_id = offset + i;
      auto vertex = active_elements_local[vertex_id];
      size_t n_edges = n_edges_local[vertex_id];
      size_t private_slice = n_edges / subgroup_size;
      auto start = graph_dev.begin(vertex) + (private_slice * llid);
      auto end = llid == subgroup_size - 1 ? graph_dev.end(vertex) : start + private_slice;

      for (auto n = start; n != end; ++n) {
        auto edge = n.getIndex();
        auto weight = graph_dev.getEdgeWeight(edge);
        auto neighbor = *n;
        if (functor(vertex, neighbor, edge, weight)) { out_dev_frontier.insert(neighbor, pad, pad_tail_ref); }
      }
      if (subgroup.leader()) { visited[ids[vertex_id]] = true; }
      // if (n_edges_local[vertex_id] >= subgroup_size * subgroup_size) {
      // }
    }
    sycl::group_barrier(group);

    // 3. process the rest
    if (!visited[lid]) {
      auto vertex = active_elements[gid];
      auto start = graph_dev.begin(vertex);
      auto end = graph_dev.end(vertex);

      for (auto n = start; n != end; ++n) {
        auto edge = n.getIndex();
        auto weight = graph_dev.getEdgeWeight(edge);
        auto neighbor = *n;
        if (functor(vertex, neighbor, edge, weight)) { out_dev_frontier.insert(neighbor, pad, pad_tail_ref); }
      }
    }

    sycl::group_barrier(group);
    out_dev_frontier.finalize(item, pad, pad_tail_ref);
  }

  const T* active_elements;
  const size_t active_elements_size;
  const FrontierDevT in_dev_frontier;
  const FrontierDevT out_dev_frontier;
  const GraphDevT graph_dev;
  const sycl::local_accessor<uint32_t, 1> n_edges_local;
  const sycl::local_accessor<bool, 1> visited;
  const sycl::local_accessor<T, 1> active_elements_local;
  const sycl::local_accessor<uint32_t, 1> active_elements_local_tail;
  const sycl::local_accessor<uint32_t, 1> ids;
  const sycl::local_accessor<T, 1> pad;
  const sycl::local_accessor<uint32_t, 1> pad_tail;
  const LambdaT functor;
};

// TODO: add the context for passing arguments
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

    // TODO: maybe remove n_edges_wg and n_edges_sg

    const size_t offset = sgroup_id * sgroup_size;
    if (context.check(item, assigned_vertex)) {
      size_t n_edges = graph_dev.getDegree(assigned_vertex);
      if (n_edges >= wgroup_size * wgroup_size) { // assign to the workgroup
        size_t loc = wg_tail.fetch_add(1);
        n_edges_wg[loc] = n_edges;
        workgroup_reduce[loc] = assigned_vertex;
        workgroup_ids[loc] = lid;
      } else if (n_edges >= sgroup_size) { // assign to the subgroup
        size_t loc = sg_tail.fetch_add(1);
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

  auto in_dev_frontier = in.getDeviceFrontier();
  auto out_dev_frontier = out.getDeviceFrontier();
  auto graph_dev = graph.getDeviceGraph();

  size_t coarsening_factor = 16; // TODO: compute this value automatically in order to fully utilize the workgroup.
  sycl::range<1> local_range;
  size_t global_size;
  if constexpr (InFW == sygraph::frontier::frontier_view::vertex) {
    size_t bitmap_range = in.getBitmapRange();
    size_t offsets_size = in.computeActiveFrontier();
    local_range = {bitmap_range * coarsening_factor};
    global_size = offsets_size * bitmap_range;
  } else if constexpr (InFW == sygraph::frontier::frontier_view::graph) {
    local_range = {256};
    global_size = num_nodes;
  } else {
    throw std::runtime_error("Invalid frontier view");
  }
  sycl::range<1> global_range{global_size > local_range[0] ? global_size + (local_range[0] - (global_size % local_range[0])) : local_range[0]};

  Context<InFW, OutFW, decltype(in_dev_frontier), decltype(out_dev_frontier)> context{num_nodes, in_dev_frontier, out_dev_frontier};
  using bitmap_kernel_t = BitmapKernel<InFW, OutFW, T, decltype(context), decltype(graph_dev), LambdaT>;

  auto e = q.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<uint32_t, 1> n_edges_wg{local_range, cgh};
    sycl::local_accessor<uint32_t, 1> n_edges_sg{local_range, cgh};
    sycl::local_accessor<bool, 1> visited{local_range, cgh};
    sycl::local_accessor<T, 1> subgroup_reduce{local_range, cgh};
    sycl::local_accessor<uint32_t, 1> subgroup_reduce_tail{types::detail::MAX_SUBGROUPS, cgh};
    sycl::local_accessor<uint32_t, 1> subgroup_ids{local_range, cgh};
    sycl::local_accessor<T, 1> workgroup_reduce{local_range, cgh};
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

template<graph::detail::GraphConcept GraphT, typename T, sygraph::frontier::frontier_type FT, typename LambdaT>
sygraph::Event
launchVectorKernel(GraphT& graph, const sygraph::frontier::Frontier<T, FT>& in, const sygraph::frontier::Frontier<T, FT>& out, LambdaT&& functor) {
  sycl::queue& q = graph.getQueue();

  const T* active_elements = in.getVector();
  size_t active_elements_size = in.getVectorSize();

  auto in_dev_frontier = in.getDeviceFrontier();
  auto out_dev_frontier = out.getDeviceFrontier();
  auto graph_dev = graph.getDeviceGraph();

  using vector_kernel_t = VectorKernel<T, decltype(in_dev_frontier), decltype(graph_dev), LambdaT>;

  auto e = q.submit([&](sycl::handler& cgh) {
    sycl::range<1> local_range{1024}; // TODO: [!] Tune on this value, or compute it dynamically
    sycl::range<1> global_range{
        active_elements_size > local_range[0] ? active_elements_size + (local_range[0] - (active_elements_size % local_range[0])) : local_range[0]};

    sycl::local_accessor<uint32_t, 1> n_edges_local{local_range, cgh};
    sycl::local_accessor<bool, 1> visited{local_range, cgh};
    sycl::local_accessor<T, 1> active_elements_local{local_range, cgh};
    sycl::local_accessor<uint32_t, 1> ids{local_range, cgh};
    sycl::local_accessor<uint32_t, 1> active_elements_local_tail{types::detail::MAX_SUBGROUPS, cgh};
    sycl::local_accessor<T, 1> pad{out_dev_frontier.getVectorMaxSize(), cgh};
    sycl::local_accessor<uint32_t, 1> pad_tail{1, cgh};

    cgh.parallel_for(sycl::nd_range<1>{global_range, local_range},
                     vector_kernel_t(active_elements,
                                     active_elements_size,
                                     in_dev_frontier,
                                     out_dev_frontier,
                                     graph_dev,
                                     n_edges_local,
                                     visited,
                                     active_elements_local,
                                     active_elements_local_tail,
                                     ids,
                                     pad,
                                     pad_tail,
                                     std::forward<LambdaT>(functor)));
  });

  return {e};
}

template<sygraph::frontier::frontier_view InFW,
         sygraph::frontier::frontier_view OutFW,
         graph::detail::GraphConcept GraphT,
         typename T,
         typename LambdaT>
sygraph::Event frontier(GraphT& graph,
                        const sygraph::frontier::Frontier<T, sygraph::frontier::frontier_type::bitvec>& in,
                        const sygraph::frontier::Frontier<T, sygraph::frontier::frontier_type::bitvec>& out,
                        LambdaT&& functor) {
  if (in.useVector()) { return launchVectorKernel(graph, in, out, std::forward<LambdaT>(functor)); }
  return launchBitmapKernel<InFW, OutFW>(graph, in, out, std::forward<LambdaT>(functor));
}

template<sygraph::frontier::frontier_view InFW,
         sygraph::frontier::frontier_view OutFW,
         graph::detail::GraphConcept GraphT,
         typename T,
         typename LambdaT>
sygraph::Event frontier(GraphT& graph,
                        const sygraph::frontier::Frontier<T, sygraph::frontier::frontier_type::bitmap>& in,
                        const sygraph::frontier::Frontier<T, sygraph::frontier::frontier_type::bitmap>& out,
                        LambdaT&& functor) {
  return launchBitmapKernel<InFW, OutFW, T>(graph, in, out, std::forward<LambdaT>(functor));
}

template<sygraph::frontier::frontier_view InFW,
         sygraph::frontier::frontier_view OutFW,
         graph::detail::GraphConcept GraphT,
         typename T,
         typename LambdaT>
sygraph::Event frontier(GraphT& graph,
                        const sygraph::frontier::Frontier<T, sygraph::frontier::frontier_type::hierachic_bitmap>& in,
                        const sygraph::frontier::Frontier<T, sygraph::frontier::frontier_type::hierachic_bitmap>& out,
                        LambdaT&& functor) {
  return launchBitmapKernel<InFW, OutFW, T>(graph, in, out, std::forward<LambdaT>(functor));
}

template<sygraph::frontier::frontier_view OutFW, graph::detail::GraphConcept GraphT, typename T, typename LambdaT>
sygraph::Event
vertices(GraphT& graph, const sygraph::frontier::Frontier<T, sygraph::frontier::frontier_type::hierachic_bitmap>& out, LambdaT&& functor) {
  auto in = sygraph::frontier::Frontier<T, sygraph::frontier::frontier_type::none>{};
  return launchBitmapKernel<sygraph::frontier::frontier_view::graph, OutFW, T>(graph, in, out, std::forward<LambdaT>(functor));
}

} // namespace workgroup_mapped
} // namespace detail
} // namespace advance
} // namespace operators
} // namespace v0
} // namespace sygraph