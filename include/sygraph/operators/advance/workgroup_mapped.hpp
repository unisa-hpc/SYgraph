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

template<typename T, typename FrontierDevT, graph::detail::DeviceGraphConcept GraphDevT, typename LambdaT>
struct vector_kernel {
  vector_kernel(T* active_elements,
                size_t active_elements_size,
                FrontierDevT in_dev_frontier,
                FrontierDevT out_dev_frontier,
                GraphDevT graph_dev,
                sycl::local_accessor<size_t, 1> n_edges_local,
                sycl::local_accessor<bool, 1> visited,
                sycl::local_accessor<T, 1> active_elements_local,
                sycl::local_accessor<T, 1> work_group_reduce,
                sycl::local_accessor<size_t, 1> work_group_reduce_tail,
                LambdaT&& functor)
      : active_elements{active_elements}, active_elements_size{active_elements_size}, in_dev_frontier{in_dev_frontier},
        out_dev_frontier{out_dev_frontier}, graph_dev{graph_dev}, n_edges_local{n_edges_local}, visited{visited},
        active_elements_local{active_elements_local}, work_group_reduce{work_group_reduce}, work_group_reduce_tail{work_group_reduce_tail},
        functor{std::forward<LambdaT>(functor)} {}

  void operator()(sycl::nd_item<1> item) const {
    // 0. retrieve global and local ids
    size_t gid = item.get_global_linear_id();
    size_t lid = item.get_local_linear_id();
    size_t local_range = item.get_local_range(0);
    auto group = item.get_group();
    auto group_id = item.get_group_linear_id();
    auto subgroup = item.get_sub_group();
    auto subgroup_id = subgroup.get_group_id();
    size_t subgroup_size = subgroup.get_local_range()[0];
    size_t llid = subgroup.get_local_linear_id();

    // 1. load number of edges in local memory
    if (lid == 0) { work_group_reduce_tail[0] = 0; }
    if (gid < active_elements_size) {
      T element = active_elements[gid];
      n_edges_local[lid] = graph_dev.getDegree(element);
      active_elements_local[lid] = element;
      visited[lid] = false;
    } else {
      n_edges_local[lid] = 0;
      visited[lid] = true;
    }

    // // 1.5 compute nodes to be computed by all the item in the workgroup
    // sycl::atomic_ref<size_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group>
    // sg_tail{work_group_reduce_tail[0]}; if (n_edges_local[lid] >= local_range && !visited[lid]) {
    //   work_group_reduce[sg_tail++] = lid;
    // }

    // sycl::group_barrier(group); // synchronize

    // // 2. process elements with more than local_range edges
    // for (size_t i = 0; i < sg_tail.load(); i++) { // TODO: [!!!!] for some reason this slows a lot
    // the performances (6ms)
    //   size_t vertex_id = work_group_reduce[i];
    //   auto vertex = active_elements_local[vertex_id];
    //   size_t n_edges = n_edges_local[vertex_id];
    //   size_t private_slice = n_edges / local_range;
    //   auto start = graph_dev.begin(vertex) + (private_slice * lid);
    //   auto end = lid == local_range - 1 ? graph_dev.end(vertex) : start + private_slice;

    //   for (auto n = start; n != end; ++n) {
    //     auto edge = n.get_index();
    //     auto weight = graph_dev.getEdgeWeight(edge);
    //     auto neighbor = *n;
    //     if (functor(vertex, neighbor, edge, weight)) {
    //       out_dev_frontier.insert(neighbor); // this might be the bottleneck
    //     }
    //   }
    //   if (lid == 0) {
    //     visited[vertex_id] = true;
    //   }
    // }
    sycl::group_barrier(group);

    // 3. process elements with less than local_range edges but more than one subgroup size edges
    for (size_t i = 0; i < subgroup_size; i++) {
      size_t vertex_id = subgroup_id * subgroup_size + i;
      if (!visited[vertex_id] && n_edges_local[vertex_id] >= subgroup_size) {
        auto vertex = active_elements_local[vertex_id];
        size_t n_edges = n_edges_local[vertex_id];
        size_t private_slice = n_edges / subgroup_size;
        auto start = graph_dev.begin(vertex) + (private_slice * llid);
        auto end = llid == subgroup_size - 1 ? graph_dev.end(vertex) : start + private_slice;

        for (auto n = start; n != end; ++n) {
          auto edge = n.get_index();
          auto weight = graph_dev.getEdgeWeight(edge);
          auto neighbor = *n;
          if (functor(vertex, neighbor, edge, weight)) { out_dev_frontier.insert(neighbor); }
        }
        sycl::group_barrier(subgroup);
        if (llid == i) { visited[vertex_id] = true; }
      }
    }

    // 4. process the rest
    if (!visited[lid]) {
      auto vertex = active_elements_local[lid];
      auto start = graph_dev.begin(vertex);
      auto end = graph_dev.end(vertex);

      for (auto n = start; n != end; ++n) {
        auto edge = n.get_index();
        auto weight = graph_dev.getEdgeWeight(edge);
        auto neighbor = *n;
        if (functor(vertex, neighbor, edge, weight)) { out_dev_frontier.insert(neighbor); }
      }
    }
  }

  T* active_elements;
  size_t active_elements_size;
  FrontierDevT in_dev_frontier;
  FrontierDevT out_dev_frontier;
  GraphDevT graph_dev;
  sycl::local_accessor<size_t, 1> n_edges_local;
  sycl::local_accessor<bool, 1> visited;
  sycl::local_accessor<T, 1> active_elements_local;
  sycl::local_accessor<T, 1> work_group_reduce;
  sycl::local_accessor<size_t, 1> work_group_reduce_tail;
  LambdaT functor;
};


template<typename T, typename FrontierDevT, graph::detail::DeviceGraphConcept GraphDevT, typename LambdaT>
struct bitmap_kernel {
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
    const int* bitmap_offsets = in_dev_frontier.getOffsets();
    const size_t bitmap_range = in_dev_frontier.getBitmapRange();

    const size_t coarsening_factor = wgroup_size / bitmap_range;
    const size_t acutal_id_offset = wgroup_id * coarsening_factor + lid / bitmap_range;
    const size_t assigned_vertex = bitmap_offsets[acutal_id_offset] * bitmap_range + lid % bitmap_range;

    // 1. load number of edges in local memory
    if (sgroup.leader()) { active_elements_tail[sgroup_id] = 0; }
    if (wgroup.leader()) { work_group_reduce_tail[0] = 0; }

    sycl::atomic_ref<size_t, sycl::memory_order::relaxed, sycl::memory_scope::sub_group> sg_tail{active_elements_tail[sgroup_id]};
    sycl::atomic_ref<size_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group> wg_tail{work_group_reduce_tail[0]};

    const size_t offset = sgroup_id * sgroup_size;
    if (assigned_vertex < num_nodes && in_dev_frontier.check(assigned_vertex)) {
      size_t n_edges = graph_dev.getDegree(assigned_vertex);
      if (n_edges >= wgroup_size * sgroup_size) { // assign to the workgroup
        size_t loc = wg_tail.fetch_add(1);
        work_group_reduce[loc] = assigned_vertex;
      } else if (n_edges >= sgroup_size) { // assign to the subgroup
        size_t loc = sg_tail.fetch_add(1);
        n_edges_local[offset + loc] = n_edges;
        active_elements_local[offset + loc] = assigned_vertex;
      }
      visited[lid] = false;
    } else {
      visited[lid] = true;
    }

    sycl::group_barrier(wgroup);
    for (size_t i = 0; i < wg_tail.load(); i++) { // TODO: fix this
      auto vertex = work_group_reduce[i];
      size_t n_edges = graph_dev.getDegree(vertex);
      size_t private_slice = n_edges / wgroup_size;
      auto start = graph_dev.begin(vertex) + (private_slice * lid);
      auto end = lid == wgroup_size - 1 ? graph_dev.end(vertex) : start + private_slice;

      for (auto n = start; n != end; ++n) {
        auto edge = n.get_index();
        auto weight = graph_dev.getEdgeWeight(edge);
        auto neighbor = *n;
        if (functor(vertex, neighbor, edge, weight)) { out_dev_frontier.insert(neighbor); }
      }
      if (wgroup.leader()) { visited[vertex % wgroup_size] = true; }
    }

    sycl::group_barrier(sgroup);

    for (size_t i = 0; i < active_elements_tail[sgroup_id]; i++) { // active_elements_tail[subgroup_id] is always less or equal than subgroup_size
      size_t vertex_id = offset + i;
      auto vertex = active_elements_local[vertex_id];
      size_t n_edges = n_edges_local[vertex_id];
      // if (n_edges < sgroup_size) { continue; }
      size_t private_slice = n_edges / sgroup_size;
      auto start = graph_dev.begin(vertex) + (private_slice * llid);
      auto end = llid == sgroup_size - 1 ? graph_dev.end(vertex) : start + private_slice;

      for (auto n = start; n != end; ++n) {
        auto edge = n.get_index();
        auto weight = graph_dev.getEdgeWeight(edge);
        auto neighbor = *n;
        if (functor(vertex, neighbor, edge, weight)) { out_dev_frontier.insert(neighbor); }
      }
      if (sgroup.leader()) { visited[vertex % wgroup_size] = true; }
    }
    sycl::group_barrier(sgroup);

    if (!visited[lid]) {
      auto vertex = assigned_vertex;
      auto start = graph_dev.begin(vertex);
      auto end = graph_dev.end(vertex);

      for (auto n = start; n != end; ++n) {
        auto edge = n.get_index();
        auto weight = graph_dev.getEdgeWeight(edge);
        auto neighbor = *n;
        if (functor(vertex, neighbor, edge, weight)) { out_dev_frontier.insert(neighbor); }
      }
    }
  }

  size_t num_nodes;
  FrontierDevT in_dev_frontier;
  FrontierDevT out_dev_frontier;
  GraphDevT graph_dev;
  sycl::local_accessor<size_t, 1> n_edges_local;
  sycl::local_accessor<T, 1> active_elements_local;
  sycl::local_accessor<size_t, 1> active_elements_tail;
  sycl::local_accessor<bool, 1> visited;
  sycl::local_accessor<T, 1> work_group_reduce;
  sycl::local_accessor<size_t, 1> work_group_reduce_tail;
  LambdaT functor;
};


namespace workgroup_mapped {

template<graph::detail::GraphConcept GraphT, typename T, sygraph::frontier::FrontierType F, typename LambdaT>
sygraph::event launchBitmapKernel(GraphT& graph,
                                  const sygraph::frontier::Frontier<T, sygraph::frontier::FrontierView::vertex, F>& in,
                                  const sygraph::frontier::Frontier<T, sygraph::frontier::FrontierView::vertex, F>& out,
                                  LambdaT&& functor) {
  if constexpr (F != sygraph::frontier::FrontierType::bitmap && F != sygraph::frontier::FrontierType::bitvec) {
    throw std::runtime_error("Invalid frontier type");
  }

  sycl::queue& q = graph.getQueue();

  size_t bitmap_range = in.getBitmapRange();
  size_t num_nodes = graph.getVertexCount();

  auto in_dev_frontier = in.getDeviceFrontier();
  auto out_dev_frontier = out.getDeviceFrontier();
  auto graph_dev = graph.getDeviceGraph();

  using bitmap_kernel_t = bitmap_kernel<T, decltype(in_dev_frontier), decltype(graph_dev), LambdaT>;

  size_t offsets_size = in.computeActiveFrontier();

  auto e = q.submit([&](sycl::handler& cgh) {
    sycl::range<1> local_range{bitmap_range};
    size_t global_size = offsets_size * bitmap_range;
    sycl::range<1> global_range{global_size > local_range[0] ? global_size + (local_range[0] - (global_size % local_range[0])) : local_range[0]};

    sycl::local_accessor<size_t, 1> n_edges_local{local_range, cgh};
    sycl::local_accessor<T, 1> active_elements_local{local_range, cgh};
    sycl::local_accessor<size_t, 1> active_elements_tail{types::detail::MAX_SUBGROUPS, cgh};
    sycl::local_accessor<bool, 1> visited{local_range, cgh};
    sycl::local_accessor<T, 1> work_group_reduce{local_range, cgh};
    sycl::local_accessor<size_t, 1> work_group_reduce_tail{1, cgh};


    cgh.parallel_for<class workgroup_mapped_advance_kernel>(sycl::nd_range<1>{global_range, local_range},
                                                            bitmap_kernel_t{num_nodes,
                                                                            in_dev_frontier,
                                                                            out_dev_frontier,
                                                                            graph_dev,
                                                                            n_edges_local,
                                                                            active_elements_local,
                                                                            active_elements_tail,
                                                                            visited,
                                                                            work_group_reduce,
                                                                            work_group_reduce_tail,
                                                                            std::forward<LambdaT>(functor)});
  });
  return {e};
}

template<graph::detail::GraphConcept GraphT, typename T, sygraph::frontier::FrontierType FT, typename LambdaT>
sygraph::event launchVectorKernel(GraphT& graph,
                                  const sygraph::frontier::Frontier<T, sygraph::frontier::FrontierView::vertex, FT>& in,
                                  const sygraph::frontier::Frontier<T, sygraph::frontier::FrontierView::vertex, FT>& out,
                                  LambdaT&& functor) {
  sycl::queue& q = graph.getQueue();

  T* active_elements = in.getDeviceFrontier().getVector();
  size_t active_elements_size = in.getDeviceFrontier().getVectorSize();

  auto inDevFrontier = in.getDeviceFrontier();
  auto outDevFrontier = out.getDeviceFrontier();
  auto graphDev = graph.getDeviceGraph();

  using vector_kernel_t = vector_kernel<T, decltype(inDevFrontier), decltype(graphDev), LambdaT>;

  auto e = q.submit([&](sycl::handler& cgh) {
    sycl::range<1> local_range{64}; // TODO: [!] Tune on this value, or compute it dynamically
    sycl::range<1> global_range{
        active_elements_size > local_range[0] ? active_elements_size + (local_range[0] - (active_elements_size % local_range[0])) : local_range[0]};

    sycl::local_accessor<size_t, 1> n_edges_local{local_range, cgh};
    sycl::local_accessor<bool, 1> visited{local_range, cgh};
    sycl::local_accessor<T, 1> active_elements_local{local_range, cgh};
    sycl::local_accessor<T, 1> work_group_reduce{local_range, cgh};
    sycl::local_accessor<size_t, 1> work_group_reduce_tail{1, cgh};

    cgh.parallel_for(sycl::nd_range<1>{global_range, local_range},
                     vector_kernel_t(active_elements,
                                     active_elements_size,
                                     inDevFrontier,
                                     outDevFrontier,
                                     graphDev,
                                     n_edges_local,
                                     visited,
                                     active_elements_local,
                                     work_group_reduce,
                                     work_group_reduce_tail,
                                     std::forward<LambdaT>(functor)));
  });

  return {e};
}

template<graph::detail::GraphConcept GraphT, typename T, typename LambdaT>
sygraph::event vertex(GraphT& graph,
                      const sygraph::frontier::Frontier<T, sygraph::frontier::FrontierView::vertex, sygraph::frontier::FrontierType::bitvec>& in,
                      const sygraph::frontier::Frontier<T, sygraph::frontier::FrontierView::vertex, sygraph::frontier::FrontierType::bitvec>& out,
                      LambdaT&& functor) {
  if (in.getDeviceFrontier().useVector()) {
    return launchVectorKernel(graph, in, out, std::forward<LambdaT>(functor));
  } else {
    return launchBitmapKernel(graph, in, out, std::forward<LambdaT>(functor));
  }
}


template<graph::detail::GraphConcept GraphT, typename T, typename LambdaT>
sygraph::event vertex(GraphT& graph,
                      const sygraph::frontier::Frontier<T, sygraph::frontier::FrontierView::vertex, sygraph::frontier::FrontierType::bitmap>& in,
                      const sygraph::frontier::Frontier<T, sygraph::frontier::FrontierView::vertex, sygraph::frontier::FrontierType::bitmap>& out,
                      LambdaT&& functor) {
  return launchBitmapKernel(graph, in, out, std::forward<LambdaT>(functor));
}

} // namespace workgroup_mapped
} // namespace detail
} // namespace advance
} // namespace operators
} // namespace v0
} // namespace sygraph