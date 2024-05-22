#pragma once

#include <sycl/sycl.hpp>
#include <memory>

#include <sygraph/sycl/event.hpp>
#include <sygraph/graph/graph.hpp>
#include <sygraph/operators/config.hpp>
#include <sygraph/utils/types.hpp>

namespace sygraph {
inline namespace v0 {
namespace operators {

namespace advance {

namespace detail {

namespace workgroup_mapped {

namespace bitmap {

namespace v0 {
template <typename graph_t,
          typename T,
          typename lambda_t>
sygraph::event vertex(graph_t& graph,   
                      const sygraph::frontier::Frontier<T, sygraph::frontier::FrontierView::edge, sygraph::frontier::FrontierType::bitmap>& in, 
                      const sygraph::frontier::Frontier<T, sygraph::frontier::FrontierView::edge, sygraph::frontier::FrontierType::bitmap>& out, 
                      lambda_t&& functor) {  
  sycl::queue& q = graph.get_queue();

  size_t active_elements_size = in.get_num_active_elements();

  T* active_elements;
  if (!in.self_allocated()) {
    active_elements = sycl::malloc_device<T>(active_elements_size, q);
  }
  in.get_active_elements(active_elements, active_elements_size);

  auto e = q.submit([&](sycl::handler& cgh) {

    sycl::range<1> local_range{64}; // TODO: [!] Tune on this value, or compute it dynamically
    sycl::range<1> global_range{active_elements_size > local_range[0] ? active_elements_size + (local_range[0] - (active_elements_size % local_range[0])) : local_range[0]};

    auto inDevFrontier = in.get_device_frontier();
    auto outDevFrontier = out.get_device_frontier();
    auto graphDev = graph.get_device_graph();

    sycl::local_accessor<size_t, 1> n_edges_local {local_range, cgh};
    sycl::local_accessor<bool, 1> visited {local_range, cgh};
    sycl::local_accessor<T, 1> active_elements_local {local_range, cgh};
    sycl::local_accessor<T, 1> work_group_reduce {local_range, cgh};
    sycl::local_accessor<size_t, 1> work_group_reduce_tail {1, cgh};
    
    cgh.parallel_for<class workgroup_mapped_advance_kernel>(sycl::nd_range<1>{global_range, local_range}, [=](sycl::nd_item<1> item) {
      // 0. retrieve global and local ids
      size_t gid = item.get_global_linear_id();
      size_t lid = item.get_local_linear_id();
      size_t local_range = item.get_local_range(0);
      auto group = item.get_group();
      auto group_id = item.get_group_linear_id();
      auto subgroup = item.get_sub_group();
      auto subgroup_id = subgroup.get_group_id();
      size_t subgroup_size = subgroup.get_local_range()[0];
      size_t sgid = subgroup.get_local_linear_id();

      // 1. load number of edges in local memory
      if (lid == 0) {
        work_group_reduce_tail[0] = 0;
      }
      if (gid < active_elements_size) {
        T element = active_elements[gid];
        n_edges_local[lid] = graphDev.get_neighbors_count(element);
        active_elements_local[lid] = element;
        visited[lid] = false;
      } else {
        n_edges_local[lid] = 0;
        visited[lid] = true;
      }

      // // 1.5 compute nodes to be computed by all the item in the workgroup
      // sycl::atomic_ref<size_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group> tail{work_group_reduce_tail[0]};
      // if (n_edges_local[lid] >= local_range && !visited[lid]) {
      //   work_group_reduce[tail++] = lid;
      // }

      // sycl::group_barrier(group); // synchronize

      // // 2. process elements with more than local_range edges
      // for (size_t i = 0; i < tail.load(); i++) { // TODO: [!!!!] for some reason this slows a lot the performances (6ms)
      //   size_t vertex_id = work_group_reduce[i];
      //   auto vertex = active_elements_local[vertex_id];
      //   size_t n_edges = n_edges_local[vertex_id];
      //   size_t private_slice = n_edges / local_range;
      //   auto start = graphDev.begin(vertex) + (private_slice * lid);
      //   auto end = lid == local_range - 1 ? graphDev.end(vertex) : start + private_slice;

      //   for (auto n = start; n != end; ++n) {
      //     auto edge = n.get_index();
      //     auto weight = graphDev.get_edge_weight(edge);
      //     auto neighbor = *n;
      //     if (functor(vertex, neighbor, edge, weight)) {
      //       outDevFrontier.insert(neighbor); // this might be the bottleneck
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
        if (!visited[vertex_id] &&  n_edges_local[vertex_id] >= subgroup_size) {
          auto vertex = active_elements_local[vertex_id];
          size_t n_edges = n_edges_local[vertex_id];
          size_t private_slice = n_edges / subgroup_size;
          auto start = graphDev.begin(vertex) + (private_slice * sgid);
          auto end = sgid == subgroup_size - 1 ? graphDev.end(vertex) : start + private_slice;

          for (auto n = start; n != end; ++n) {
            auto edge = n.get_index();
            auto weight = graphDev.get_edge_weight(edge);
            auto neighbor = *n;
            if (functor(vertex, neighbor, edge, weight)) {
              outDevFrontier.insert(neighbor);
            }
          }
          sycl::group_barrier(subgroup);
          if (sgid == i) {
            visited[vertex_id] = true;
          }
        }
      }

      // 4. process the rest
      if (!visited[lid]) {
        auto vertex = active_elements_local[lid];
        auto start = graphDev.begin(vertex);
        auto end = graphDev.end(vertex);

        for (auto n = start; n != end; ++n) {
          auto edge = n.get_index();
          auto weight = graphDev.get_edge_weight(edge);
          auto neighbor = *n;
          if (functor(vertex, neighbor, edge, weight)) {
            outDevFrontier.insert(neighbor);
          }
        }
      }
    });
  });

  if (!in.self_allocated()) {
    sycl::free(active_elements, q);
  }
  return {e};
}
} // namespace v0

inline namespace v1 {
template <typename graph_t,
          typename T,
          typename lambda_t>
sygraph::event vertex(graph_t& graph,   
                      const sygraph::frontier::Frontier<T, sygraph::frontier::FrontierView::vertex, sygraph::frontier::FrontierType::bitmap>& in, 
                      const sygraph::frontier::Frontier<T, sygraph::frontier::FrontierView::vertex, sygraph::frontier::FrontierType::bitmap>& out, 
                      lambda_t&& functor) {  
  sycl::queue& q = graph.get_queue();

  size_t bitmap_range = in.get_bitmap_range();
  size_t num_nodes = graph.get_vertex_count();
  constexpr size_t COARSENING_FACTOR = 8;

  auto e = q.submit([&](sycl::handler& cgh) {

    sycl::range<1> local_range{bitmap_range * COARSENING_FACTOR};
    sycl::range<1> global_range{num_nodes > local_range[0] ? num_nodes + (local_range[0] - (num_nodes % local_range[0])) : local_range[0]};

    auto inDevFrontier = in.get_device_frontier();
    auto outDevFrontier = out.get_device_frontier();
    auto graphDev = graph.get_device_graph();

    sycl::local_accessor<size_t, 1> n_edges_local {local_range, cgh};
    sycl::local_accessor<T, 1> active_elements_local {local_range, cgh};
    sycl::local_accessor<size_t, 1> active_elements_tail {local_range / 8, cgh};
    sycl::local_accessor<bool, 1> visited {local_range, cgh};
    sycl::local_accessor<T, 1> work_group_reduce {local_range, cgh};
    sycl::local_accessor<size_t, 1> work_group_reduce_tail {1, cgh};
    
    cgh.parallel_for<class workgroup_mapped_advance_kernel>(sycl::nd_range<1>{global_range, local_range}, [=](sycl::nd_item<1> item) {
      // 0. retrieve global and local ids
      size_t gid = item.get_global_linear_id();
      size_t lid = item.get_local_linear_id();
      size_t local_range = item.get_local_range(0);
      auto group = item.get_group();
      auto group_id = item.get_group_linear_id();
      auto subgroup = item.get_sub_group();
      auto subgroup_id = subgroup.get_group_id();
      size_t subgroup_size = subgroup.get_local_range()[0];
      size_t sgid = subgroup.get_local_linear_id();

      // 1. load number of edges in local memory
      if (subgroup.leader()) {
        active_elements_tail[subgroup_id] = 0;
      }
      if (group.leader()) {
        work_group_reduce_tail[0] = 0;
      }

      sycl::atomic_ref<size_t, sycl::memory_order::acq_rel, sycl::memory_scope::sub_group> tail{active_elements_tail[subgroup_id]};
      sycl::atomic_ref<size_t, sycl::memory_order::acq_rel, sycl::memory_scope::work_group> tail_global{active_elements_tail[0]};

      size_t offset = subgroup_id * subgroup_size;
      if (gid < num_nodes && inDevFrontier.check(gid)) {
        size_t n_edges = graphDev.get_neighbors_count(gid);
        // if (n_edges > local_range * 2) { // assign to the workgroup
        //   work_group_reduce[tail_global++] = gid;
        // } else { // assign to the subgroup
        //   size_t loc = tail.fetch_add(1);
        //   n_edges_local[offset + loc] = n_edges;
        //   active_elements_local[offset + loc] = gid;
        // }
        size_t loc = tail.fetch_add(1);
        n_edges_local[offset + loc] = n_edges;
        active_elements_local[offset + loc] = gid;
        visited[lid] = false;
      } else {
        visited[lid] = true;
      }

      sycl::group_barrier(subgroup);

      for (size_t i = 0; i < active_elements_tail[subgroup_id]; i++) { // active_elements_tail[subgroup_id] is always less or equal than subgroup_size
        size_t vertex_id = offset + i;
        auto vertex = active_elements_local[vertex_id];
        size_t n_edges = n_edges_local[vertex_id];
        if (n_edges < subgroup_size) {
          continue;
        }
        size_t private_slice = n_edges / subgroup_size;
        auto start = graphDev.begin(vertex) + (private_slice * sgid);
        auto end = sgid == subgroup_size - 1 ? graphDev.end(vertex) : start + private_slice;

        for (auto n = start; n != end; ++n) {
          auto edge = n.get_index();
          auto weight = graphDev.get_edge_weight(edge);
          auto neighbor = *n;
          if (functor(vertex, neighbor, edge, weight)) {
            outDevFrontier.insert(neighbor);
          }
        }
        if (subgroup.leader()) {
          visited[vertex % local_range] = true;
        }
      }
      sycl::group_barrier(group);

      // for (size_t i = 0; i < work_group_reduce_tail[0]; i++) { // TODO: fix this
      //   auto vertex = work_group_reduce[i];
      //   size_t n_edges = graphDev.get_neighbors_count(vertex);
      //   size_t private_slice = n_edges / local_range;
      //   auto start = graphDev.begin(vertex) + (private_slice * lid);
      //   auto end = lid == local_range - 1 ? graphDev.end(vertex) : start + private_slice;

      //   for (auto n = start; n != end; ++n) {
      //     auto edge = n.get_index();
      //     auto weight = graphDev.get_edge_weight(edge);
      //     auto neighbor = *n;
      //     if (functor(vertex, neighbor, edge, weight)) {
      //       outDevFrontier.insert(neighbor);
      //     }
      //   }
      //   if (group.leader()) {
      //     visited[vertex % local_range] = true;
      //   }
      // }
      // sycl::group_barrier(group);

      if (!visited[lid]) {
        auto vertex = gid;
        auto start = graphDev.begin(vertex);
        auto end = graphDev.end(vertex);

        for (auto n = start; n != end; ++n) {
          auto edge = n.get_index();
          auto weight = graphDev.get_edge_weight(edge);
          auto neighbor = *n;
          if (functor(vertex, neighbor, edge, weight)) {
            outDevFrontier.insert(neighbor);
          }
        }
      }
    });
  });
  return {e};
}
} // namespace v1

} // namespace bitmap
} // namespace workitem_mapped
} // namespace detail  
} // namespace advance
} // namespace operators
} // namespace v0
} // namespace sygraph