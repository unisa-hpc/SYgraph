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

template <typename T, typename frontier_dev_t, typename graph_dev_t, typename lambda_t>
struct vector_kernel {

  vector_kernel(T* active_elements, 
                size_t active_elements_size, 
                frontier_dev_t inDevFrontier, 
                frontier_dev_t outDevFrontier, 
                graph_dev_t graphDev,
                sycl::local_accessor<size_t, 1> n_edges_local, 
                sycl::local_accessor<bool, 1> visited, 
                sycl::local_accessor<T, 1> active_elements_local, 
                sycl::local_accessor<T, 1> work_group_reduce, 
                sycl::local_accessor<size_t, 1> work_group_reduce_tail, 
                lambda_t&& functor) 
    : active_elements{active_elements}, 
      active_elements_size{active_elements_size}, 
      inDevFrontier{inDevFrontier}, 
      outDevFrontier{outDevFrontier}, 
      graphDev{graphDev},
      n_edges_local{n_edges_local}, 
      visited{visited}, 
      active_elements_local{active_elements_local}, 
      work_group_reduce{work_group_reduce}, 
      work_group_reduce_tail{work_group_reduce_tail}, 
      functor{std::forward<lambda_t>(functor)} {}

  void operator() (sycl::nd_item<1> item) const {
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
      n_edges_local[lid] = graphDev.get_degree(element);
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
  }

  T* active_elements;
  size_t active_elements_size;
  frontier_dev_t inDevFrontier;
  frontier_dev_t outDevFrontier;
  graph_dev_t graphDev;
  sycl::local_accessor<size_t, 1> n_edges_local;
  sycl::local_accessor<bool, 1> visited;
  sycl::local_accessor<T, 1> active_elements_local;
  sycl::local_accessor<T, 1> work_group_reduce;
  sycl::local_accessor<size_t, 1> work_group_reduce_tail;
  lambda_t functor;
};


template <typename T, typename frontier_dev_t, typename graph_dev_t, typename lambda_t>
struct bitmap_kernel {

  void operator() (sycl::nd_item<1> item) const {
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
    int* bitmap_offsets = inDevFrontier.get_offsets();

    size_t actual_id = bitmap_offsets[group_id] * inDevFrontier.get_bitmap_range() + lid;

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
    if (actual_id < num_nodes && inDevFrontier.check(actual_id)) {
      size_t n_edges = graphDev.get_degree(actual_id);
      // if (n_edges > local_range * 2) { // assign to the workgroup
      //   work_group_reduce[tail_global++] = gid;
      // } else { // assign to the subgroup
      //   size_t loc = tail.fetch_add(1);
      //   n_edges_local[offset + loc] = n_edges;
      //   active_elements_local[offset + loc] = gid;
      // }
      size_t loc = tail.fetch_add(1);
      n_edges_local[offset + loc] = n_edges;
      active_elements_local[offset + loc] = actual_id;
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
    //   size_t n_edges = graphDev.get_degree(vertex);
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
    //   sycl::group_barrier(group);
    // }

    if (!visited[lid]) {
      auto vertex = actual_id;
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
  }

  size_t num_nodes;
  frontier_dev_t inDevFrontier;
  frontier_dev_t outDevFrontier;
  graph_dev_t graphDev;
  sycl::local_accessor<size_t, 1> n_edges_local;
  sycl::local_accessor<T, 1> active_elements_local;
  sycl::local_accessor<size_t, 1> active_elements_tail;
  sycl::local_accessor<bool, 1> visited;
  sycl::local_accessor<T, 1> work_group_reduce;
  sycl::local_accessor<size_t, 1> work_group_reduce_tail;
  lambda_t functor;
};


namespace workgroup_mapped {

template <typename graph_t,
          typename T,
          typename lambda_t>
sygraph::event vertex_vec(graph_t& graph,   
                      const sygraph::frontier::Frontier<T, sygraph::frontier::FrontierView::vertex, sygraph::frontier::FrontierType::bitvec>& in, 
                      const sygraph::frontier::Frontier<T, sygraph::frontier::FrontierView::vertex, sygraph::frontier::FrontierType::bitvec>& out, 
                      lambda_t&& functor) {  
  sycl::queue& q = graph.get_queue();

  T* active_elements = in.get_device_frontier().get_vector();
  size_t active_elements_size = in.get_device_frontier().get_vector_size();

  auto inDevFrontier = in.get_device_frontier();
  auto outDevFrontier = out.get_device_frontier();
  auto graphDev = graph.get_device_graph();

  using vector_kernel_t = vector_kernel<T, decltype(inDevFrontier), decltype(graphDev), lambda_t>;

  auto e = q.submit([&](sycl::handler& cgh) {

    sycl::range<1> local_range{64}; // TODO: [!] Tune on this value, or compute it dynamically
    sycl::range<1> global_range{active_elements_size > local_range[0] ? active_elements_size + (local_range[0] - (active_elements_size % local_range[0])) : local_range[0]};

    sycl::local_accessor<size_t, 1> n_edges_local {local_range, cgh};
    sycl::local_accessor<bool, 1> visited {local_range, cgh};
    sycl::local_accessor<T, 1> active_elements_local {local_range, cgh};
    sycl::local_accessor<T, 1> work_group_reduce {local_range, cgh};
    sycl::local_accessor<size_t, 1> work_group_reduce_tail {1, cgh};
    
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
                      std::forward<lambda_t>(functor)));
  });

  return {e};
}

template <typename graph_t,
          typename T,
          typename lambda_t>
sygraph::event vertex_bitmap(graph_t& graph,   
                      const sygraph::frontier::Frontier<T, sygraph::frontier::FrontierView::vertex, sygraph::frontier::FrontierType::bitvec>& in, 
                      const sygraph::frontier::Frontier<T, sygraph::frontier::FrontierView::vertex, sygraph::frontier::FrontierType::bitvec>& out,
                      lambda_t&& functor) {  
  sycl::queue& q = graph.get_queue();

  size_t bitmap_range = in.get_bitmap_range();
  size_t num_nodes = graph.get_vertex_count();
  constexpr size_t COARSENING_FACTOR = 1;
  auto inDevFrontier = in.get_device_frontier();
  auto outDevFrontier = out.get_device_frontier();
  auto graphDev = graph.get_device_graph();

  size_t offsets_size = in.compute_offsets();

  auto e = q.submit([&](sycl::handler& cgh) {

    sycl::range<1> local_range{bitmap_range * COARSENING_FACTOR};
    size_t global_size = offsets_size * local_range[0];
    sycl::range<1> global_range{global_size > local_range[0] ? global_size + (local_range[0] - (global_size % local_range[0])) : local_range[0]};
    // sycl::range<1> global_range{num_nodes > local_range[0] ? num_nodes + (local_range[0] - (num_nodes % local_range[0])) : local_range[0]};
    sycl::local_accessor<size_t, 1> n_edges_local {local_range, cgh};
    sycl::local_accessor<T, 1> active_elements_local {local_range, cgh};
    sycl::local_accessor<size_t, 1> active_elements_tail {local_range / 8, cgh};
    sycl::local_accessor<bool, 1> visited {local_range, cgh};
    sycl::local_accessor<T, 1> work_group_reduce {local_range, cgh};
    sycl::local_accessor<size_t, 1> work_group_reduce_tail {1, cgh};
    
    cgh.parallel_for(sycl::nd_range<1>{global_range, local_range}, 
      bitmap_kernel<T, decltype(inDevFrontier), decltype(graphDev), lambda_t>{num_nodes,
                      inDevFrontier, 
                      outDevFrontier, 
                      graphDev,
                      n_edges_local, 
                      active_elements_local, 
                      active_elements_tail, 
                      visited, 
                      work_group_reduce, 
                      work_group_reduce_tail, 
                      std::forward<lambda_t>(functor)});
  });
  return {e};
}
template <typename graph_t,
          typename T,
          typename lambda_t>
sygraph::event vertex(graph_t& graph,   
                      const sygraph::frontier::Frontier<T, sygraph::frontier::FrontierView::vertex, sygraph::frontier::FrontierType::bitvec>& in, 
                      const sygraph::frontier::Frontier<T, sygraph::frontier::FrontierView::vertex, sygraph::frontier::FrontierType::bitvec>& out, 
                      lambda_t&& functor) {
  if (in.get_device_frontier().use_vector()) {
    return vertex_vec(graph, in, out, std::forward<lambda_t>(functor));
  } else {
    return vertex_bitmap(graph, in, out, std::forward<lambda_t>(functor));
  }
}


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
  constexpr size_t COARSENING_FACTOR = 1;

  auto inDevFrontier = in.get_device_frontier();
  auto outDevFrontier = out.get_device_frontier();
  auto graphDev = graph.get_device_graph();

  using bitmap_kernel_t = bitmap_kernel<T, decltype(inDevFrontier), decltype(graphDev), lambda_t>;

  size_t offsets_size = in.compute_offsets();

  auto e = q.submit([&](sycl::handler& cgh) {

    sycl::range<1> local_range{bitmap_range * COARSENING_FACTOR};
    size_t global_size = offsets_size * local_range[0];
    sycl::range<1> global_range{global_size > local_range[0] ? global_size + (local_range[0] - (global_size % local_range[0])) : local_range[0]};
    // sycl::range<1> global_range{num_nodes > local_range[0] ? num_nodes + (local_range[0] - (num_nodes % local_range[0])) : local_range[0]};

    sycl::local_accessor<size_t, 1> n_edges_local {local_range, cgh};
    sycl::local_accessor<T, 1> active_elements_local {local_range, cgh};
    sycl::local_accessor<size_t, 1> active_elements_tail {local_range / 8, cgh};
    sycl::local_accessor<bool, 1> visited {local_range, cgh};
    sycl::local_accessor<T, 1> work_group_reduce {local_range, cgh};
    sycl::local_accessor<size_t, 1> work_group_reduce_tail {1, cgh};

    
    cgh.parallel_for<class workgroup_mapped_advance_kernel>(sycl::nd_range<1>{global_range, local_range}, 
      bitmap_kernel_t{num_nodes,
                      inDevFrontier, 
                      outDevFrontier, 
                      graphDev, 
                      n_edges_local, 
                      active_elements_local, 
                      active_elements_tail, 
                      visited, 
                      work_group_reduce, 
                      work_group_reduce_tail, 
                      std::forward<lambda_t>(functor)});
  });
  return {e};
}

} // namespace workitem_mapped
} // namespace detail  
} // namespace advance
} // namespace operators
} // namespace v0
} // namespace sygraph