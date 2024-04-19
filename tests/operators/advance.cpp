#include <sycl/sycl.hpp>
#include <sygraph/sygraph.hpp>

int main() {
  sycl::queue q {sycl::gpu_selector_v};

  auto mat = sygraph::io::storage::matrices::symmetric_6nodes;
  std::istringstream iss(mat.data());
  auto csr = sygraph::io::csr::from_matrix<uint, uint, uint>(iss);
  auto G = sygraph::graph::build::from_csr<sygraph::memory::space::shared>(q, csr);
  
  using type_t = decltype(G)::vertex_t;
  sycl::buffer<type_t> nighbours_count_buf(G.get_vertex_count());
  sycl::buffer<type_t> first_neighbours_buf(G.get_vertex_count());


  using load_balance_t = sygraph::operators::LoadBalancer;
  using direction_t = sygraph::operators::Direction;
  using frontier_view_t = sygraph::frontier::FrontierView;
  using frontier_impl_t = sygraph::frontier::FrontierType;

  auto inFrontier = sygraph::frontier::make_frontier<frontier_view_t::vertex, frontier_impl_t::bitmap>(q, G);
  auto outFrontier = sygraph::frontier::make_frontier<frontier_view_t::edge, frontier_impl_t::bitmap>(q, G);

  bool* visited = sycl::malloc_shared<bool>(G.get_vertex_count(), q);
  size_t* distances = sycl::malloc_shared<size_t>(G.get_vertex_count(), q);

  auto device_graph = G.get_device_graph();

  inFrontier.insert(0);
  distances[0] = 0;
  visited[0] = true;

  sygraph::operators::advance::push<load_balance_t::workitem_mapped>(G, inFrontier, outFrontier, [=](auto id) {
    auto start_idx = device_graph.get_first_neighbour_idx(id);
    auto end_idx = start_idx + device_graph.get_neighbour_count(id);

    for (auto i = start_idx; i < end_idx; i++) {
      auto neighbour = device_graph.get_column_indices()[i];
      if (!visited[neighbour]) {
        visited[neighbour] = true;
        distances[neighbour] = distances[id] + 1;
      }
    }
  });

  for (size_t i = 0; i < G.get_vertex_count(); i++) {
    std::cout << "Vertex " << i << " has distance " << distances[i] << std::endl;
  }

  sycl::free(visited, q);
  sycl::free(distances, q);
}