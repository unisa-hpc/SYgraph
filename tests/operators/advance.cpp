#include <sycl/sycl.hpp>
#include <sygraph/sygraph.hpp>
#include <chrono>

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
  auto outFrontier = sygraph::frontier::make_frontier<frontier_view_t::vertex, frontier_impl_t::bitmap>(q, G);

  bool* visited = sycl::malloc_shared<bool>(G.get_vertex_count(), q);
  size_t* distances = sycl::malloc_shared<size_t>(G.get_vertex_count(), q);

  auto device_graph = G.get_device_graph();

  inFrontier.insert(0); // Start from vertex 0
  distances[0] = 0;
  visited[0] = true;

  auto start = std::chrono::high_resolution_clock::now();

  while (!inFrontier.empty()) {
    sygraph::operators::advance::vertex<load_balance_t::workitem_mapped>(G, inFrontier, outFrontier, [=](auto u, auto v) -> bool {
      if (!(visited[v])) {
        visited[v] = true;
        distances[v] = distances[u] + 1;
        return true;
      }
      return false;
    });
    inFrontier.swap_and_clear(outFrontier);
  }

  auto end = std::chrono::high_resolution_clock::now();

  for (size_t i = 0; i < G.get_vertex_count(); i++) {
    std::cout << "Vertex " << i << " has distance " << distances[i] << std::endl;
  }

  std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

  sycl::free(visited, q);
  sycl::free(distances, q);
}