#include <chrono>
#include <sycl/sycl.hpp>
#include <sygraph/sygraph.hpp>

int main() {
  sycl::queue q{sycl::gpu_selector_v};

  auto mat = sygraph::io::storage::matrices::symmetric_6nodes;
  std::istringstream iss(mat.data());
  auto csr = sygraph::io::csr::fromMatrix<uint, uint, uint>(iss);
  auto G = sygraph::graph::build::fromCSR<sygraph::memory::space::shared>(q, csr);

  using type_t = decltype(G)::vertex_t;
  sycl::buffer<type_t> nighbours_count_buf(G.getVertexCount());
  sycl::buffer<type_t> first_neighbors_buf(G.getVertexCount());


  using load_balance_t = sygraph::operators::load_balancer;
  using direction_t = sygraph::operators::direction;
  using frontier_view_t = sygraph::frontier::frontier_view;
  using frontier_impl_t = sygraph::frontier::frontier_type;

  auto in_frontier = sygraph::frontier::makeFrontier<frontier_view_t::vertex, frontier_impl_t::bitmap>(q, G);
  auto out_frontier = sygraph::frontier::makeFrontier<frontier_view_t::vertex, frontier_impl_t::bitmap>(q, G);

  bool* visited = sycl::malloc_shared<bool>(G.getVertexCount(), q);
  size_t* distances = sycl::malloc_shared<size_t>(G.getVertexCount(), q);

  auto device_graph = G.getDeviceGraph();

  in_frontier.insert(0); // Start from vertex 0
  distances[0] = 0;
  visited[0] = true;

  auto start = std::chrono::high_resolution_clock::now();

  // while (!inFrontier.empty()) { TODO: Implement this
  // sygraph::operators::advance::vertex<load_balance_t::workitem_mapped>(G, inFrontier, outFrontier, [=](auto u, auto v, auto e, auto w) -> bool {
  //   if (!(visited[v])) {
  //     visited[v] = true;
  //     distances[v] = distances[u] + 1;
  //     return true;
  //   }
  //   return false;
  // });
  // inFrontier.swap_and_clear(outFrontier);
  // }

  auto end = std::chrono::high_resolution_clock::now();

  for (size_t i = 0; i < G.getVertexCount(); i++) { std::cout << "Vertex " << i << " has distance " << distances[i] << std::endl; }

  std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

  sycl::free(visited, q);
  sycl::free(distances, q);
}