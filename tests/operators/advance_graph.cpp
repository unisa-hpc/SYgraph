#include <chrono>
#include <sycl/sycl.hpp>
#include <sygraph/sygraph.hpp>

int main() {
  sycl::queue q{sycl::gpu_selector_v};

  auto mat = sygraph::io::storage::matrices::two_cc;
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

  auto out_frontier = sygraph::frontier::makeFrontier<frontier_view_t::vertex, frontier_impl_t::mlb>(q, G);

  bool* visited = sycl::malloc_shared<bool>(G.getVertexCount(), q);
  q.fill(visited, visited + G.getVertexCount(), false).wait();

  auto device_graph = G.getDeviceGraph();

  auto start = std::chrono::high_resolution_clock::now();

  sygraph::operators::advance::vertices<load_balance_t::workgroup_mapped, sygraph::frontier::frontier_view::vertex>(
      G, out_frontier, [=](auto u, auto v, auto e, auto w) -> bool { return device_graph.getDegree(u) != 0; });

  sygraph::operators::compute::execute(G, out_frontier, [=](auto v) { visited[v] = true; });

  for (int i = 0; i < G.getVertexCount(); i++) {}

  auto end = std::chrono::high_resolution_clock::now();

  for (size_t i = 0; i < G.getVertexCount(); i++) { std::cout << "Vertex " << i << ": " << (visited[i] ? "Yes" : "No") << std::endl; }

  std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

  sycl::free(visited, q);
}