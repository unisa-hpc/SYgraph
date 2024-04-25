#include <iostream>
#include <sycl/sycl.hpp>
#include <sygraph/sygraph.hpp>
#include <chrono>

int main(int argc, char** argv) {

  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <path-to-graph> <source>" << std::endl;
    return 1;
  }
  std::string path = argv[1];
  uint source = std::stoi(argv[2]);

  std::ifstream file(path);
  if (!file.is_open()) {
    std::cerr << "Error: could not open file " << path << std::endl;
    return 1;
  }

  sycl::queue q {sycl::gpu_selector_v};

  std::cout << "[*  ] Loading graph file" << std::endl;
  auto coo = sygraph::io::coo::from_coo<uint, uint, uint>(file);
  std::cout << "[** ] Converting to CSR" << std::endl;
  auto csr = sygraph::io::csr::from_coo(coo);
  std::cout << "[***] Building Graph" << std::endl;
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
  uint* parents = sycl::malloc_shared<uint>(G.get_vertex_count(), q);

  auto device_graph = G.get_device_graph();

  inFrontier.insert(source); // Start from vertex 0
  distances[source] = 0; 
  parents[source] = source; 
  visited[source] = true;

  auto start = std::chrono::high_resolution_clock::now();
  int iter = 0;
  while (!inFrontier.empty()) {
    // std::cerr << "Iteration: " << (iter++) << std::endl;
    sygraph::operators::advance::vertex<load_balance_t::workitem_mapped>(G, inFrontier, outFrontier, [=](auto src, auto dst) -> bool {
      if (!(visited[dst])) {
        visited[dst] = true;
        distances[dst] = distances[src] + 1;
        parents[dst] = src;
        return true;
      }
      return false;
    });
    // std::cerr << "GPU Computation..." << std::endl;
    outFrontier.swap_and_clear(inFrontier);
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::cerr << "[*] Done" << std::endl;

  for (size_t i = 0; i < G.get_vertex_count(); i++) {
    if (i == source) {
      std::cout << "[" << i << "] SOURCE" << std::endl;
    } else {
      std::cout << "[" << i << "] D: " << distances[i] << " | P: " << parents[i] << std::endl;
    }
  }

  std::cerr << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

  sycl::free(visited, q);
  sycl::free(distances, q);
}
