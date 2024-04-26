#include <iostream>
#include <sycl/sycl.hpp>
#include <sygraph/sygraph.hpp>
#include <chrono>
#include <iomanip>

template<typename T>
void PRINT_FRONTIER(T& f, std::string prefix = "") {
  using type_t = typename T::type_t;
  auto size = f.get_bitmap_size() * f.get_bitmap_range();
  std::cout << prefix;
  for (int i = size - 1; i >= 0; --i) {
    std::cout << (f.check(static_cast<type_t>(i)) ? "1" : "0");
  }
  std::cout << " [" << f.get_device_frontier().get_data()[0] << "]" << std::endl; 
  std::cout << std::endl;
}

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

  std::cerr << "[*  ] Loading graph file" << std::endl;
  auto coo = sygraph::io::coo::from_coo<uint, uint, uint>(file);
  std::cerr << "[** ] Converting to CSR" << std::endl;
  auto csr = sygraph::io::csr::from_coo(coo);
  std::cerr << "[***] Building Graph" << std::endl;
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

  size_t* distances = sycl::malloc_shared<size_t>(G.get_vertex_count(), q);
  size_t size = G.get_vertex_count();

  auto device_graph = G.get_device_graph();

  q.parallel_for(sycl::range<1>(G.get_vertex_count()), [distances, size](sycl::id<1> idx) {
    distances[idx] = size + 1;
  }).wait();

  inFrontier.insert(source); // Start from vertex 0
  distances[source] = 0;

  auto start = std::chrono::high_resolution_clock::now();
  int iter = 0;
  while (!inFrontier.empty()) {
    sygraph::operators::advance::vertex<load_balance_t::workitem_mapped>(G, inFrontier, outFrontier, [=](auto src, auto dst, auto edge, auto weight) -> bool {
      return (iter + 1) < distances[dst];
    }).wait();
    sygraph::operators::parallel_for::execute(G, outFrontier, [=](auto v) {
      distances[v] = iter + 1;
    }).wait();

    inFrontier = outFrontier;
    outFrontier.clear();
    iter++;
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::cerr << "[*] Done" << std::endl;

  std::cout << std::left;
  std::cout << std::setw(10) << "Vertex" << std::setw(10) << "Distance" << std::endl;
  for (size_t i = 0; i < G.get_vertex_count(); i++) {
    std::cout << std::setw(10) << i;
    if (distances[i] == size + 1) {
      std::cout << std::setw(10) << "inf";
    } else {
      std::cout << std::setw(10) << distances[i];
    }
    std::cout << std::endl;
  }

  std::cerr << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

  sycl::free(distances, q);
}
