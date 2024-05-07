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

template<typename GraphT, typename BenchT>
bool validate(const GraphT& graph, BenchT& bfs, uint source) {
  // TODO implement
  return true;
}

int main(int argc, char** argv) {
  bool print_output = false;
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <path-to-graph> <source>" << std::endl;
    return 1;
  }
  if (argc >= 4) {
    if (std::string(argv[3]) == "-p") {
      print_output = true;
    }
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
  size_t size = G.get_vertex_count();
  
  sygraph::algorithms::SSSP sssp {G};
  sssp.init(source);

  auto info = sssp.run<true>();
  
  std::cerr << "[*] Done" << std::endl;

  if (!validate(G, sssp, source)) {
    std::cerr << "Validation failed" << std::endl;
  }

  if (print_output) {
    std::cout << std::left;
    std::cout << std::setw(10) << "Vertex" << std::setw(10) << "Distance" << std::endl;
    for (size_t i = 0; i < G.get_vertex_count(); i++) {
      std::cout << std::setw(10) << i;
      auto distance = sssp.get_distance(i);
      if (distance == size + 1) {
        std::cout << std::setw(10) << "inf";
      } else {
        std::cout << std::setw(10) << distance;
      }
      std::cout << std::endl;
    }  
  }

  std::cerr << "Time: " << info.get_duration<std::chrono::milliseconds>() << " ms" << std::endl;
}
