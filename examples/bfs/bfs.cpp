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

template<typename GraphT, typename BFS_T>
bool validate(const GraphT& graph, BFS_T& bfs, uint source) {
  using vertex_t = typename GraphT::vertex_t;
  assert(bfs.get_distance(source) == 0);
  std::vector<uint> distances(graph.get_vertex_count(), graph.get_vertex_count() + 1);
  std::vector<vertex_t> inFrontier;
  std::vector<vertex_t> outFrontier;
  inFrontier.push_back(source);
  distances[source] = 0;

  auto row_offsets = graph.get_row_offsets();
  auto col_indices = graph.get_column_indices();

  size_t iter = 0;
  size_t mismatches = 0;
  while (inFrontier.size()) {
    for (size_t i = 0; i < inFrontier.size(); i++) {
      auto vertex = inFrontier[i];
      
      auto start = row_offsets[vertex];
      auto end = row_offsets[vertex + 1];

      for (size_t j = start; j < end; j++) {
        auto neighbor = col_indices[j];
        if (distances[neighbor] == graph.get_vertex_count() + 1) {
          distances[neighbor] = distances[vertex] + 1;
          if (distances[neighbor] != bfs.get_distance(neighbor)) {
            std::cout << "Distance mismatch at vertex " << neighbor << " expected " << distances[neighbor] << " got " << bfs.get_distance(neighbor) << std::endl;
            mismatches++;
          }
          outFrontier.push_back(neighbor);
        }
      }
    }
    std::swap(inFrontier, outFrontier);
    outFrontier.clear();
    iter++;
  }
  if (mismatches) {
    std::cout << "Mismatches: " << mismatches << std::endl;
  }
  return mismatches == 0;
}

int main(int argc, char** argv) {
  bool print_output = false;
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <path-to-graph> <source>" << std::endl;
    return 1;
  }
  if (argc <= 4) {
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
  
  sygraph::algorithms::BFS bfs {G};
  bfs.init(source);

  auto info = bfs.run<true>();
  
  std::cerr << "[*] Done" << std::endl;

  if (!validate(G, bfs, source)) {
    std::cerr << "Validation failed" << std::endl;
  }

  if (print_output) {
    std::cout << std::left;
    std::cout << std::setw(10) << "Vertex" << std::setw(10) << "Distance" << std::endl;
    for (size_t i = 0; i < G.get_vertex_count(); i++) {
      std::cout << std::setw(10) << i;
      auto distance = bfs.get_distance(i);
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
