#include <iostream>
#include <sycl/sycl.hpp>
#include <sygraph/sygraph.hpp>
#include <chrono>
#include <iomanip>
#include "../include/utils.hpp"

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
  args_t args {argc, argv};

  std::cerr << "[*  ] Reading CSR" << std::endl;
  auto csr = read_csr<uint, uint, uint>(args);

  sycl::queue q {sycl::gpu_selector_v};
  
  std::cerr << "[** ] Building Graph" << std::endl;
  auto G = sygraph::graph::build::from_csr<sygraph::memory::space::shared>(q, csr);
  print_graph_info(G);
  size_t size = G.get_vertex_count();
  
  sygraph::algorithms::BFS bfs {G};
  if (args.random_source) {
    args.source = get_random_source(size);
  }
  bfs.init(args.source);

  std::cerr << "[***] Running BFS on source " << args.source << std::endl;
  auto info = bfs.run<true>();
  
  std::cerr << "[!] Done" << std::endl;

  if (args.validate) {
    std::cerr << "Validation: [";
    auto validation_start = std::chrono::high_resolution_clock::now();
    if (!validate(G, bfs, args.source)) {
      std::cerr << "\033[1;31mFailed\033[0m";
    } else {
      std::cerr << "\033[1;32mSuccess\033[0m";
    }
    std::cerr << "] | ";
    auto validation_end = std::chrono::high_resolution_clock::now();
    std::cerr << "Validation Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(validation_end - validation_start).count() << " ms" << std::endl;
  }

  if (args.print_output) {
    std::cout << std::left;
    std::cout << std::setw(10) << "Vertex" << std::setw(10) << "Distance" << std::endl;
    for (size_t i = 0; i < G.get_vertex_count(); i++) {
      auto distance = bfs.get_distance(i);
      if (distance == size + 1) {
        continue;
      } else {
        std::cout << std::setw(10) << i << std::setw(10) << distance << std::endl;
      }
    }  
  }

  std::cerr << "Time: " << info.get_duration<std::chrono::milliseconds>() << " ms" << std::endl;
}
