#include <iostream>
#include <sycl/sycl.hpp>
#include <sygraph/sygraph.hpp>
#include <chrono>
#include <iomanip>
#include <random>
#include "../include/utils.hpp"

template<typename GraphT, typename BenchT>
bool validate(const GraphT& graph, BenchT& bfs, uint source) {
  // TODO implement
  return true;
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
  
  sygraph::algorithms::SSSP sssp {G};
  if (args.random_source) {
    args.source = get_random_source(size);
  }
  sssp.init(args.source);

  std::cerr << "[***] Running SSSP on source " << args.source << std::endl;
  auto info = sssp.run<true>();
  
  std::cerr << "[!] Done" << std::endl;

  if (args.validate) {
    std::cerr << "Validation: [";
    auto validation_start = std::chrono::high_resolution_clock::now();
    if (!validate(G, sssp, args.source)) {
      std::cerr << "\033[1;32mFailed\033[0m";
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
      auto distance = sssp.get_distance(i);
      if (distance == size + 1) {
        continue;
      } else {
        std::cout << std::setw(10) << i << std::setw(10) << distance << std::endl;
      }
    }  
  }

  std::cerr << "Time: " << info.get_duration<std::chrono::milliseconds>() << " ms" << std::endl;
}
