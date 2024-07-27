#include "../include/utils.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sycl/sycl.hpp>
#include <sygraph/sygraph.hpp>

template<typename GraphT, typename BfsT>
bool validate(const GraphT& graph, BfsT& bfs, uint source) {
  throw std::runtime_error("Not implemented");
}

int main(int argc, char** argv) {
  using type_t = unsigned int;
  ArgsT<type_t> args{argc, argv};

  std::cerr << "[*] Reading CSR" << std::endl;
  auto csr = readCSR<type_t, type_t, type_t>(args);

#ifdef ENABLE_PROFILING
  sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::enable_profiling()};
#else
  sycl::queue q{sycl::gpu_selector_v};
#endif

  printDeviceInfo(q, "[*] ");

  std::cerr << "[*] Building Graph" << std::endl;
  auto G = sygraph::graph::build::fromCSR<sygraph::memory::space::device>(q, csr);
  printGraphInfo(G);
  size_t size = G.getVertexCount();

  sygraph::algorithms::TC tc{G};
  tc.init();

  std::cout << "[*] Running TC" << std::endl;
  tc.run<true>();

  std::cerr << "[!] Done" << std::endl;

  if (args.validate) {
    std::cout << "Validation: [";
    auto validation_start = std::chrono::high_resolution_clock::now();
    if (!validate(G, tc, args.source)) {
      std::cout << failString();
    } else {
      std::cout << successString();
    }
    std::cout << "] | ";
    auto validation_end = std::chrono::high_resolution_clock::now();
    std::cout << "Validation Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(validation_end - validation_start).count() << " ms"
              << std::endl;
  }

  if (args.print_output) { /* TODO implement*/
    std::cout << "Total num triangles: " << tc.getNumTriangles() << std::endl;
  }

#ifdef ENABLE_PROFILING
  sygraph::Profiler::print();
#endif
}
