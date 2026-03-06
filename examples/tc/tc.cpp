#include "../include/utils.hpp"
#include <CLI/CLI.hpp>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sycl/sycl.hpp>
#include <sygraph/sygraph.hpp>
#include <synergy.hpp>

template<typename GraphT, typename BfsT>
bool validate(const GraphT& graph, BfsT& bfs, uint source) {
  throw std::runtime_error("Not implemented");
}

int main(int argc, char** argv) {
  using type_t = unsigned int;
  GraphOptions opts;
  CLI::App app{"SYgraph example"};
  auto source_option = configureBaseCLI(app, opts);
  CLI11_PARSE(app, argc, argv);
  finalizeGraphOptions(opts, source_option);

  std::cerr << "[*] Reading CSR" << std::endl;
  sygraph::graph::Properties properties;
  auto csr = readCSR<type_t, type_t, type_t>(opts, &properties);

#ifdef ENABLE_PROFILING
  synergy::queue q{sycl::gpu_selector_v, sycl::property::queue::enable_profiling()};
#else
  synergy::queue q{sycl::gpu_selector_v};
#endif

  printDeviceInfo(q, "[*] ");

  std::cerr << "[*] Building Graph" << std::endl;
  auto G = sygraph::graph::build::fromCSR<sygraph::memory::space::shared>(q, csr, properties);
  printGraphInfo(G);
  size_t size = G.getVertexCount();

  sygraph::algorithms::TC tc{G};
  tc.init();

  std::cout << "[*] Running TC" << std::endl;
  auto start_energy_j = q.device_energy_consumption();
  auto start_timer = std::chrono::high_resolution_clock::now();
  tc.run<true>();
  auto end_timer = std::chrono::high_resolution_clock::now();
  auto end_energy_j = q.device_energy_consumption();


  std::cerr << "[!] Done" << std::endl;

  if (opts.validate) {
    std::cout << "Validation: [";
    auto validation_start = std::chrono::high_resolution_clock::now();
    if (!validate(G, tc, opts.source)) {
      std::cout << failString();
    } else {
      std::cout << successString();
    }
    std::cout << "] | ";
    auto validation_end = std::chrono::high_resolution_clock::now();
    std::cout << "Validation Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(validation_end - validation_start).count() << " ms"
              << std::endl;
  }

  if (opts.print_output) { /* TODO implement*/
    std::cout << "Total num triangles: " << tc.getNumTriangles() << std::endl;
  }
  
  #ifdef ENABLE_PROFILING
  sygraph::Profiler::print();
  #endif
  std::cout << "Total Host Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_timer - start_timer).count() << " ms" << std::endl;
  std::cout << "Total Energy Consumption: " << end_energy_j - start_energy_j << " J" << std::endl;


}
