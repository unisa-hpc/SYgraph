/*
 * Copyright (c) 2025 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */
#include "../include/utils.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sycl/sycl.hpp>
#include <sygraph/sygraph.hpp>

template<typename GraphT, typename BfsT>
bool validate(const GraphT& graph, BfsT& bfs, uint source) {
  return false;
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
  auto G = sygraph::graph::build::fromCSR<graph_location>(q, csr);
  printGraphInfo(G);
  size_t size = G.getVertexCount();

  sygraph::algorithms::BC bfs{G};
  if (args.random_source) { args.source = getRandomSource(size); }
  bfs.init(args.source);

  std::cout << "[*] Running BC on source " << args.source << std::endl;
  bfs.run();

  std::cerr << "[!] Done" << std::endl;

  if (args.validate) {
    std::cout << "Validation: [";
    auto validation_start = std::chrono::high_resolution_clock::now();
    if (!validate(G, bfs, args.source)) {
      std::cout << failString();
    } else {
      std::cout << successString();
    }
    std::cout << "] | ";
    auto validation_end = std::chrono::high_resolution_clock::now();
    std::cout << "Validation Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(validation_end - validation_start).count() << " ms"
              << std::endl;
  }

  // if (args.print_output) {
  //   std::cout << std::left;
  //   std::cout << std::setw(10) << "Vertex" << std::setw(10) << "Distance" << std::endl;
  //   for (size_t i = 0; i < G.getVertexCount(); i++) {
  //     auto distance = bfs.getDistance(i);
  //     if (distance != size + 1) { std::cout << std::setw(10) << i << std::setw(10) << distance << std::endl; }
  //   }
  // }

#ifdef ENABLE_PROFILING
  sygraph::Profiler::print();
#endif
}
