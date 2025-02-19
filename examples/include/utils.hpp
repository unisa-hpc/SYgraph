/*
 * Copyright (c) 2025 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <unistd.h>

#include <sygraph/sygraph.hpp>

// Map the numeric macro to the actual object
#if GRAPH_LOCATION == 0
constexpr sygraph::memory::space graph_location = sygraph::memory::space::host;
#elif GRAPH_LOCATION == 1
constexpr sygraph::memory::space graph_location = sygraph::memory::space::device;
#elif GRAPH_LOCATION == 2
constexpr sygraph::memory::space graph_location = sygraph::memory::space::shared;
#else
#error "Invalid GRAPH_LOCATION value. Must be 0 (host), 1 (device), or 2 (shared)."
#endif

template<typename IndexT>
struct ArgsT {
  bool print_output = false;
  bool validate = false;
  bool binary_format = false;
  bool matrix_market = false;
  bool random_source = true;
  bool undirected = false;
  std::string path;
  IndexT source;

  void printUsage() {
    std::cerr << "Usage: " << path << " [-b] <path-to-graph> [-p] [-v] [-u] [-s <source>]" << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  -h: show this message" << std::endl;
    std::cerr << "  -b: binary format" << std::endl;
    std::cerr << "  -p: print output" << std::endl;
    std::cerr << "  -v: validate output" << std::endl;
    std::cerr << "  -u: undirected graph [only for non-binary format]" << std::endl;
    std::cerr << "  -s <source>: source vertex" << std::endl;
  }

  ArgsT(int argc, char** argv) {
    if (argc < 2) {
      printUsage();
      exit(1);
    } else {
      if (std::string(argv[1]) == "-b") {
        binary_format = true;
        path = argv[2];
      } else if (std::string(argv[1]) == "-m") {
        matrix_market = true;
        path = argv[2];
      } else if (std::string(argv[1]) == "-h") {
        printUsage();
        exit(0);
      } else {
        path = argv[1];
      }
    }
    for (int i = 2; i < argc; i++) {
      if (std::string(argv[i]) == "-p") {
        print_output = true;
      } else if (std::string(argv[i]) == "-v") {
        validate = true;
      } else if (std::string(argv[i]) == "-u") {
        undirected = true;
      } else if (std::string(argv[i]) == "-s") {
        random_source = false;
        try {
          source = std::stoi(argv[i + 1]);
        } catch (std::exception& e) {
          std::cerr << "Error: -s flag requires an integer argument" << std::endl;
          exit(1);
        }
      } else if (std::string(argv[i]) == "-h") {
        printUsage();
        exit(0);
      } else {
        continue;
      }
    }
  }
};

template<typename T>
void printFrontier(T& f, std::string prefix = "") {
  using type_t = typename T::type_t;
  auto size = f.getBitmapSize() * f.getBitmapRange();
  std::cout << prefix;
  for (int i = size - 1; i >= 0; --i) { std::cout << (f.check(static_cast<type_t>(i)) ? "1" : "0"); }
  std::cout << " [" << f.getDeviceFrontier().get_data()[0] << "]" << std::endl;
  std::cout << std::endl;
}

uint getRandomSource(size_t size) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, size - 1);
  return dis(gen);
}

template<typename ValueT, typename IndexT, typename OffsetT>
sygraph::formats::CSR<ValueT, IndexT, OffsetT> readCSR(const ArgsT<IndexT>& args) {
  sygraph::formats::CSR<ValueT, IndexT, OffsetT> csr;
  if (args.binary_format) {
    std::ifstream file(args.path, std::ios::binary);
    if (!file.is_open()) {
      std::cerr << "Error: could not open file " << args.path << std::endl;
      exit(1);
    }
    csr = sygraph::io::csr::fromBinary<ValueT, IndexT, OffsetT>(file);
  } else if (args.matrix_market) {
    csr = sygraph::io::csr::fromMM<ValueT, IndexT, OffsetT>(args.path);
  } else {
    std::ifstream file(args.path);
    if (!file.is_open()) {
      std::cerr << "Error: could not open file " << args.path << std::endl;
      exit(1);
    }
    auto coo = sygraph::io::coo::fromCOO<ValueT, IndexT, OffsetT>(file, args.undirected);
    csr = sygraph::io::csr::fromCOO(coo);
  }

  std::ifstream file(args.path);
  if (!file.is_open()) {
    std::cerr << "Error: could not open file " << args.path << std::endl;
    exit(1);
  }
  return csr;
}

template<typename GraphT>
void printGraphInfo(const GraphT& g) {
  std::cerr << "-----------------------------------" << std::endl;
  std::cerr << std::left;
  std::cerr << std::setw(17) << "Vertex count:" << std::setw(10) << g.getVertexCount() << std::endl;
  std::cerr << std::setw(17) << "Edge count:" << std::setw(10) << g.getEdgeCount() << std::endl;
  std::cerr << std::setw(17) << "Average degree:" << std::setw(10) << g.getEdgeCount() / g.getVertexCount() << std::endl;
  std::cerr << "-----------------------------------" << std::endl;
}

void printDeviceInfo(sycl::queue& queue, std::string prefix = "") {
  std::string device_name = queue.get_device().get_info<sycl::info::device::name>();
  std::string device_backend = queue.get_device().get_platform().get_info<sycl::info::platform::name>();
  std::cerr << prefix << "Running on: " << "[" << device_backend << "] " << device_name << std::endl;
}

bool isConsoleOutput() { return static_cast<int>(static_cast<int>(isatty(STDOUT_FILENO) != 0)) != 0; }

std::string successString() {
  if (!isConsoleOutput()) { return "Success"; }
  return "\033[1;32mSuccess\033[0m";
}

std::string failString() {
  if (!isConsoleOutput()) { return "Failed"; }
  return "\033[1;31mFailed\033[0m";
}