#include <string>
#include <iostream>
#include <random>

#include <sygraph/sygraph.hpp>


template <typename index_t>
struct args_t {
  bool print_output = false;
  bool validate = false;
  bool binary_format = false;
  bool random_source = true;
  bool undirected = false;
  std::string path;
  index_t source;

  void print_usage() {
    std::cerr << "Usage: " << path << " [-b] <path-to-graph> [-p] [-v] [-u] [-s <source>]" << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  -h: show this message" << std::endl;
    std::cerr << "  -b: binary format" << std::endl;
    std::cerr << "  -p: print output" << std::endl;
    std::cerr << "  -v: validate output" << std::endl;
    std::cerr << "  -u: undirected graph" << std::endl;
    std::cerr << "  -s <source>: source vertex" << std::endl;
  }

  args_t(int argc, char** argv) {
    if (argc < 2) {
      print_usage();
      exit(1);
    } else {
      if (std::string(argv[1]) == "-b") {
        binary_format = true;
        path = argv[2];
      } else if (std::string(argv[1]) == "-h") {
        print_usage();
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
        print_usage();
        exit(0);
      } else {
        continue;
      }
    }
  }
};

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

uint get_random_source(size_t size) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, size - 1);
  return dis(gen);
}

template<typename value_t, typename index_t, typename offset_t>
sygraph::formats::CSR<value_t, index_t, offset_t> read_csr(const args_t<index_t>& args) {
  sygraph::formats::CSR<value_t, index_t, offset_t> csr;
  if (args.binary_format) {
    std::ifstream file(args.path, std::ios::binary);
    if (!file.is_open()) {
      std::cerr << "Error: could not open file " << args.path << std::endl;
      exit(1);
    }
    csr = sygraph::io::csr::from_binary<value_t, index_t, offset_t>(file);
  } else {
    std::ifstream file(args.path);
    if (!file.is_open()) {
      std::cerr << "Error: could not open file " << args.path << std::endl;
      exit(1);
    }
    auto coo = sygraph::io::coo::from_coo<value_t, index_t, offset_t>(file, args.undirected);
    csr = sygraph::io::csr::from_coo(coo);
  }
  
  std::ifstream file(args.path);
  if (!file.is_open()) {
    std::cerr << "Error: could not open file " << args.path << std::endl;
    exit(1);
  }
  return csr;
}

template<typename GraphT>
void print_graph_info(const GraphT& g) {
  std::cerr << "-----------------------------------" << std::endl;
  std::cerr << std::left;
  std::cerr << std::setw(17) << "Vertex count:" << std::setw(10) << g.get_vertex_count()<< std::endl;
  std::cerr << std::setw(17) << "Edge count:" << std::setw(10) << g.get_edge_count() << std::endl;
  std::cerr << std::setw(17) << "Average degree:" << std::setw(10) << g.get_edge_count() / g.get_vertex_count() << std::endl;
  std::cerr << "-----------------------------------" << std::endl;
}

void print_device_info(sycl::queue& queue, std::string prefix = "") {
  std::string device_name = queue.get_device().get_info<sycl::info::device::name>();
  std::string device_backend = queue.get_device().get_platform().get_info<sycl::info::platform::name>();
  std::cerr << prefix << "Running on: " << "[" << device_backend << "] " << device_name << std::endl;
}