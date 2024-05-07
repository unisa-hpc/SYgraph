#include <iostream>
#include <fstream>

#include <sygraph/sygraph.hpp>

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <COO file format> <out>" << std::endl;
    return 1;
  }

  std::ifstream in_file(argv[1]);
  if (!in_file.is_open()) {
    std::cerr << "Error: could not open file " << argv[1] << std::endl;
    return 1;
  }

  sycl::queue q {sycl::gpu_selector_v};

  std::cerr << "[* ] Loading COO file" << std::endl;
  auto coo = sygraph::io::coo::from_coo<uint, uint, uint>(in_file);
  std::cerr << "[**] Converting to CSR" << std::endl;
  auto csr = sygraph::io::csr::from_coo(coo);

  std::ofstream out_file(argv[2], std::ios::out | std::ios::binary);
  if (!out_file.is_open()) {
    std::cerr << "Error: could not open file " << argv[2] << std::endl;
    return 1;
  }
  sygraph::io::csr::to_binary(csr, out_file);

  std::cout << "[!] Done" << std::endl;
}