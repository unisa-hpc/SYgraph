/*
 * Copyright (c) 2025 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */
#include <fstream>
#include <iostream>

#include <sygraph/sygraph.hpp>

template<typename ValueT, typename IndexT, typename OffsetT>
void writeToFile(const std::string& filename, const sygraph::formats::CSR<ValueT, IndexT, OffsetT>& csr) {
  std::cout << "[**] Writing to file" << std::endl;
  std::ofstream out_file(filename, std::ios::binary);
  if (!out_file.is_open()) {
    std::cerr << "Error: could not open file " << filename << std::endl;
    exit(1);
  }
  sygraph::io::csr::toBinary(csr, out_file);
}

int main(int argc, char** argv) {
  bool undirected = false;
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <MatrixMarket file format> <out>" << std::endl;
    return 1;
  }

  std::ifstream in_file(argv[1]);
  if (!in_file.is_open()) {
    std::cerr << "Error: could not open file " << argv[1] << std::endl;
    return 1;
  }

  // get first line of the file
  std::string line;
  std::getline(in_file, line);
  sygraph::io::detail::mm::Banner banner;
  banner.read(line);
  in_file.seekg(0, std::ios::beg);

  std::cerr << "[* ] Reading MatrixMarket file" << std::endl;
  if (banner.isInteger()) {
    auto csr = sygraph::io::csr::fromMM<uint, uint, uint>(in_file);
    writeToFile(argv[2], csr);
  } else if (banner.isReal() || banner.isPattern()) {
    auto csr = sygraph::io::csr::fromMM<float, uint, uint>(in_file);
    writeToFile(argv[2], csr);
  } else {
    std::cerr << "Error: unsupported field type" << std::endl;
    return 1;
  }

  std::cout << "[!] Done" << std::endl;
}