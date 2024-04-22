#include <sycl/sycl.hpp>
#include <sygraph/sygraph.hpp>
#include <iostream>

int main() {
  sycl::queue q {sycl::gpu_selector_v};

  std::string mat = "8\n"
    "1 2 2\n"
    "1 0 3\n"
    "0 2 1\n"
    "2 0 1\n"
    "1 3 2\n"
    "2 1 2\n"
    "3 1 2\n"
    "0 1 1";
  std::istringstream iss(mat.data());
  auto coo = sygraph::io::coo::from_coo<uint, uint, uint>(iss);

  auto csr = sygraph::io::csr::from_coo(coo);
  auto row_offsets = csr.get_row_offsets();
  auto col_indices = csr.get_column_indices();
  auto values = csr.get_values();

  assert(row_offsets.size() - 1 == 4);
  assert(row_offsets[0] == 0);
  assert(row_offsets[1] == 2);
  assert(row_offsets[2] == 5);
  assert(row_offsets[3] == 7);
  assert(row_offsets[4] == 8);
  assert(col_indices.size() == 8);
  assert(values.size() == 8);

  std::cout << "             ";
  for (int i = 0; i < row_offsets.size(); ++i) {
    std::cout << i << " ";
  }
  std::cout << std::endl << "row_offsets: ";
  for (int i = 0; i < row_offsets.size(); ++i) {
    std::cout << row_offsets[i] << " ";
  }
  std::cout << std::endl;

  std::cout << "col_indices: ";
  for (int i = 0; i < col_indices.size(); ++i) {
    std::cout << col_indices[i] << " ";
  }
  std::cout << std::endl << "      values:";
  for (int i = 0; i < values.size(); ++i) {
    std::cout << values[i] << " ";
  }
  std::cout << std::endl;
}