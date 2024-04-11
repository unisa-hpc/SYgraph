#include <sycl/sycl.hpp>
#include <sygraph/sygraph.hpp>

int main() {
  sycl::queue q {sycl::gpu_selector_v};
  
  sygraph::formats::CSR<int> csr {2, 2};
  csr.getNnzValues()[0] = 1;
  csr.getNnzValues()[1] = 2;
  csr.getColumnIndices()[0] = 0;
  csr.getColumnIndices()[1] = 1;

  auto G = sygraph::graph::build::from_csr<sygraph::memory::space::shared>(q, csr);

  for (int i = 0; i < G.getNumVals(); i++) {
    std::cout << G.getNnzValues()[i] << std::endl;
  }
  
  return 0;  
}