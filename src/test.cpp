#include <sycl/sycl.hpp>
#include <sygraph/sygraph.hpp>

int main() {
  sycl::queue q {sycl::gpu_selector_v};
  
  sygraph::formats::CSR<int> csr {10, 10};

  auto G = sygraph::graph::build::from_csr<sygraph::memory::space::device>(q, csr);
  
}