#include <sycl/sycl.hpp>
#include <sygraph/sygraph.hpp>

int main() {
  sycl::queue q {sycl::gpu_selector_v};
  
  sygraph::formats::CSR<int> csr {2, 2};
  csr.get_values()[0] = 1;

  auto G = sygraph::graph::build::from_csr<sygraph::memory::space::shared>(q, csr);

  for (int i = 0; i < G.get_values_size(); i++) {
    std::cout << G.get_values()[i] << std::endl;
  }

  q.submit([&](sycl::handler &h) {
    auto v = G.get_values();
    h.parallel_for(sycl::range<1>{G.get_values_size()}, [=](sycl::id<1> idx) {
      v[idx] = v[idx] + 100;
    });
  });
  q.wait();

  for (int i = 0; i < G.get_values_size(); i++) {
    std::cout << G.get_values()[i] << std::endl;
  }
}