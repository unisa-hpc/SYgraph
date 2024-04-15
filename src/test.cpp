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

  q.submit([&](sycl::handler &h) {
    auto v = G.getNnzValues();
    h.parallel_for(sycl::range<1>{G.getNumVals()}, [=](sycl::id<1> idx) {
      v[idx] = v[idx] + 100;
    });
  });
  q.wait();

  for (int i = 0; i < G.getNumVals(); i++) {
    std::cout << G.getNnzValues()[i] << std::endl;
  }

  sygraph::frontier::Frontier<size_t> f {q, 20};
  auto n = f.getNumActiveElements();
  std::cout << "Range: " << f.getBitmapRange() << std::endl;
  std::cout << "Num Elements: " << f.getNumElems() << std::endl;
  std::cout << "Size of bitmap: " << f.getSize() << std::endl;
  std::cout << "Active elements: " << n << std::endl;

  f.insert(5);
  
  std::cout << "Next step" << std::endl;
  n = f.getNumActiveElements();
  std::cout << "Active elements: " << n << std::endl;
  
  return 0;  
}