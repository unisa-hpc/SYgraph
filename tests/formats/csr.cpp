#include <sycl/sycl.hpp>
#include <sygraph/sygraph.hpp>

int main() {
  sycl::queue q{sycl::gpu_selector_v};

  auto mat = sygraph::io::storage::matrices::symmetric_6nodes;
  std::istringstream iss(mat.data());
  auto csr = sygraph::io::csr::fromMatrix<uint, uint, uint>(iss);
  auto G = sygraph::graph::build::fromCSR<sygraph::memory::space::shared>(q, csr);

  assert(G.getVertexCount() == csr.getRowOffsets().size() - 1);
  assert(G.getEdgeCount() == csr.getNumNonzeros());

  auto frontier = sygraph::frontier::makeFrontier<sygraph::frontier::frontier_view::vertex, sygraph::frontier::frontier_type::bitmap>(q, G);

  q.submit([&](sycl::handler& h) {
    auto v = G.getValues();
    auto frontier_d = frontier.getDeviceFrontier();
    h.parallel_for(sycl::range<1>{G.getVertexCount()}, [=](sycl::id<1> idx) { v[idx] = frontier_d.getBitmapRange(); });
  });
  q.wait();

  for (int i = 0; i < G.getValuesSize(); i++) { std::cout << G.getValues()[i] << std::endl; }
}