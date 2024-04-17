#include <sycl/sycl.hpp>
#include <sygraph/sygraph.hpp>

int main() {
  sycl::queue q {sycl::gpu_selector_v};

  auto mat = sygraph::io::storage::matrices::symmetric_small;
  std::istringstream iss(mat.data());
  auto csr = sygraph::io::csr::from_matrix<uint, uint, uint>(iss);
  auto G = sygraph::graph::build::from_csr<sygraph::memory::space::shared>(q, csr);
  
  assert(G.get_vertex_count() == csr.get_row_offsets_size() - 1);
  assert(G.get_edge_count() == csr.get_num_nonzeros());

  auto frontier = sygraph::frontier::make_frontier<sygraph::frontier::FrontierView::vertex, sygraph::frontier::FrontierType::bitmap>(q, G);

  q.submit([&](sycl::handler &h) {
    auto v = G.get_values();
    auto frontier_d = frontier.get_device_frontier();
    h.parallel_for(sycl::range<1>{G.get_vertex_count()}, [=](sycl::id<1> idx) {
      v[idx] = frontier_d.get_bitmap_range();
    });
  });
  q.wait();

  for (int i = 0; i < G.get_values_size(); i++) {
    std::cout << G.get_values()[i] << std::endl;
  }
}