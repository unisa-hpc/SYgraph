#include <sycl/sycl.hpp>
#include <sygraph/sygraph.hpp>

int main() {
  sycl::queue q {sycl::gpu_selector_v};

  auto mat = sygraph::io::storage::matrices::symmetric_6nodes;
  std::istringstream iss(mat.data());
  auto csr = sygraph::io::csr::from_matrix<uint, uint, uint>(iss);
  auto G = sygraph::graph::build::from_csr<sygraph::memory::space::shared>(q, csr);
  
  assert(G.get_vertex_count() == csr.get_row_offsets_size());
  assert(G.get_edge_count() == csr.get_num_nonzeros());
  
  using type_t = decltype(G)::vertex_t;
  sycl::buffer<type_t> nighbours_count_buf(G.get_vertex_count());
  sycl::buffer<type_t> first_neighbors_buf(G.get_vertex_count());

  q.submit([&](sycl::handler &h) {
    auto v = G.get_values();
    auto dG = G.get_device_graph();
    sycl::accessor nighbours_count {nighbours_count_buf, h, sycl::write_only};
    sycl::accessor first_neighbors {first_neighbors_buf, h, sycl::write_only};

    h.parallel_for(sycl::range<1>{G.get_vertex_count()}, [=](sycl::id<1> idx) {
      auto id = idx[0];
      nighbours_count[id] = dG.get_neighbors_count(id);
      first_neighbors[id] = dG.get_first_neighbor_idx(id);
    });
  });
  q.wait();

  sycl::host_accessor nighbours_count {nighbours_count_buf};
  sycl::host_accessor first_neighbors {first_neighbors_buf};

  for (type_t i = 0; i < G.get_vertex_count(); i++) {
    assert(nighbours_count[i] == csr.get_row_offsets()[i + 1] - csr.get_row_offsets()[i]);
    assert(first_neighbors[i] == csr.get_row_offsets()[i]);
  }
}