#include "../include/utils.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <queue>
#include <random>
#include <sycl/sycl.hpp>
#include <sygraph/sygraph.hpp>
#include <utility>
#include <vector>

template<typename VertexT, typename WeightT>
class Prioritize {
public:
  bool operator()(std::pair<VertexT, WeightT>& p1, std::pair<VertexT, WeightT>& p2) { return p1.second > p2.second; }
};

template<typename GraphT, typename BenchT>
bool validate(const GraphT& graph, BenchT& sssp, uint source) {
  using vertex_t = typename GraphT::vertex_t;
  using weight_t = typename GraphT::weight_t;
  using edge_t = typename GraphT::edge_t;
  auto* row_offsets = graph.getRowOffsets();
  auto* column_indices = graph.getColumnIndices();
  auto* nonzero_values = graph.getValues();

  std::vector<weight_t> distances(graph.getVertexCount(), graph.getVertexCount() + 1);
  distances[source] = 0;

  std::priority_queue<std::pair<vertex_t, weight_t>, std::vector<std::pair<vertex_t, weight_t>>, Prioritize<vertex_t, weight_t>> pq;
  pq.push(std::make_pair(source, 0.0));

  while (!pq.empty()) {
    std::pair<vertex_t, weight_t> curr = pq.top();
    pq.pop();

    vertex_t curr_node = curr.first;
    weight_t curr_dist = curr.second;

    vertex_t start = row_offsets[curr_node];
    vertex_t end = row_offsets[curr_node + 1];

    for (vertex_t offset = start; offset < end; offset++) {
      vertex_t neib = column_indices[offset];
      weight_t new_dist = curr_dist + nonzero_values[offset];
      if (new_dist < distances[neib]) {
        distances[neib] = new_dist;
        pq.push(std::make_pair(neib, new_dist));
      }
    }
  }

  for (auto i = 0; i < graph.getVertexCount(); i++) {
    if (distances[i] != sssp.getDistance(i)) {
      std::cerr << "Mismatch at vertex " << i << " | Expected: " << distances[i] << " | Got: " << sssp.getDistance(i) << std::endl;
      return false;
    }
  }

  return true;
}

int main(int argc, char** argv) {
  using type_t = unsigned int;
  ArgsT<type_t> args{argc, argv};

  std::cerr << "[*  ] Reading CSR" << std::endl;
  auto csr = readCSR<float, type_t, type_t>(args);

#ifdef ENABLE_PROFILING
  sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::enable_profiling()};
#else
  sycl::queue q{sycl::gpu_selector_v};
#endif

  std::cerr << "[** ] Building Graph" << std::endl;
  auto G = sygraph::graph::build::fromCSR<sygraph::memory::space::shared>(q, csr);
  printGraphInfo(G);
  size_t size = G.getVertexCount();

  sygraph::algorithms::SSSP sssp{G};
  if (args.random_source) { args.source = getRandomSource(size); }
  sssp.init(args.source);

  std::cerr << "[***] Running SSSP on source " << args.source << std::endl;
  sssp.run<true>();

  std::cerr << "[!] Done" << std::endl;

  if (args.validate) {
    std::cerr << "Validation: [";
    auto validation_start = std::chrono::high_resolution_clock::now();
    if (!validate(G, sssp, args.source)) {
      std::cerr << "\033[1;32mFailed\033[0m";
    } else {
      std::cerr << "\033[1;32mSuccess\033[0m";
    }
    std::cerr << "] | ";
    auto validation_end = std::chrono::high_resolution_clock::now();
    std::cerr << "Validation Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(validation_end - validation_start).count() << " ms"
              << std::endl;
  }

  if (args.print_output) {
    std::cout << std::left;
    std::cout << std::setw(10) << "Vertex" << std::setw(10) << "Distance" << std::endl;
    for (size_t i = 0; i < G.getVertexCount(); i++) {
      auto distance = sssp.getDistance(i);
      if (distance != size + 1) { std::cout << std::setw(10) << i << std::setw(10) << distance << std::endl; }
    }
  }

#ifdef ENABLE_PROFILING
  sygraph::Profiler::print();
#endif
}
