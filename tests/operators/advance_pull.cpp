#include "test_utils.hpp"
#include <array>
#include <type_traits>
#include <sycl/sycl.hpp>
#include <sygraph/sygraph.hpp>

template<sygraph::operators::load_balancer LoadBalancer, sygraph::frontier::frontier_type FrontierType, typename GraphT>
uint32_t run_pull_case(GraphT& G, bool* active_vertices) {
  using direction_t = sygraph::operators::direction;
  using frontier_view_t = sygraph::frontier::frontier_view;

  auto& q = G.getQueue();
  auto in_frontier = sygraph::frontier::makeFrontier<frontier_view_t::vertex, FrontierType>(q, G);
  auto out_frontier = sygraph::frontier::makeFrontier<frontier_view_t::vertex, FrontierType>(q, G);
  auto visits = sygraph::memory::detail::memoryAlloc<uint32_t, sygraph::memory::space::shared>(G.getVertexCount(), q);

  for (size_t i = 0; i < G.getVertexCount(); ++i) {
    visits[i] = 0;
    active_vertices[i] = false;
  }

  in_frontier.insert(0);
  in_frontier.insert(1);

  auto e = sygraph::operators::advance::frontier<direction_t::pull, LoadBalancer, frontier_view_t::vertex, frontier_view_t::vertex>(
      G,
      in_frontier,
      out_frontier,
      [=](auto src, auto, auto, auto) -> bool {
        sygraph::sync::atomicFetchAdd(visits + src, 1U);
        return true;
      },
      sygraph::frontier::size::infer_from_device);
  e.waitAndThrow();

  auto mark_e = sygraph::operators::compute::execute<frontier_view_t::vertex>(G, out_frontier, [=](auto v) { active_vertices[v] = true; });
  mark_e.waitAndThrow();

  const uint32_t vertex_two_visits = visits[2];
  sygraph::memory::detail::releaseUSM(visits, q);
  return vertex_two_visits;
}

int main() {
  auto q = sygraph::tests::makeQueue();

  auto mat = sygraph::io::storage::matrices::symmetric_6nodes;
  std::istringstream iss(mat.data());
  auto csr = sygraph::io::csr::fromMatrix<uint, uint, uint>(iss);
  auto G = sygraph::graph::build::fromCSR<sygraph::memory::space::shared>(q, csr);

  auto run_frontier_case = [&](auto frontier_tag) {
    constexpr auto frontier_type = decltype(frontier_tag)::value;
    auto workgroup_default_active = sygraph::memory::detail::memoryAlloc<bool, sygraph::memory::space::shared>(G.getVertexCount(), q);
    auto bucketing_active = sygraph::memory::detail::memoryAlloc<bool, sygraph::memory::space::shared>(G.getVertexCount(), q);

    const auto workgroup_visits =
        run_pull_case<sygraph::operators::load_balancer::workgroup_mapped, frontier_type>(G, workgroup_default_active);
    const auto bucketing_visits = run_pull_case<sygraph::operators::load_balancer::bucketing, frontier_type>(G, bucketing_active);

    for (size_t i = 0; i < G.getVertexCount(); ++i) {
      assert(workgroup_default_active[i] == bucketing_active[i]);
    }

    assert(workgroup_default_active[2]);
    assert(workgroup_visits >= 1);
    assert(bucketing_visits == 1);

    sygraph::memory::detail::releaseUSM(workgroup_default_active, q);
    sygraph::memory::detail::releaseUSM(bucketing_active, q);
  };

  run_frontier_case(std::integral_constant<sygraph::frontier::frontier_type, sygraph::frontier::frontier_type::mlb>{});
  run_frontier_case(std::integral_constant<sygraph::frontier::frontier_type, sygraph::frontier::frontier_type::bitmap>{});
}
