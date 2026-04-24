#include "test_utils.hpp"
#include <type_traits>

int main() {
  using frontier_view_t = sygraph::frontier::frontier_view;
  using frontier_type_t = sygraph::frontier::frontier_type;

  auto q = sygraph::tests::makeQueue();
  auto graph = sygraph::tests::buildGraphFromMatrix(q, sygraph::tests::fixtures::star_5);
  auto run_case = [&](auto frontier_tag) {
    constexpr auto frontier_type = decltype(frontier_tag)::value;

    auto input = sygraph::frontier::makeFrontier<frontier_view_t::vertex, frontier_type>(q, graph);
    auto output = sygraph::frontier::makeFrontier<frontier_view_t::vertex, frontier_type>(q, graph);

    for (uint v : {0u, 1u, 2u, 3u, 4u}) { input.insert(v); }

    auto external = sygraph::operators::filter::external(graph, input, output, [](auto vertex) { return vertex % 2 == 0; });
    external.waitAndThrow();
    sygraph::tests::expectFrontier(output, std::vector<uint>{0, 2, 4});

    auto inplace = sygraph::operators::filter::inplace(graph, input, [](auto vertex) { return vertex < 2; });
    inplace.waitAndThrow();
    sygraph::tests::expectFrontier(input, std::vector<uint>{2, 3, 4});

    auto seen = sygraph::memory::detail::memoryAlloc<uint32_t, sygraph::memory::space::shared>(graph.getVertexCount(), q);
    q.fill(seen, 0U, graph.getVertexCount()).wait();

    auto execute = sygraph::operators::compute::execute<frontier_view_t::vertex>(graph, output, [=](auto vertex) { seen[vertex] = 1; });
    execute.waitAndThrow();

    std::vector<uint32_t> expected_seen{1, 0, 1, 0, 1};
    std::vector<uint32_t> actual_seen(graph.getVertexCount());
    for (size_t i = 0; i < actual_seen.size(); ++i) { actual_seen[i] = seen[i]; }
    sygraph::tests::expectEqual(actual_seen, expected_seen);

    uint32_t sum = 0;
    auto reduce = sygraph::operators::compute::reduce<frontier_view_t::vertex, sycl::plus<uint32_t>>(
        graph, output, sum, [=](auto vertex, auto& acc) { acc += static_cast<uint32_t>(vertex); });
    reduce.waitAndThrow();
    assert(sum == 6);

    sygraph::memory::detail::releaseUSM(seen, q);
  };

  run_case(std::integral_constant<frontier_type_t, frontier_type_t::mlb>{});
  run_case(std::integral_constant<frontier_type_t, frontier_type_t::bitmap>{});
}
