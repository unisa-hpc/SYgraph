#include "test_utils.hpp"

int main() {
  auto q = sygraph::tests::makeQueue();
  auto graph = sygraph::tests::buildGraphFromMatrix(q, sygraph::tests::fixtures::line_5);
  uint source = 0;

  auto run_suite = [&](auto& bfs) {
    bfs.init(source);
    auto push_details = bfs.run(sygraph::algorithms::bfs_direction::push);
    sygraph::tests::expectEqual(bfs.getDistances(), std::array<uint, 5>{0, 1, 2, 3, 4});
    assert(!push_details.push_steps.empty());
    assert(push_details.pull_steps.empty());
    bfs.reset();

    bfs.init(source);
    auto pull_details = bfs.run(sygraph::algorithms::bfs_direction::pull);
    sygraph::tests::expectEqual(bfs.getDistances(), std::array<uint, 5>{0, 1, 2, 3, 4});
    assert(pull_details.push_steps.empty());
    assert(!pull_details.pull_steps.empty());
    bfs.reset();

    bfs.init(source);
    auto hybrid_details = bfs.run(sygraph::algorithms::bfs_direction::hybrid, 1.0f, 1.0f);
    sygraph::tests::expectEqual(bfs.getDistances(), std::array<uint, 5>{0, 1, 2, 3, 4});
    assert(hybrid_details.iterations == 5);
  };

  sygraph::algorithms::BFS<decltype(graph)> bfs_mlb(graph);
  run_suite(bfs_mlb);

  sygraph::algorithms::BFS<decltype(graph), sygraph::frontier::frontier_type::bitmap> bfs_bitmap(graph);
  run_suite(bfs_bitmap);
}
