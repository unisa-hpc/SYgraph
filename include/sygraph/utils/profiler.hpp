#pragma once

#ifdef ENABLE_PROFILING

#include <sycl/sycl.hpp>
#include <sygraph/sycl/event.hpp>
#include <unordered_map>
#include <vector>

namespace sygraph {
inline namespace v0 {

namespace details {

static std::unordered_map<std::string, std::vector<sygraph::event>> events;
static size_t num_visited_edges = 0;

} // namespace details

class profiler {
public:
  static void addEvent(sygraph::event event, std::string tag = "") {
    if (tag.empty()) { tag = "default"; }
    if (details::events.find(tag) == details::events.end()) { details::events[tag] = std::vector<sygraph::event>(); }
    details::events[tag].push_back(event);
  }

  static void addVisitedEdges(size_t visited_edges) { details::num_visited_edges += visited_edges; }

  static void clear() {
    details::events.clear();
    details::num_visited_edges = 0;
  }

  static void print() {
    double milliseconds = 0.0;
    double total_ms = 0.0;
    for (auto& [tag, events] : details::events) {
      std::cout << " Kernel [" << tag << " x " << events.size() << "]";
      for (auto& event : events) {
        milliseconds += static_cast<double>(event.get_profiling_info<sycl::info::event_profiling::command_end>()
                                            - event.get_profiling_info<sycl::info::event_profiling::command_start>())
                        / 1e6;
      }
      std::cout << " Time: " << milliseconds << " ms" << std::endl;
      total_ms += milliseconds;
      milliseconds = 0.0;
    }
    std::cout << "Total GPU Time: " << total_ms << " ms" << std::endl;
    std::cout << "Total Edge-Througput (MTEPS): " << ((details::num_visited_edges / 1e6) / (total_ms / 1e3)) << " MTEPS" << std::endl;
  }
};

} // namespace v0
} // namespace sygraph

#endif