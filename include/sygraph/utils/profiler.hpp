#pragma once

#ifdef ENABLE_PROFILING

#include <sycl/sycl.hpp>
#include <vector>
#include <unordered_map>
#include <sygraph/sycl/event.hpp>

namespace sygraph {
inline namespace v0 {

namespace details {

static std::unordered_map<std::string, std::vector<sygraph::event>> events;


} // namespace details

class profiler {
public:
  static void add_event(sygraph::event event, std::string tag = "") {\
    if (tag.empty()) {
      tag = "default";
    }
    if (details::events.find(tag) == details::events.end()) {
      details::events[tag] = std::vector<sygraph::event>();
    }
    details::events[tag].push_back(event);
  }

  static void clear() {
    details::events.clear();
  }

  static void print() {
    double milliseconds = 0.0;
    double total = 0.0;
    for (auto& [tag, events] : details::events) {
      std::cout << "[" << tag << "]";
      for (auto& event : events) {
        milliseconds += static_cast<double>(event.get_profiling_info<sycl::info::event_profiling::command_end>() - event.get_profiling_info<sycl::info::event_profiling::command_start>()) / 1000000;
      }
      std::cout << " GPU Time: " << milliseconds << " ms" << std::endl;
      total += milliseconds;
      milliseconds = 0.0;
    }
    std::cout << "Total GPU Time: " << total << " ms" << std::endl;
  }

};

} // namespace v0
} // namespace sygraph

#endif