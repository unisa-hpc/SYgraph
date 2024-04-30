#pragma once

#include <chrono>

namespace sygraph {
inline namespace v0 {
namespace detail {
namespace profiling {

struct profiling_info_t {
  std::chrono::time_point<std::chrono::high_resolution_clock> start;
  std::chrono::time_point<std::chrono::high_resolution_clock> end;

  template <typename Duration>
  double get_duration() {
    return std::chrono::duration_cast<Duration>(end - start).count();
  }
};

} // namespace profiling
} // namespace detail
} // namespace v0
} // namespace sygraph