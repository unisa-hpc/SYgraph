#pragma once

#include <sycl/sycl.hpp>

namespace sygraph {
inline namespace v0 {

class Event : public sycl::event {
public:
  Event() = default;
  Event(const sycl::event& e) : sycl::event(e) {}
  Event(const Event& e) : sycl::event(e) {}
  Event(Event&& e) : sycl::event(e) {}
  Event& operator=(const Event& e) {
    sycl::event::operator=(e);
    return *this;
  }
  Event& operator=(Event&& e) {
    sycl::event::operator=(e);
    return *this;
  }
  ~Event() = default;

  void wait() { sycl::event::wait(); }

  void waitAndThrow() { sycl::event::wait_and_throw(); }
};

} // namespace v0
} // namespace sygraph