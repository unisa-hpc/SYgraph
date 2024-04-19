#pragma once

#include <sycl/sycl.hpp>

namespace sygraph {
inline namespace v0 {

class event : sycl::event {
public:
  event() = default;
  event(const sycl::event& e) : sycl::event(e) {}
  event(const event& e) : sycl::event(e) {}
  event(event&& e) : sycl::event(e) {}
  event& operator=(const event& e) {
    sycl::event::operator=(e);
    return *this;
  }
  event& operator=(event&& e) {
    sycl::event::operator=(e);
    return *this;
  }
  ~event() = default;

  void wait() {
    sycl::event::wait();
  }

  void wait_and_throw() {
    sycl::event::wait_and_throw();
  }
};

} // namespace v0
} // namespace sygraph