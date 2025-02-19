/*
 * Copyright (c) 2025 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <sycl/sycl.hpp>

namespace sygraph {


/**
 * @class Event
 * @brief A wrapper class for sycl::event providing additional functionality.
 *
 * The Event class extends the functionality of the sycl::event class by
 * providing additional constructors, assignment operators, and member functions.
 */
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

} // namespace sygraph