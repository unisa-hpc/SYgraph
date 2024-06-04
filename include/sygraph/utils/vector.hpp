#pragma once

#include <sycl/sycl.hpp>
#include <sygraph/utils/memory.hpp>


namespace sygraph {
inline namespace v0 {

/**
 * @todo Remove it, it might be not necessary.
*/
template<typename T>
class Vector {
public:
  Vector(sycl::queue& q, size_t size) : q(q), _size{size} {
    data = sycl::malloc_shared<T>(size, q);
  }

  ~Vector() {
    sycl::free(data, q);
  }

  T* getData() const {
    return data;
  }

  size_t size() const {
    return _size;
  }

private:
  sycl::queue& q;
  T* data;
  size_t _size;
};

} // namespace v0
} // namespace sygraph
