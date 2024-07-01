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
  Vector(sycl::queue& q, size_t size) : _q(q), _size{size} { _data = sycl::malloc_shared<T>(size, q); }

  ~Vector() { sycl::free(_data, _q); }

  T* getData() const { return _data; }

  size_t size() const { return _size; }

private:
  sycl::queue& _q;
  T* _data;
  size_t _size;
};

} // namespace v0
} // namespace sygraph
