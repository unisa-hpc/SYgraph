#pragma once

#include <sycl/sycl.hpp>
#include <memory>

#include <sygraph/formats/csr.hpp>
#include <sygraph/formats/csc.hpp>
#include <sygraph/formats/coo.hpp>

namespace sygraph {
inline namespace v0 {
namespace graph {
namespace detail {

template <class... graph_view_t>
class graph_impl_t : public ...graph_view_t{
public:
  
};

} // namespace detail
} // namespace graph
} // namespace v0
} // namespace sygraph