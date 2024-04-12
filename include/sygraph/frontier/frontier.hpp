#pragma once

#include <sygraph/frontier/frontier_settings.hpp>
#include <sygraph/frontier/impls/bitmap_frontier.hpp>

namespace sygraph {
inline namespace v0 {
namespace frontier {
namespace detail {
template <typename type_t,  
          FrontierType _type>
class frontier_impl_t;

template <typename type_t>
class frontier_impl_t<type_t, FrontierType::bitmap> : public frontier_bitmap_t<type_t> {
  using frontier_bitmap_t<type_t>::frontier_bitmap_t;
};
} // namespace detail

template <typename type_t, 
          FrontierType type = FrontierType::bitmap,
          FrontierView view = FrontierView::vertex>
class Frontier : public detail::frontier_impl_t<type_t, type> {
  using detail::frontier_impl_t<type_t, type>::frontier_impl_t;

  
};

} // namespace frontier
} // namespace v0
} // namespace sygraph