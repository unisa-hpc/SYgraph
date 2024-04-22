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

template <typename T, 
          FrontierView view = FrontierView::vertex,
          FrontierType type = FrontierType::bitmap>
class Frontier : public detail::frontier_impl_t<T, type> {
public:
  using detail::frontier_impl_t<T, type>::frontier_impl_t;
  using type_t = T;
};

template <FrontierView view,
          FrontierType type,
          typename GraphType>
auto make_frontier(sycl::queue& q, const GraphType& graph) {
  size_t frontier_size = 0;
  if constexpr (view == FrontierView::vertex) {
    frontier_size = graph.get_vertex_count();
    return Frontier<typename GraphType::vertex_t, view, type>(q, frontier_size);
  } else {
    frontier_size = graph.get_edge_count();
    return Frontier<typename GraphType::edge_t, view, type>(q, frontier_size);
  }
}

} // namespace frontier
} // namespace v0
} // namespace sygraph