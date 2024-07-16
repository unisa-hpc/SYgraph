#pragma once

#include <sygraph/frontier/frontier_settings.hpp>
#include <sygraph/frontier/impls/bitmap_frontier.hpp>
#include <sygraph/frontier/impls/bitvec_frontier.hpp>
#include <sygraph/frontier/impls/hierarchic_bitmap_frontier.hpp>
#include <sygraph/frontier/impls/vector_frontier.hpp>

namespace sygraph {
inline namespace v0 {
namespace frontier {
namespace detail {
template<typename T, frontier_type Type>
class frontier_impl_t;

template<typename T>
class frontier_impl_t<T, frontier_type::none> {
public:
  const bool getDeviceFrontier() const { return false; }
};

template<typename T>
class frontier_impl_t<T, frontier_type::bitmap> : public FrontierBitmap<T> {
  using FrontierBitmap<T>::FrontierBitmap;
};

template<typename T>
class frontier_impl_t<T, frontier_type::hierachic_bitmap> : public FrontierHierarchicBitmap<T> {
  using FrontierHierarchicBitmap<T>::FrontierHierarchicBitmap;
};

template<typename T>
class frontier_impl_t<T, frontier_type::vector> : public FrontierVector<T> {
  using FrontierVector<T>::FrontierVector;
};

template<typename T>
class frontier_impl_t<T, frontier_type::bitvec> : public FrontierBitvec<T> {
  using FrontierBitvec<T>::FrontierBitvec;
};
} // namespace detail

template<typename T, frontier_type Type = frontier_type::hierachic_bitmap>
class Frontier : public detail::frontier_impl_t<T, Type> {
public:
  using detail::frontier_impl_t<T, Type>::frontier_impl_t;
  using type_t = T;
};

template<frontier_view View, frontier_type Type, typename GraphType>
auto makeFrontier(sycl::queue& q, const GraphType& graph) {
  size_t frontier_size = 0;
  if constexpr (View == frontier_view::vertex) {
    frontier_size = graph.getVertexCount();
    return Frontier<typename GraphType::vertex_t, Type>(q, frontier_size);
  } else {
    frontier_size = graph.getEdgeCount();
    return Frontier<typename GraphType::edge_t, Type>(q, frontier_size);
  }
}

template<typename T, frontier_type FT>
void swap(Frontier<T, FT>& a, Frontier<T, FT>& b) {
  if constexpr (FT == frontier_type::bitmap) {
    detail::FrontierBitmap<T>::swap(a, b);
  } else if constexpr (FT == frontier_type::vector) {
    detail::FrontierVector<T>::swap(a, b);
  } else if constexpr (FT == frontier_type::bitvec) {
    detail::FrontierBitvec<T>::swap(a, b);
  } else if constexpr (FT == frontier_type::hierachic_bitmap) {
    detail::FrontierHierarchicBitmap<T>::swap(a, b);
  }
}

} // namespace frontier
} // namespace v0
} // namespace sygraph