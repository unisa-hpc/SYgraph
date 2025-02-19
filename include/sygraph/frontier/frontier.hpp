/*
 * Copyright (c) 2025 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <sygraph/frontier/frontier_settings.hpp>
#include <sygraph/frontier/impls/bitmap_frontier.hpp>
#include <sygraph/frontier/impls/mlb_frontier.hpp>

namespace sygraph {
namespace frontier {
namespace detail {
template<typename T, frontier_type Type>
class frontier_impl_t;

template<typename T>
class frontier_impl_t<T, frontier_type::none> {
public:
  frontier_impl_t() {};
  const bool getDeviceFrontier() const { return false; }
};

template<typename T>
class frontier_impl_t<T, frontier_type::bitmap> : public FrontierBitmap<T> {
  using FrontierBitmap<T>::FrontierBitmap;
};

template<typename T>
class frontier_impl_t<T, frontier_type::mlb> : public FrontierMLB<T> {
  using FrontierMLB<T>::FrontierMLB;
};
} // namespace detail

/**
 * @class Frontier
 * @brief A class that extends the functionality of detail::frontier_impl_t.
 *
 * This class serves as a public interface for the frontier implementation.
 * It inherits from detail::frontier_impl_t and exposes its constructor.
 *
 * @tparam T The type parameter for the frontier implementation.
 * @tparam Type The frontier implementation (bitmap, mlb).
 */
template<typename T, frontier_type Type = frontier_type::mlb>
class Frontier : public detail::frontier_impl_t<T, Type> {
public:
  using detail::frontier_impl_t<T, Type>::frontier_impl_t;
  using type_t = T;
};

/**
 * @brief Creates a Frontier object based on the specified view type.
 *
 * This function initializes a Frontier object using the provided SYCL queue and graph.
 * The size of the Frontier is determined by the view type:
 * - If the view type is `frontier_view::vertex`, the Frontier size is set to the number of vertices in the graph.
 * - If the view type is `frontier_view::edge`, the Frontier size is set to the number of edges in the graph.
 *
 * @tparam View The view type, which can be either `frontier_view::vertex` or `frontier_view::edge`.
 * @tparam Type The type of elements stored in the Frontier.
 * @tparam GraphType The type of the graph.
 * @param q The SYCL queue used for the Frontier.
 * @param graph The graph from which the Frontier is created.
 * @return A Frontier object of the appropriate type and size.
 * @throws std::runtime_error If the view type is invalid.
 */
template<frontier_view View, frontier_type Type, typename GraphType>
auto makeFrontier(sycl::queue& q, const GraphType& graph) {
  size_t frontier_size = 0;
  if constexpr (View == frontier_view::vertex) {
    frontier_size = graph.getVertexCount();
    return Frontier<typename GraphType::vertex_t, Type>(q, frontier_size);
  } else if constexpr (View == frontier_view::edge) {
    frontier_size = graph.getEdgeCount();
    return Frontier<typename GraphType::edge_t, Type>(q, frontier_size);
  } else {
    throw std::runtime_error("Invalid frontier view");
  }
}

/**
 * @brief Swaps the contents of two Frontier objects.
 *
 * This function swaps the contents of two Frontier objects based on their frontier type.
 * If the frontier type is bitmap, it uses FrontierBitmap's swap method.
 * If the frontier type is mlb, it uses FrontierMLB's swap method.
 *
 * @tparam T The type of elements in the Frontier.
 * @tparam FT The frontier type, which determines the specific swap method to use.
 * @param a The first Frontier object to swap.
 * @param b The second Frontier object to swap.
 */
template<typename T, frontier_type FT>
void swap(Frontier<T, FT>& a, Frontier<T, FT>& b) {
  if constexpr (FT == frontier_type::bitmap) {
    detail::FrontierBitmap<T>::swap(a, b);
  } else if constexpr (FT == frontier_type::mlb) {
    detail::FrontierMLB<T>::swap(a, b);
  }
}

} // namespace frontier
} // namespace sygraph