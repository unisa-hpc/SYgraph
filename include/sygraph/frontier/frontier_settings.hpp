/*
 * Copyright (c) 2025 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

namespace sygraph {
namespace frontier {

/**
 * @brief Enumeration representing the different views of the frontier.
 */
enum class frontier_view {
  graph,  /**< View representing the entire graph. */
  vertex, /**< View representing only the vertices in the frontier. */
  edge,   /**< View representing only the edges in the frontier. */
  none,   /**< View representing no frontier. */
};

/**
 * @brief Enumeration representing the different types of frontiers.
 */
enum class frontier_type {
  bitmap, /**< Frontier implemented as a bitmap. */
  mlb,    /**< Frontier implemented as a hierarchic bitmap. */
  none,   /**< Dummy frontier. Should not use. */
};

} // namespace frontier
} // namespace sygraph