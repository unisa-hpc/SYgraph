#pragma once

namespace sygraph {
inline namespace v0 {
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
  none,   /**< Frontier implemented as a bit vector. */
};

} // namespace frontier
} // namespace v0
} // namespace sygraph