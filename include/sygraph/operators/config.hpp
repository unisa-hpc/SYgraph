#pragma once

namespace sygraph {
inline namespace v0 {
namespace operators {

enum class LoadBalancer {
  workitem_mapped,
  subgroup_mapped,
  workgroup_mapped,
};

enum class Direction {
  push,
  pull,
};

} // namespace operators
} // namespace v0
} // namespace sygraph