#pragma once

namespace sygraph {
inline namespace v0 {
namespace operators {

enum class load_balancer {
  workitem_mapped,
  subgroup_mapped,
  workgroup_mapped,
};

enum class direction {
  push,
  pull,
};

} // namespace operators
} // namespace v0
} // namespace sygraph