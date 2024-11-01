#pragma once

namespace sygraph {
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
} // namespace sygraph