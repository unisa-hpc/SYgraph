#pragma once

namespace sygraph {
inline namespace v0 {
namespace operators {
namespace load_balancing {

enum class LoadBalancer {
  Static,
  Dynamic,
  MergeBased,
};

} // namespace load_balancing
} // namespace operators
} // namespace v0
} // namespace sygraph