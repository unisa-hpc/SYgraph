#ifdef GENERATE_SAMPLE_DATA
#include <iostream>

namespace sygraph {
inline namespace v0 {
namespace io {
namespace storage {
namespace matrices {

/**
  o
  |\
  o-o-o-o
  |
  o
*/
constexpr std::string_view symmetric_6nodes = "6\n"
                                              "0 1 1 0 0 0\n"
                                              "1 0 1 0 0 0\n"
                                              "1 1 0 1 1 0\n"
                                              "0 0 1 0 0 0\n"
                                              "0 0 1 0 0 1\n"
                                              "0 0 0 0 1 0";

} // namespace matrices
} // namespace storage
} // namespace io
} // namespace v0
} // namespace sygraph
#endif