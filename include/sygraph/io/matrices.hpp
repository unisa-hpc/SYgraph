/*
 * Copyright (c) 2025 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */
#ifdef GENERATE_SAMPLE_DATA
#include <iostream>

namespace sygraph {
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
/**
  o
  |\
  o-o-o-o

  o
*/
constexpr std::string_view two_cc = "6\n"
                                    "0 1 1 0 0 0\n"
                                    "1 0 1 0 0 0\n"
                                    "1 1 0 1 1 0\n"
                                    "0 0 1 0 0 0\n"
                                    "0 0 1 0 0 0\n"
                                    "0 0 0 0 0 0";

} // namespace matrices
} // namespace storage
} // namespace io
} // namespace sygraph
#endif