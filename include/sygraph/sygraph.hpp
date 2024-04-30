#pragma once

// Include formats
#include <sygraph/formats/csr.hpp>
#include <sygraph/formats/coo.hpp>

// Include Frontier
#include <sygraph/frontier/frontier.hpp>
#include <sygraph/frontier/frontier_settings.hpp>

// Inlcude Graphs
#include <sygraph/graph/graph.hpp>
#include <sygraph/graph/properties.hpp>
#include <sygraph/graph/build.hpp>

// Include IO
#ifdef GENERATE_SAMPLE_DATA
#include <sygraph/io/matrices.hpp>
#endif
#include <sygraph/io/read_csr.hpp>
#include <sygraph/io/read_coo.hpp>

// Include utils
#include <sygraph/utils/memory.hpp>
#include <sygraph/sycl/event.hpp>

// Include operators
#include <sygraph/operators/config.hpp>
#include <sygraph/operators/advance/advance.hpp>
#include <sygraph/operators/for/for.hpp>

// Include algorithms
#include <sygraph/algorithms/bfs.hpp>
