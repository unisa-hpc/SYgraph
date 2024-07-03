#pragma once

// Include formats
#include <sygraph/formats/coo.hpp>
#include <sygraph/formats/csr.hpp>

// Include Frontier
#include <sygraph/frontier/frontier.hpp>
#include <sygraph/frontier/frontier_settings.hpp>

// Inlcude Graphs
#include <sygraph/graph/build.hpp>
#include <sygraph/graph/graph.hpp>
#include <sygraph/graph/properties.hpp>

// Include IO
#ifdef GENERATE_SAMPLE_DATA
#include <sygraph/io/matrices.hpp>
#endif
#include <sygraph/io/read_coo.hpp>
#include <sygraph/io/read_csr.hpp>

// Include utils
#include <sygraph/sycl/event.hpp>
#include <sygraph/utils/memory.hpp>

// Include operators
#include <sygraph/operators/advance/advance.hpp>
#include <sygraph/operators/config.hpp>
#include <sygraph/operators/for/for.hpp>
#include <sygraph/operators/intersection/intersection.hpp>

// Include algorithms
#include <sygraph/algorithms/bfs.hpp>
#include <sygraph/algorithms/sssp.hpp>
#include <sygraph/algorithms/tc.hpp>

// Include sync
#include <sygraph/sync/atomics.hpp>

// Include profiler
#ifdef ENABLE_PROFILING
#include <sygraph/utils/profiler.hpp>
#endif