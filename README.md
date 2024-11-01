![Logo](/docs/logo.png)

SYgraph is a high-performance graph analytics framework built using [SYCL](https://www.khronos.org/sycl/) and C++20. It is designed for heterogeneous systems, allowing for optimized graph processing across various GPU backends, including support for Intel, AMD, and NVIDIA devices. By leveraging SYCL's abstraction, SYgraph aims to maximize hardware resource utilization, specifically optimizing for sparse dataset processing.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## Introduction

SYgraph is intended for researchers and developers focused on high-performance computing (HPC) and graph analytics, providing specialized data structures and mechanisms tailored to GPU execution. This project is especially valuable for applications requiring highly optimized graph traversal and manipulation, making it suitable for large-scale data analysis and machine learning workflows.

## Features

- **Cross-platform Support**: Built with SYCL, SYgraph enables compatibility with various GPUs (Intel, AMD, NVIDIA).
- **Core Primitives**: Supports essential graph processing primitives, including:
  - `Advance`: Efficiently navigates through graph edges.
  - `Filter`: Selectively includes or excludes nodes and edges based on specific conditions.
  - `Compute`: Applies computations across vertices or edges.
  - `Segmented Intersection`: Optimized for handling intersecting edge lists in segments, crucial for operations like triangle counting.
- **Kernel Fusion**: Allows incorporation of custom GPU kernels into these core primitives via lambda functions, reducing kernel launch overhead and enhancing computational efficiency.
- **Optimized Frontiers**: Implements specialized data structures for graph traversal called frontier, implemented as a multi-level bitmap frontier for advanced workload balancing.
- **Sparse Dataset Efficiency**: Benchmark results indicate higher performance on sparse datasets with Intel and AMD GPUs compared to NVIDIA.

## Installation

### Prerequisites

Ensure you have the following dependencies:

- **CMake** 3.18 or higher
- **SYCL Compiler** (e.g., [DPC++](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html), [AdaptiveCpp](https://adaptivecpp.github.io))
- **C++ 20**
- **Doxygen** >= 1.9.1 (optional)

### Install SYgraph

1. Clone the repository:
   ```bash
   git clone https://github.com/unisa-hpc/SYgraph.git
   cd SYgraph
   ```
2. Configure and build the project:
   ```bash
   mkdir build && cd build
   cmake .. -DCMAKE_CXX_COMPILER_PATH=/path/to/sycl/compiler
   make install
   ```
This will create the SYgraph cmake package that can be used in other CMake projects.

### Build Examples
After cloning the repository you can build the example projects with the following procedure.

2. Set the target architecture (only if using oneAPI compiler).
   ```bash
   cmake .. -DCMAKE_CXX_COMPILER_PATH=/path/to/sycl/compiler -DARCH=target_architecture
   ```
   The list of available targets is defined [here](https://github.com/intel/llvm/blob/sycl/sycl/doc/UsersManual.md).

3. Build the project:
   ```bash
   cmake --build . -j
   ```
   The build files will be in the `build/bin` folder.

## Usage
Since SYgraph is a header-only library, you simply need to include the following in your code:

```c++
#include <sygraph/sygraph.hpp>
```

### Examples
If you have enabled the option SYGRAPH_BUILD_EXAMPLES=ON, a set of example algorithms will be available for execution. For details on how to use these examples, type `-h` when running an example to display usage instructions.

```bash
$ ./SYgraph/build/bin/bfs -m ./SYgraph/datasets/hollywood-2009/hollywood-2009.mtx
```

### Dataset Manager
Under the `/datasets` directory, there is a script called `manager.py` that is essential for various dataset-related operations. This script supports tasks such as:
- Downloading datasets;
- Converting datasets into a binary format;
- Viewing details about each dataset.

To see a full list of commands and options, run `$ python manager.py --help`

## Configuration
To configure the project, in the following the list of the available options of the CMake.
|Option|Type|Default|Description|
|-|-|-|-|
|SYGRAPH_BITMAP_SIZE|Integer|32|Bitmap size in bits. It should match the sub-group (i.e., warp, wavefront) size.|
|SYGRAPH_CU_SIZE|Integer|512|Number of threads (`X`) in a compute-unit of the target architecture.|
|SYGRAPH_BUILD_EXAMPLES|Boolean|OFF|Builds example algorithms: BFS, SSSP, BC, CC. If it is ON, it is mandatory to specify also the target architecture with `ARCH` variable (e.g., nvidia_gpu_sm_70) if using the oneAPI compiler.|
|SYGRAPH_ENABLE_PROFILING|Boolean|OFF|Enables kernel profiling.|
|SYGRAPH_ENABLE_PREFETCH|Boolean|OFF|Enable runtime to prefetch shared memory allocation. Turn it OFF for compatibility.|
|SYGRAPH_BUILD_TESTS|Boolean|OFF|Builds tests.|
|SYGRAPH_SAMPLE_DATA|Boolean|OFF|Builds also default sparse datasets for testing purposes.|

## Contributing
We welcome contributions! If you have improvements or bug fixes, please fork the repository and open a pull request against the develop branch. Ensure your changes are tested on multiple backends where possible.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
