#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

BENCHMARK_LIST=("bfs")
GRAPHS=("road_usa" "hollywood-2009" "indochina-2004" "roadNet-CA" "kron_g500-logn21" "uk-2002" "rgg_n_2_24_s0" "soc-LiveJournal1" "soc-orkut" "delaunay_n13" "delaunay_n21" "delaunay_n24")

benchmark="bfs"
build_dir=""
out_dir=""
n_reps=1
graph_dir="$SCRIPT_DIR/datasets/"

function short_help {
  echo "Use -h or --help for help"
}

function help {
  echo "Usage: $0 --bench <bfs> --bin-dir <path> --out-dir <path> -n <n>"
  echo "Options:"
  echo "  -b, --bench <bfs>       Benchmark to run"
  echo "  -d, --bin-dir <path>    Path to the binary directory"
  echo "  -o, --out-dir <path>    Path to the output directory"
  echo "  -g, --graph-dir <path>  Path to the graph directory"
  echo "  -n, --reps <n>          Number of repetitions [default: 1]"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    -n | --reps)
      n_reps=$2
      shift
      shift
      ;;
    -b | --bench)
      benchmark=$2
      shift
      shift
      ;;
    -d | --bin-dir)
      build_dir=$2
      shift
      shift
      ;;
    -o | --out-dir)
      out_dir=$2
      shift
      shift
      ;;
    -g | --graph-dir)
      graph_dir=$2
      shift
      shift
      ;;
    -h | --help)
      help
      exit 0
      ;;
    *)
    echo "Invalid argument: $1"
      short_help
      exit 1
      ;;
  esac
done

# check bin dir
if [ -z "$build_dir" ]; then
  echo "Binary directory not specified"
  short_help
  exit 1
fi

# check log dir
if [ -z "$out_dir" ]; then
  echo "Output directory not specified"
  short_help
  exit 1
fi

# check graph dir
if [ ! -d "$graph_dir" ]; then
  echo "Graph directory not found: $graph_dir"
  short_help
  exit 1
fi

# check if selected benchmark is valid
if [[ ! " ${BENCHMARK_LIST[@]} " =~ " ${benchmark} " ]]; then
  echo "Invalid benchmark: $benchmark"
  echo "Valid benchmarks: ${BENCHMARK_LIST[@]}"
  exit 1
fi

# check if all graphs are present
for graph in "${GRAPHS[@]}"
do
  if [ ! -f "$graph_dir/$graph/$graph.bin" ]; then
    echo "Graph file not found: $graph_dir/$graph/$graph.bin"
    echo "Aborting..."
    exit 1
  fi
done

echo "Benchmark: $benchmark"
echo "Running benchmarks on graphs: ${GRAPHS[@]}"

for graph in "${GRAPHS[@]}"
do
  echo "Running benchmark on graph: $graph"
  for i in $(seq 1 $n_reps)
  do
    echo "Repetition: $i"
    $build_dir/$benchmark -b $graph_dir/$graph/$graph.bin -v >> $out_dir/$benchmark$n_reps-$graph.log
  done
done
