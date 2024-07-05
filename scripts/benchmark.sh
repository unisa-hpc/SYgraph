#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

BENCHMARK_LIST=("bfs")
GRAPHS=("road_usa" "hollywood-2009" "indochina-2004" "roadNet-CA" "kron_g500-logn21" "uk-2002" "rgg_n_2_24_s0" "soc-LiveJournal1" "soc-orkut" "delaunay_n13" "delaunay_n21" "delaunay_n24")

benchmark="bfs"
build_dir=""
out_dir=""
n_reps=1
graph_dir="$SCRIPT_DIR/datasets/"
graph_source_pairs=()
default_graphs=()
validate="-v"

function list_graphs {
  for graph in "${GRAPHS[@]}"
  do
    echo -n "$graph,"
  done
  echo ""
}

function short_help {
  echo "Use -h or --help for help"
}

function help {
  echo "Usage: $0 --bench <bfs> --bin-dir <path> --out-dir <path> -n <n>"
  echo "Options:"
  echo "  -b,  --bench <bfs>       Benchmark to run"
  echo "  -d,  --bin-dir <path>    Path to the binary directory"
  echo "  -o,  --out-dir <path>    Path to the output directory"
  echo "  -g,  --graph-dir <path>  Path to the graph directory"
  echo "  -G,  --graphs <list>     Comma-separated list of graphs with optional sources (graph:source1,source2;graph2)"
  echo "  -LG, --list-graphs       List the default graphs for the benchmark"
  echo "  -nv, --no-validate       Skip validation"
  echo "  -n,  --reps <n>          Number of repetitions [default: 1]"
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
    -G | --graphs)
      IFS=';' read -r -a graph_source_pairs <<< "$2"
      shift
      shift
      ;;
    -LG | --list-graphs)
      list_graphs
      exit 0
      ;;
    -nv | --no-validate)
      validate=""
      exit 0
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

# Process graph-source pairs
declare -A graph_sources
for pair in "${graph_source_pairs[@]}"; do
  if [[ "$pair" == *":"* ]]; then
    IFS=':' read -r graph sources <<< "$pair"
    IFS=',' read -r -a source_array <<< "$sources"
    graph_sources["$graph"]="${source_array[@]}"
  else
    default_graphs+=("$pair")
  fi
done

# Add default graphs without sources to the graph_sources array
for graph in "${default_graphs[@]}"; do
  graph_sources["$graph"]=""
done

# Check if all graphs are present
for graph in "${!graph_sources[@]}"; do
  if [ ! -f "$graph_dir/$graph/$graph.bin" ]; then
    echo "Graph file not found: $graph_dir/$graph/$graph.bin"
    echo "Aborting..."
    exit 1
  fi
done

echo "Benchmark: $benchmark"
echo "Running benchmarks on graphs: ${!graph_sources[@]}"

for graph in "${!graph_sources[@]}"; do
  sources=(${graph_sources["$graph"]})
  if [ ${#sources[@]} -gt 0 ]; then
    echo "Running benchmark on graph: $graph with sources: ${sources[@]}"
    for source in "${sources[@]}"; do
      echo "Using source: $source"
      $build_dir/$benchmark -b $graph_dir/$graph/$graph.bin -s $source $validate >> $out_dir/$benchmark-$graph.log
    done
  else
    echo "Running benchmark on graph: $graph"
    for i in $(seq 1 $n_reps); do
      echo "Repetition: $i"
      $build_dir/$benchmark -b $graph_dir/$graph/$graph.bin $validate >> $out_dir/$benchmark-$graph.log
    done
  fi
done
