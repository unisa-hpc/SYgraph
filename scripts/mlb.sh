#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) 
SCRIPT_DIR="$SCRIPT_DIR/.."

datasets_path="/data/datasets"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    -d | --datasets)
      datasets_path=$2
      shift
      shift
      ;;
    -h | --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  -d, --datasets <path>  Path to the datasets directory"
      echo "  -h, --help             Display this help message"
      return 0 2>/dev/null
      exit 0
      ;;
    *)
    echo "Invalid argument: $1"
      return 1 2>/dev/null
      exit 1
      ;;
  esac
done

# check if the datasets path exists
if [ ! -d $datasets_path ]; then
  echo "Datasets path does not exist: $datasets_path"
  return 1 2>/dev/null
  exit 1
fi

graphs=("hollywood-2009" "roadNet-CA" "indochina-2004")
wg_size=("32" "64" "128" "256" "512" "1024")
frontier=("bitmap" "mlb")

declare -A SOURCES=(
  ["hollywood-2009"]="320853 927132 517242 884131 247188 42013 214421 1002581 258499 59764"
  ["roadNet-CA"]="63595 69413 162712 215065 231882 242756 291628 313321 337137 380410" 
  ["indochina-2004"]="7226369 4248961 7239584 5741099 5085857 7100239 3031103 4823825 4433225 2162849"
)

function benchmark {
  graph=$1
  wg=$2
  f=$3
  graph_path=$datasets_path/$graph/$graph.bin

  sources=(${SOURCES[$graph]})

  mkdir -p $SCRIPT_DIR/logs/frontier

  log_file=$SCRIPT_DIR/logs/frontier/${graph}_${wg}_${f}.log
  err_file=$SCRIPT_DIR/logs/frontier/${graph}_${wg}_${f}.err

  echo Running bfs-hybrid on $graph with $wg and $f
  for source in "${sources[@]}"
  do
    echo -n '#'
    $SCRIPT_DIR/build/bin/bfs_${f}_${wg} \
      -b $graph_path \
      -s $source \
      -v \
      >> $log_file 2>> $err_file
  done
  echo 
}

for graph in "${graphs[@]}"
do
  for wg in ${wg_size[@]}
  do
    for f in "${frontier[@]}"
    do
      benchmark $graph $wg $f
    done
  done
done
