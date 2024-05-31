#! /bin/env python3

import os
import re
import argparse
import subprocess
import pandas as pd
from typing import List, Tuple, Dict, Any
from tqdm import tqdm

from types import SimpleNamespace
from collections import namedtuple

GraphObj = namedtuple('Graph', ['path', 'name', 'format', 'undirected', 'source'])
METRICS = ['mean', 'median', 'std', 'min', 'max']
COMMANDS = ['bfs', 'sssp']
DEF_COLS = ['graph', 'source', 'success']
MET_COLS = ['gpu_time_ms', 'edge_throughput']
COLUMNS = DEF_COLS + MET_COLS
FORMATS = ['mtx', 'bin']    

def parse_commands():
  parser = argparse.ArgumentParser(description='Run a graph algorithm')
  parser.add_argument('command', type=str, help='Command to run', choices=COMMANDS)
  parser.add_argument('graphs', nargs='+', type=str, help='Graph file')
  parser.add_argument('--source', type=int, help='Source vertex', default=None)
  parser.add_argument('--random-source', '-r', action='store_true', help='Random source vertex')
  parser.add_argument('--num-iterations', '-n', type=int, help='Number of iterations', default=1)
  parser.add_argument('--validate', '-v', action='store_true', help='Validate the output')
  parser.add_argument('--directory', '-d', type=str, help='Working direcotry', default='.')
  parser.add_argument('--parse', '-P', action='store_true', help='Parse output')
  parser.add_argument('--output', '-o', type=str, help='Output file', default=None)
  
  return parser.parse_args()

def parse_graph_file(graph_file: str):
  
  patt = re.compile(r'^(.+\/([^\/]+)\.(bin|mtx))(?:\[(\d+)?:?(u|d)?\])?$')
  path, name, format, source, flag = patt.search(graph_file).groups()
  if flag is not None:
    flag = flag == 'u'
  else:
    flag = False
    
  if source is not None:
    source = int(source)
    
  if format is None:
    raise ValueError(f'Invalid graph file: {graph_file}')
  
  return GraphObj(path=path, name=name, format=format, undirected=flag, source=source)


def parse_output(graph: GraphObj, output: str) -> pd.Series:
  total_gpu_time_patt = re.compile(r'Total GPU Time: (\d+\.\d+) ms')
  edge_throughput_patt = re.compile(r'Total Edge-Througput \(MTEPS\): (\d+\.\d+) MTEPS')
  source_patt = re.compile(r'.* on source (\d+)')
  source, gpu_time_ms, edge_throughput = graph.source, None, None
  
  for line in output.split('\n'):
    if total_gpu_time_patt.match(line):
      gpu_time_ms = float(total_gpu_time_patt.match(line).group(1))
    elif edge_throughput_patt.match(line):
      edge_throughput = float(edge_throughput_patt.match(line).group(1))
    elif source_patt.match(line):
      source = int(source_patt.match(line).group(1))  
  
  ret = pd.Series({'graph': graph.name, 'source': source, 'gpu_time_ms': gpu_time_ms, 'edge_throughput': edge_throughput, 'success': True})
  return ret

def calculate_metrics(data: pd.DataFrame) -> pd.DataFrame:
  cols = [f'{col}_{met}' for col in MET_COLS for met in METRICS]
  cols = DEF_COLS + cols
  ret = pd.DataFrame(columns=cols)
   
  for name, group in data.groupby('graph'):
    row = {col: None for col in cols}
    row['graph'] = name
    row['source'] = group.iloc[0]['source']
    row['success'] = group['success'].all()
    
    for col in MET_COLS:
      for met in METRICS:
        row[f'{col}_{met}'] = group[col].agg(met)
    
    tmp = pd.DataFrame([row])
    ret = pd.concat([ret.astype(tmp.dtypes), tmp.astype(ret.dtypes)])
  return ret


def exec_command(command, graph: GraphObj, cwd):
  cwd = os.path.abspath(cwd)
  args = [os.path.join('.', command)]
  if graph.format == 'bin':
    args.append('-b')
  args.append(graph.path)
  if graph.format == 'mtx' and graph.undirected:
    args.append('-u')
  if graph.source is not None:
    args.append('-s')
    args.append(str(graph.source))

  proc = subprocess.run(args, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
  if proc.returncode != 0:
    raise ValueError(f'Error running command: {proc.stdout}')
  val = parse_output(graph, proc.stdout)
  return val
      
def main():
  args = parse_commands()
  graphs = [parse_graph_file(x) for x in args.graphs]

  data = pd.DataFrame(columns=COLUMNS)
  for graph in graphs:
    for _ in tqdm(range(args.num_iterations), desc=f'Running on graph {graph.name}'):
      output = exec_command(args.command, graph, args.directory)
      if not args.random_source and graph.source is None:
        graph = GraphObj(graph.path, graph.name, graph.format, graph.undirected, int(output['source']))
      data.loc[len(data)] = output
      
  
  if args.parse:
    data = calculate_metrics(data)
  if args.output:
    data.to_csv(args.output, index=False)
  else:
    print(data.to_string())
  

if __name__ == '__main__':
  main()