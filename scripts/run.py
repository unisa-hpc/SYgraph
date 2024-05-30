#! /bin/env python3

import os
import re
import argparse
import subprocess
import pandas as pd
from typing import List, Tuple, Dict, Any

from types import SimpleNamespace
from collections import namedtuple

GraphObj = namedtuple('Graph', ['path', 'format', 'undirected'])
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

def parse_graph_file(graph_file):
  path = ''
  undirected = False
  format = ''
  
  if graph_file.endswith(']'):
    bracets = graph_file.split('[')
    path = bracets[0]
    flag = bracets[1][:-1]
    if flag != 'd' and flag != 'u':
      raise ValueError(f'Invalid flag: {flag}')
    undirected = flag == 'u'
  else:
    path = graph_file
    undirected = False
  
  format = path.split('.')[-1]
  if format not in FORMATS:
    raise ValueError(f'Invalid format: {format}')
  
  return GraphObj(path=path, format=format, undirected=undirected)


def parse_output(graph_path, output: str) -> pd.Series:
  total_gpu_time_patt = re.compile(r'Total GPU Time: (\d+\.\d+) ms')
  edge_throughput_patt = re.compile(r'Total Edge-Througput \(MTEPS\): (\d+\.\d+) MTEPS')
  source_patt = re.compile(r'.* on source (\d+)')
  source, gpu_time_ms, edge_throughput = None, None, None
  
  graph_name = os.path.basename(graph_path).split('.')[0]
  for line in output.split('\n'):
    if total_gpu_time_patt.match(line):
      gpu_time_ms = float(total_gpu_time_patt.match(line).group(1))
    elif edge_throughput_patt.match(line):
      edge_throughput = float(edge_throughput_patt.match(line).group(1))
    elif source_patt.match(line):
      source = int(source_patt.match(line).group(1))  
  
  ret = pd.Series({'graph': graph_name, 'source': source, 'gpu_time_ms': gpu_time_ms, 'edge_throughput': edge_throughput, 'success': True})
  return ret

def parse_args(command, graph, source):
  args = [os.path.join('.', command)]
  if graph.format == 'bin':
    args.append('-b')
  args.append(graph.path)
  if graph.format == 'mtx' and graph.undirected:
    args.append('-u')
  if source is not None:
    args.append('-s')
    args.append(str(source))

def exec_command(command, graph: GraphObj, cwd, num_runs, random_source: bool = False, source = None):
  cwd = os.path.abspath(cwd)
  def parse_args():
    args = [os.path.join('.', command)]
    if graph.format == 'bin':
      args.append('-b')
    args.append(graph.path)
    if graph.format == 'mtx' and graph.undirected:
      args.append('-u')
    if source is not None:
      args.append('-s')
      args.append(str(source))
    return args
  args = parse_args()
  
  ret = []
  for it in range(num_runs):
    print(f'Running iteration {it+1}/{num_runs}')
    proc = subprocess.run(args, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
      raise ValueError(f'Error running command: {proc.stdout}')
    val = parse_output(graph.path, proc.stdout)
    if not random_source:
      source = int(val['source'])
      args = parse_args()
    ret.append(val)
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
    
    ret = pd.concat([ret, pd.DataFrame([row])])
  return ret
      
def main():
  args = parse_commands()
  graphs = [parse_graph_file(x) for x in args.graphs]

  data = pd.DataFrame(columns=COLUMNS)
  for graph in graphs:
    print("Running on graph: ", graph.path)
    output = exec_command(args.command, graph, args.directory, args.num_iterations, args.random_source, args.source)
    tmp = pd.DataFrame(output)
    data = pd.concat([data.astype(tmp.dtypes), tmp.astype(data.dtypes)])
  
  if args.parse:
    data = calculate_metrics(data)
  if args.output:
    data.to_csv(args.output, index=False)
  else:
    print(data.to_string())
  

if __name__ == '__main__':
  main()