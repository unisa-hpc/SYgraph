#! /usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

from typing import Dict
import yaml
import argparse
import glob
import os
import _utils as utl


def get_available_graphs() -> Dict[str, Dict]:
  graphs = {}
  for f in glob.glob('*'):
    # if f is folder, then it is a graph
    if os.path.isdir(f) and not f.startswith('_') and not f.startswith('.'):
      # check if there is a info.yaml file inside
      path = f + '/info.yaml'
      if os.path.isfile(path):
        # read yaml field and append to graphs
        with open(path, 'r') as file:
          info = yaml.safe_load(file)
          info['folder'] = os.path.abspath(f)
          name = info['name']
          graphs[name] = info
        
  return graphs


def list_command(args: argparse.Namespace, graphs: Dict[str, Dict]):
  if not args.filter:
    args.filter = []
  utl.list_items(graphs, order_by=args.order_by, desc=args.desc, filters=args.filter)

def download_command(args: argparse.Namespace, graphs: Dict[str, Dict]):
  to_download = []
  if args.all:
    to_download = list(graphs.keys())
  else:
    to_download = args.graph
  for name in to_download:
    info = graphs[name]
    utl.download_and_extract(name, info['url'], info['folder'])

def clean_command(args: argparse.Namespace, graphs: Dict[str, Dict]):
  to_clean = []
  if args.all:
    to_clean = list(graphs.keys()) 
  else:
    to_clean = args.graph
  for name in to_clean:
    info = graphs[name]
    utl.clean_graph(info['folder'], args.only_installation)
    
def convert_command(args: argparse.Namespace, graphs: Dict[str, Dict]):
  to_convert = []
  if args.all:
    to_convert = list(graphs.keys())
  else:
    to_convert = args.graph
  for name in to_convert:
    info = graphs[name]
    utl.convert_graph(args.converter_path, info['folder'], args.undirected, args.always)
  

def info_command(args: argparse.Namespace, graphs: Dict[str, Dict]):
  info = graphs[args.graph]
  if args.json:
    print(info)
  else:
    utl.print_graph(info)
  
def main():
  graphs = get_available_graphs()
  
  parser = argparse.ArgumentParser(description='Graph repo manager')
  # define commands 
  subparsers = parser.add_subparsers(dest='command')
  
  # list command
  list_parser = subparsers.add_parser('list', help='List available graphs', description='List available graphs')
  list_parser.set_defaults(func=list_command)
  list_parser.add_argument('--order-by', choices=['name', 'date', 'nodes', 'edges'], default='name', help='Order by field')
  list_parser.add_argument('--desc', action='store_true', help='Order in decreasing order')
  list_parser.add_argument('-f', '--filter', nargs='*', help='Filter graphs by parameter. Format: field[=,<,>,<=,>=]value', metavar='FIELD[=,<,>,<=,>=]VALUE')
  
  # add download command
  download_parser = subparsers.add_parser('download', help='Download a graph', description='Download one or more graph')
  # set a list of graphs as argument of the download command
  download_parser.set_defaults(func=download_command)
  download_parser.add_argument('-a', '--all', action='store_true', help='Download all graphs')
  download_parser.add_argument('graph', nargs='*', help='graph(s) to download', metavar='GRAPH')

  # add info command
  info_parser = subparsers.add_parser('info', help='Get info of a graph', description='Get info of a graph')
  info_parser.set_defaults(func=info_command)
  # set a list of graphs as argument of the info command
  info_parser.add_argument('graph', help='graph to analyze', metavar='GRAPH')
  info_parser.add_argument('--json', action='store_true', help='Print info in json format')
  
  # add clean command
  clean_parser = subparsers.add_parser('clean', help='Clean downloaded graphs', description='Clean one or more downloaded graphs')
  clean_parser.set_defaults(func=clean_command)
  clean_parser.add_argument('-a', '--all', action='store_true', help='Clean all graphs')
  clean_parser.add_argument('--only-installation', action='store_true', help='Clean only installation files')
  clean_parser.add_argument('graph', nargs='*', help='Graph(s) to clean', metavar='GRAPH')
  
  # add convert command
  convert_parser = subparsers.add_parser('convert', help='Convert a graph', description='Convert a graph to binary format')
  convert_parser.set_defaults(func=convert_command)
  convert_parser.add_argument('converter_path', help='Path to the converter')
  convert_parser.add_argument('-a', '--all', action='store_true', help='Download all graphs')
  convert_parser.add_argument('-u', '--undirected', action='store_true', help='Convert to undirected graph')
  convert_parser.add_argument('-y', '--always', action='store_true', help='Always convert the graph')
  convert_parser.add_argument('graph', nargs='*', help='graph(s) to download', metavar='GRAPH')
  
  # parse arguments
  args = parser.parse_args()
  args.func(args, graphs)
      

if __name__ == '__main__':
  main()
  