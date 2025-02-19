# Copyright (c) 2025 University of Salerno
# SPDX-License-Identifier: Apache-2.0

from termcolor import colored
from shutil import get_terminal_size

def parse_graph(data):
  # replace all None with empty string
  for key in data:
    print(key)
    if data[key] is None:
      data[key] = ""
    for key in data['metadata']:
      if data['metadata'][key] is None:
        data['metadata'][key] = ""
    for key in data['structure']:
      if data['structure'][key] is None:
        data['structure'][key] = ""

def beautify_optional_arg(key):
  return key


class GraphInfo:
  def __init__(self, graph: dict) -> None:
    metadata = graph['metadata']
    structure = graph['structure']
    
    self.name = graph['name']
    # process metadata
    self.tags = metadata['tags'] if metadata['tags'] is not None else []
    self.author = metadata['author']
    self.date = metadata['date']
    self.description = metadata['description']
    self.citations = graph['citations'] if 'citations' in graph else None
    
    # process structure
    self.nodes = structure['nodes']
    self.edges = structure['edges']
    self.labeled = structure['labeled']
    self.directed = structure['directed']
    self.weighted = structure['weighted']
    self.optional = {}
    for key in structure['optional']:
      if structure['optional'][key] is not None:
        self.optional[beautify_optional_arg(key)] = structure['optional'][key]
    

class Printer:
  def __init__(self, *, max_width: int = 80, h_spacer = "-", v_spacer = '%') -> None:
    self._max_width = max_width
    self._h_spacer = h_spacer
    self._v_spacer = v_spacer
    
  def _beautify_number(self, number: int) -> str:
    metrics = {1e9: "B", 1e6: "M", 1e3: "K"}
    for metric in metrics:
      if number >= metric:
        val = round(number / metric, 1)
        if val.is_integer():
          return f"{int(val)}{metrics[metric]}"
        return f"{round(number / metric, 1)}{metrics[metric]}"
    return str(number)
      
  def get_string(self, graph: GraphInfo):
    tags = f"\n|{('|'.join(graph.tags))}|" if len(graph.tags) > 0 else ""
    val = f"""---
{graph.name}{tags}
{graph.author} {graph.date}
    
{graph.description}
---
Nodes: {self._beautify_number(graph.nodes)} 
Edges: {self._beautify_number(graph.edges)}

Directed: {'True' if graph.directed else 'False'} 
Weighted: {'True' if graph.weighted else 'False'} 
Labeled: {'True' if graph.labeled else 'False'}
"""
    if len(graph.optional) > 0:
      val += "\n"
      max_len_str = 0 if not len(graph.optional) else max([len(key) for key in graph.optional])
      for key in graph.optional:
        # if it's a number and it's not 0, beautify it
        tmp = graph.optional[key]
        if isinstance(graph.optional[key], int) and graph.optional[key] != 0:
          tmp = self._beautify_number(graph.optional[key])
        tmp = str(tmp)
        key = key.replace("-", " ")
        # justify the key to the right and the value to the left to the middle of the string
        val += f"{key.ljust(max_len_str)} = {tmp}\n"      
    val += "---"
    
    if graph.citations:
      val += "\nCitations\n\n"
      for citation in graph.citations:
        val += f"{citation}\n"
      val += "---"
    
    return val

  def beautify(self, val):
    lines = val.split("\n")
    separator = self._v_spacer + (self._h_spacer * (self._max_width - 2)) + self._v_spacer
    
    # split in two lines if the line is too long, and put a '-' in the middle
    for i, line in enumerate(lines):
      if len(line) >= self._max_width - 4:
        split = self._max_width - 5
        lines[i] = line[:split] + "-"
        lines.insert(i + 1, line[split:])
    
    for i, line in enumerate(lines):
      if line == '---':
        lines[i] = separator
      else:
        lines[i] = f"{self._v_spacer} {line.ljust(self._max_width - 4)} {self._v_spacer}"
    # bold first line
    lines[1] = colored(lines[1], attrs=['bold'])
    return "\n".join(lines)

def print_graph(graph, args = {}):
  graph_info = GraphInfo(graph)
  size = get_terminal_size()

  printer = Printer(max_width = size.columns - 2, **args)
  val = printer.get_string(graph_info)
  val = printer.beautify(val)
  print(val)