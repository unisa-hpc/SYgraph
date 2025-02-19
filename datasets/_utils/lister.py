# Copyright (c) 2025 University of Salerno
# SPDX-License-Identifier: Apache-2.0

mapping = {
  'name': (lambda x: x['name'], False),
  'date': (lambda x: x['metadata']['date'], True),
  'nodes': (lambda x: x['structure']['nodes'], True),
  'edges': (lambda x: x['structure']['edges'], True),
}

FILTERS_OPERATORS = {'=': (lambda a, b: a == b) , '!=': (lambda a, b: a != b), '<': (lambda a, b: a < b), '>': (lambda a, b: a > b), '<=': (lambda a, b: a <= b), '>=': (lambda a, b: a >= b)}

def order_by_edges(graph):
  return graph['structure']['edges']

def find_type(value):
  if value.isdigit():
    return int(value)
  elif value == 'True' or value == 'False':
    return value == 'True'
  return value

def find_field(graph, field):
  if field in graph['structure']:
    return graph['structure'][field]
  if field in graph['structure']['optional']:
    return graph['structure']['optional'][field]
  if field in graph:
    return graph[field]
  if field in graph['metadata']:
    return graph['metadata'][field]

  return None

def process_filter(graphs: list, flt: str):
  operator = None
  for op in FILTERS_OPERATORS.keys():
    if op in flt:
      operator = op
      break
  if operator is None:
    raise ValueError(f"Invalid filter: {flt}")

  field, value = flt.split(operator)
  value = find_type(value)
  
  filtered = []
  for g in graphs:
    op = FILTERS_OPERATORS[operator]
    attr = find_field(g, field)
    if attr is not None and op(attr, value):
      filtered.append(g)
    
  return filtered

def print_graphs(graphs):
  print(f"Avaliable graphs ({len(graphs)}):")
  print()
  for i, info in enumerate(graphs, start=1):
    print(f"{i}) {info['name']}")
  print()
  

def list_items(graphs, order_by, desc, filters):
  graphs = list(graphs.values())
  ordered = sorted(graphs, key=mapping[order_by][0], reverse=mapping[order_by][1] if not desc else not mapping[order_by][1])
  
  for filter in filters:
    ordered = process_filter(ordered, filter)
  
  print_graphs(ordered)