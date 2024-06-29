
mapping = {
  'name': (lambda x: x['name'], False),
  'date': (lambda x: x['metadata']['date'], True),
  'nodes': (lambda x: x['structure']['nodes'], True),
  'edges': (lambda x: x['structure']['edges'], True),
}

def order_by_edges(graph):
  return graph['structure']['edges']

def list_items(graphs, order_by, desc):
  graphs = list(graphs.values())
  ordered = sorted(graphs, key=mapping[order_by][0], reverse=mapping[order_by][1] if not desc else not mapping[order_by][1])
  print(f"Avaliable graphs ({len(ordered)}):")
  print()
  for i, info in enumerate(ordered, start=1):
    print(f"{i}) {info['name']}")
  print()