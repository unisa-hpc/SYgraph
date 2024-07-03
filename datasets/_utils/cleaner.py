import os

def clean_graph(folder, only_installation: bool = False):
  try:
    for root, dirs, files in os.walk(folder):
      if only_installation:
        for file in files:
          if file.endswith('.tar.gz'):
            os.remove(os.path.join(root, file))
      else:
        for file in files:
          if not file.endswith('.yaml') and not file.endswith('.py'):
            os.remove(os.path.join(root, file))
  except Exception as e:
    print(f"An error occurred while cleaning the graph: {e}")