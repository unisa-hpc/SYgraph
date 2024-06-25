import os

def clean_graph(folder):
  try:
    for root, dirs, files in os.walk(folder):
      for file in files:
        if not file.endswith('.yaml') and not file.endswith('.py'):
          os.remove(os.path.join(root, file))
  except Exception as e:
    print(f"An error occurred while cleaning the graph: {e}")