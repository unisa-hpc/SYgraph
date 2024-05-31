import matplotlib.pyplot as plt
import pandas as pd
import seaborn
import argparse

if __name__ == '__main__':
  seaborn.set_theme()
  
  parser = argparse.ArgumentParser(description='Plot graph algorithm results')
  parser.add_argument('input', type=str, help='Input file')
  parser.add_argument('output', type=str, help='Output file')
  args = parser.parse_args()
  
  df = pd.read_csv(args.input)
  
  seaborn.violinplot(data=df, x='graph', y='gpu_time_ms', split=True)
  plt.ylabel('GPU Time (ms)')
  plt.xlabel('Graph')
  plt.xticks(rotation=45)
  
  plt.tight_layout()
  plt.savefig(args.output)