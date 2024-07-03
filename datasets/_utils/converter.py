import os
import glob
import subprocess

def convert_graph(converter_path, folder, undirected: bool, always: bool = False):
  if not (lst := glob.glob(f'{folder}/*.mtx')):
    return
  mtx = lst[0]
  basename = os.path.basename(mtx).split('.')[0]
  bin_file = f'{folder}/{basename}.bin'

  args = [converter_path, mtx, bin_file]
  if undirected:
    args.append('-u')
  
  if os.path.exists(bin_file) and not always:
    # ask user if they want to convert again
    if input(f'{basename} already converted. Do you want to convert again? [y/n]: ').lower() != 'y':
      return    
  
  print(f'Converting {basename}')
  subprocess.run(args, check=True)