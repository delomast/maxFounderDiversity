# For relative imports to work in Python 3.6
# import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import os, sys, shutil

from pathlib import Path
sys.path[0] = str(Path(__file__).parents[0])
print(sys.path[0])
if os.path.exists(r'__pycache__'): 
  shutil.rmtree(r'__pycache__')
