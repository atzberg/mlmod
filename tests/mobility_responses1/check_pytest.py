# attempt to auto install, if not available 
try:
  import pytest;
except:
  print("Please install 'pytest' python package."); 
  print("Attempting to auto-install the package:");
  try:
    import os;
    os.system('pip install pytest'); # install if not already 
    import pytest;
  except:
    print("Auto-install failed, please install it manually.");

