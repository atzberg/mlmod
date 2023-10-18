# Attempts to quick install the package using pip and sources on-line 

import subprocess;

# version and URL
vstr='1.0.3';
server='https://web.math.ucsb.edu/~atzberg/mlmod/distr'

# distributions to try
manylinux_list = ['manylinux_2_24_x86_64','any'];

# try to quick install
print("-"*80);
print("Trying different wheels to install the package.");
flag_installed = False;
flag_cont=True; nn = len(manylinux_list);I=0;
while flag_cont:
  print("."*80);
  mlstr=manylinux_list[I];
  wheel_name='mlmod_lammps-%s-py3-none-%s.whl'%(vstr,mlstr);
  print("wheel_name = " + wheel_name);
  args="install -U " + server + '/' + wheel_name;
  cmd='pip ' + args;
  results = subprocess.run([cmd],shell=True);
  if results.returncode == 0:
    flag_cont=False;flag_installed=True;
  I=I+1;
  if I>=nn:
    flag_cont=False;

# if installed peform a quick test (or report further installation steps needed) 
if flag_installed:
  print("-"*80);
  print("Test the package works:");
  print("-"*80);
  import mlmod_lammps.lammps as lammps; L = lammps.lammps(); L.command("info system");
  print("."*80);
  print("Looks like the package succeeded in installing.");
  print("To set up models and simulations, please see the examples and docs.");
  print("."*80);
else:
  print("."*80);
  print("Did not succeed in quick installing the package.");
  print("Please read the docs further to install.");
  print("."*80);

