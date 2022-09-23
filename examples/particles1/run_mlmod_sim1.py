#!/usr/bin/env python
# coding: utf-8

# #### Creates a PyTorch model for testing the MLMOD-LAMMPS simulation codes
# Paul J. Atzberger 
# http://atzberger.org
#
# Shows how to run a simulation in LAMMPS using the PyTorch models.
#

import os,sys;
import shutil;
import numpy as np;
import pickle;
script_base_name = "run_mlmod_sim1";
script_dir = os.getcwd();

# import the mlmod_lammps module
from mlmod_lammps.lammps import IPyLammps # use this for the pip install of pre-built package
lammps_import_comment = "from mlmod_lammps.lammps import IPyLammps";  

# filesystem management
def create_dir(dir_name):
  if not os.path.exists(dir_name):
    os.makedirs(dir_name);    
    
def rm_dir(dir_name):
  if os.path.exists(dir_name):    
    shutil.rmtree(dir_name);
  else: 
    print("WARNING: rm_dir(): The directory does not exist, dir_name = " + dir_name);    

def copytree2(src, dst, symlinks=False, ignore=None):
  for ff in os.listdir(src):
    s = os.path.join(src, ff); d = os.path.join(dst, ff);
    if os.path.isdir(s):
      shutil.copytree(s, d, symlinks, ignore);
    else:
      shutil.copy2(s, d);

# test the package
print("Testing library loaded...");
flag=True;
if flag:
  from mlmod_lammps.tests import t1; t1.test()

# @base_dir
base_dir_output   = '%s/output/%s'%(script_dir,script_base_name);
create_dir(base_dir_output);

dir_run_name = 'batch_00';
base_dir = '%s/%s_test001'%(base_dir_output,dir_run_name);

# remove all data from dir
rm_dir(base_dir);

# setup the directories
base_dir_fig    = '%s/fig'%base_dir;
create_dir(base_dir_fig);

base_dir_vtk    = '%s/vtk'%base_dir;
create_dir(base_dir_vtk);

# setup logging

# print the import comment
print(lammps_import_comment);

# copy the model files to the destination
src = script_dir + '/' + "mlmod_model1";
dst = base_dir + '/';
copytree2(src,dst,symlinks=False,ignore=None);

# change directory for running LAMMPS in output
print("For running LAMMPS changing the current working directory to:\n%s"%base_dir);
os.chdir(base_dir); # base the current working directory

# ### Setup LAMMPs
L = IPyLammps();

# ### Perform the simulation (using script in file below)
#
# read the LAMMPS command file
LAMMPS_script_filename = 'Model.LAMMPS_script';
print("Running script, LAMMPS_script_filename = " + LAMMPS_script_filename);
print(80*"=");
print("LAMMPS_script = " + LAMMPS_script_filename);
print(80*"-");
f = open(LAMMPS_script_filename,'r');
file_lines = f.readlines();
for line in file_lines:
  sys.stdout.write(line);
f.close();
print(80*"=");  

# run by feeding commands to LAMMPs one line at a time
print("Sending commands to LAMMPs");
f = open(LAMMPS_script_filename,'r');
file_lines = f.readlines();
for line in file_lines:
  sys.stdout.write(line);
  L.command(line);

print("Done");

