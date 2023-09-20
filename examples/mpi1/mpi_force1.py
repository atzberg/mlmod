#!/home/atzberg/anaconda3/envs/mlmod-lammps/bin/python -i

# preceding line should have path for Python on your machine

# simple.py
# Purpose: mimic operation of examples/COUPLE/simple/simple.cpp via Python

# Serial syntax: simple.py in.lammps
#                in.lammps = LAMMPS input script

# Parallel syntax: mpirun -np 4 simple.py in.lammps
#                  in.lammps = LAMMPS input script
# also need to uncomment mpi4py sections below

from __future__ import print_function
import os,sys,pickle;
import ctypes

script_base_name = "mpi_force1"; script_dir = os.getcwd();

def wrap_print0(ss):
  print(ss);

def wrap_print1(fid,ss):
  fid.write(ss + '\n');
  print(ss);

# parse command line

#argv = sys.argv
#if len(argv) != 2:
#  print("Usage: python %s.py in.lammps"%script_base_name)
#  sys.exit()

#infile = sys.argv[1]
infile='in.force1';

me = 0

# uncomment this if running in parallel via mpi4py
from mpi4py import MPI
me = MPI.COMM_WORLD.Get_rank()
nprocs = MPI.COMM_WORLD.Get_size()

#from lammps import lammps
# import the mlmod_lammps module
from mlmod_lammps.lammps import lammps; 
import mlmod_lammps.lammps.constants as lconst
import mlmod_lammps.util as m_util;

flag_wrap_print = False;
if flag_wrap_print:
  proc_filename = "%s__p_%.3d.out"%(script_base_name,me); 
  fid_proc = open(proc_filename,'w');
  printw = lambda ss: wrap_print1(fid_proc,ss);
  printw("proc_filename = " + proc_filename);
else:
  printw = lambda ss: wrap_print0(ss);

# wait for me == 0 to finish setup before other processes proceed
#@@@

force_case = 'force1';
base_dir_output = '%s/output/%s'%(script_dir,script_base_name);
dir_run_name = 'batch_00';
base_dir = '%s/%s_test001'%(base_dir_output,dir_run_name);
base_dir_fig = '%s/fig'%base_dir;
base_dir_vtk = '%s/vtk'%base_dir;

# remove all data from dir
# setup the directories
if me == 0: 
  m_util.create_dir(base_dir_output);
  m_util.rm_dir(base_dir);
  m_util.create_dir(base_dir_fig);
  m_util.create_dir(base_dir_vtk);

  ## copy the model files to the destination
  template_mlmod_model = 'mlmod_model1';
  src = script_dir + '/' + template_mlmod_model;
  dst = base_dir + '/';
  m_util.copytree2(src,dst,symlinks=False,ignore=None);

MPI.COMM_WORLD.Barrier();  # all processes must reach here, before continuing 

printw("="*80);
printw("rank: me = " + str(me) + ' of ' + str(nprocs) + ' processors');

## change directory for running LAMMPS in output
printw("For running LAMMPS changing the current working directory to:\n%s"%base_dir);
os.chdir(base_dir); # base the current working directory

# create a run lammps
lmp = lammps()

# run infile one line at a time
lines = open(infile,'r').readlines()
for line in lines: lmp.command(line)

printw("."*80);

# setup mlmod force 
flag_force=True;
#force_type='F_ML1';
force_type='F_X_ML1';
#force_type='F_Pair_ML1';
print("force_type = " + force_type);
if flag_force: 
  f_basename = 'F_' + force_case + '_' + force_type;
  filename = f_basename + '_params.pickle';
  model_params = pickle.load(open(filename,'rb'));   
  mlmod_params = {'model_type':force_type,
		  'model_data':{
		    'base_name':force_case,
		    'base_dir':'./' + f_basename,
		    'mask_fix':'POST_FORCE',
		    'mask_input':model_params['mask_input'],  # ensure aligns with the model
		    'F_filename':f_basename + '.pt',
		    }
		 };
  filename_mlmod_params = force_type + '_' + force_case + '.mlmod_params';
  if me == 0:  # just write file for rank == 0 process (assumes shared filesystem)
    m_util.write_mlmod_params(filename_mlmod_params,mlmod_params);
  #lmp.command("fix F_ML1_" + force_case " all mlmod " + filename_mlmod_params);
  lmp.command("fix %s_1 all mlmod"%force_type + " " + filename_mlmod_params + " verbose 2");

lmp.command("variable dumpfreq equal 1");
lmp.command("dump dvtk_mlmod1 all vtk ${dumpfreq} ./vtk/Particles_mlmod_*.vtp fx fy fz id type vx vy vz");
lmp.command("dump_modify dvtk_mlmod1 pad 8"); # ensures filenames file_000000.data
lmp.command("dump_modify dvtk_mlmod1 sort id");

printw("."*80);
# run 10 more steps
# get coords from LAMMPS
# change coords of 1st atom
# put coords back into LAMMPS
# run a single step with changed coords

lmp.command("run 300")

# uncomment if running in parallel via mpi4py
printw("Proc %d out of %d procs has"%(me,nprocs))
printw("Proc %d: Done."%me);

printw("="*80);

if flag_wrap_print:
  fid_proc.close();

