#!/usr/bin/env python
### Particle simulation with mlmod-based mobility.

import os,sys, shutil, pickle;
import numpy as np;

script_base_name = "run_sim_particles1"; script_dir = os.getcwd();

# import the mlmod_lammps module
from mlmod_lammps.lammps import lammps; 
import mlmod_lammps.lammps.constants as lconst
import mlmod_lammps.util as m_util;
lammps_import_comment = "from mlmod_lammps.lammps import lammps";  

#@model_case
model_case = 'rpy1';
#model_case = 'oseen1';

# @base_dir
base_dir_output   = '%s/output/%s'%(script_dir,script_base_name);
m_util.create_dir(base_dir_output);

dir_run_name = 'batch_00';
base_dir = '%s/%s_test001'%(base_dir_output,dir_run_name);

# remove all data from dir
m_util.rm_dir(base_dir);

# setup the directories
base_dir_fig    = '%s/fig'%base_dir;
m_util.create_dir(base_dir_fig);

base_dir_vtk    = '%s/vtk'%base_dir;
m_util.create_dir(base_dir_vtk);

## print the import comment
print(lammps_import_comment);

## copy the model files to the destination
template_mlmod_model = 'mlmod_model1';
src = script_dir + '/' + template_mlmod_model;
dst = base_dir + '/';
m_util.copytree2(src,dst,symlinks=False,ignore=None);
#
## change directory for running LAMMPS in output
print("For running LAMMPS changing the current working directory to:\n%s"%base_dir);
os.chdir(base_dir); # base the current working directory

# #### Setup LAMMPs
L = lammps(); Lc = m_util.wrap_L(L,m_util.Lc_print);

Lc("log log.lammps");
Lc("variable dumpfreq equal 1");
Lc("variable restart equal 0");
# distance for bins beyond force cut-off (1.0 = 1.0 Ang for units = real)
Lc("variable neighborSkinDist equal 1.0"); 
Lc("variable baseFilename universe Model");

Lc("units nano");
Lc("atom_style angle");
Lc("bond_style none");
Lc("angle_style none");

# bounding box set up
LL = 36; dims=(-LL/2.0,LL/2.0,-LL/2.0,LL/2.0,-LL/2.0,LL/2.0);
Lc("region mybox prism %.2e %.2e %.2e %.2e %.2e %.2e 0 0 0"%dims); 
Lc("boundary p p p");
Lc("create_box 1 mybox");
Lc("lattice none 1.0");  # NOTE: Very important both create_box and lattice called before create_atoms
#Lc("lattice fcc 1.0"); 

# atom set up
nn = num_per_dir = 20; LLL = 0.8*LL;
#num_dim = 3; num_atoms = 2*nn*nn;
num_dim = 3; num_atoms = nn*nn;
xx = np.zeros((num_atoms,num_dim)); 
m_xx = np.linspace(-LLL/2.0,LLL/2.0,nn);
m_yy = np.linspace(-LLL/2.0,LLL/2.0,nn);
m_zz = np.linspace(-LLL/2.0,LLL/2.0,nn);

II0 = 0;
#m_x,m_y = np.meshgrid(m_xx,m_yy,indexing='ij');
#II = np.arange(0,nn*nn);
#xx[II,0] = m_x.flatten();
#xx[II,1] = m_y.flatten();
#xx[II,2] = 0*m_x.flatten();
#II0=nn*nn;

m_x,m_z = np.meshgrid(m_xx,m_zz,indexing='ij');
II = np.arange(II0,II0+nn*nn);
xx[II,0] = m_x.flatten();
xx[II,1] = 0*m_x.flatten();
xx[II,2] = m_z.flatten();

Lc("mass 1 1.1230000");

# WARNING: Need to make sure both create_box and lattice called before this command.
#Lc("create_atoms 1 random 25 1234 mybox remap yes");
Lc("create_atoms 1 random %d 1234 mybox remap yes"%num_atoms);

# each numpy array gives direct access to lammps memory
atom_x = L.numpy.extract_atom("x");
atom_v = L.numpy.extract_atom("v"); 
atom_id = L.numpy.extract_atom("id"); 
atom_type = L.numpy.extract_atom("type"); 

atom_x[:] = xx[:]; 
atom_v[:] = 0*atom_x[:]; 
#atom_id = np.array([1,2],dtype=np.int64);
#atom_type = np.array([1,1],dtype=np.int64);

print("num_atoms = " + str(atom_x.shape[0]));
print("atom_type = " + str(atom_type));
print("atom_x = " + str(atom_x));
print("atom_v = " + str(atom_v));
print("atom_id = " + str(atom_id));

Lc("pair_style none");

Lc("atom_modify sort 1000 ${neighborSkinDist}");          # setup sort data explicitly since no interactions to set this data. 
Lc("comm_style tiled");
Lc("comm_modify mode single cutoff 18 vel yes");

Lc("neighbor ${neighborSkinDist} bin");                    # first number gives a distance beyond the force cut-off ${neighborSkinDist}
Lc("neigh_modify every 1");

Lc("atom_modify sort 0 ${neighborSkinDist}");           # setup sort data explicitly since no interactions to set this data. 

mlmod_params = {'model_type':'dX_MF_ML1',
                'model_data':{
                  'M_ii_filename':'M_ii_' + model_case + '.pt',
                  'M_ij_filename':'M_ij_' + model_case + '.pt'
                  }
               };
filename_mlmod_params = 'main.mlmod_params';
m_util.write_mlmod_params(filename_mlmod_params,mlmod_params);
Lc("fix ml_1 all mlmod " + filename_mlmod_params);
timestep = 0.35;
Lc("timestep %f"%timestep);

x0 = np.array([-LL/4.0,0,0]);
ii0 = m_util.find_closest_pt(x0,xx); id1=ii0+1; #lammps ids base 1
Lc("group cforce1 id %d"%id1); # create group using id's
force_f1 = np.array([10.0,0.0,0.0]);
Lc("fix force_f1 cforce1 addforce \
%.4e %.4e %.4e"%(force_f1[0],force_f1[1],force_f1[2])); # add force to the cforce group

x0 = np.array([LL/4.0,0,0]);
ii0 = m_util.find_closest_pt(x0,xx); id2=ii0+1; #lammps ids base 1
Lc("group cforce2 id %d"%id2); # create group using id's
force_f2 = -1.0*force_f1;
Lc("fix force_f2 cforce2 addforce \
%.4e %.4e %.4e"%(force_f2[0],force_f2[1],force_f2[2])); # add force to the cforce group

#fix 1 all langevin 298.15 298.15 0.00001 48279
#fix 2 all nve

Lc("dump dvtk_mlmod1 all vtk ${dumpfreq} ./vtk/Particles_mlmod_*.vtp fx fy fz id type vx vy vz");
Lc("dump_modify dvtk_mlmod1 pad 8"); # ensures filenames file_000000.data
Lc("dump_modify dvtk_mlmod1 sort id");

Lc("run 300");

# gives direct access to memory to lammps
atom_x = L.numpy.extract_atom("x");  
atom_v = L.numpy.extract_atom("v"); 
atom_f = L.numpy.extract_atom("f"); 
atom_id = L.numpy.extract_atom("id"); 
atom_type = L.numpy.extract_atom("type");

print("num_atoms = " + str(atom_x.shape[0]));
print("atom_type = " + str(atom_type));
print("atom_id = " + str(atom_id));
print("atom_x = " + str(atom_x));
print("atom_v = " + str(atom_v));
print("atom_f = " + str(atom_f));

# #### Perform the simulation
print("Done");

