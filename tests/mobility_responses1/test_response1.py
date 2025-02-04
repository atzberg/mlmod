#!/usr/bin/env python

# Shows how to run a simulation in LAMMPS using the PyTorch models.
import os,sys, shutil, pickle, time;
import numpy as np;

try:
  import pytest;
except:  
  os.system('pip install pytest'); # install if not already 
  import pytest;

script_base_name = "test_response1"; script_dir = os.getcwd();

# import the mlmod_lammps module
from mlmod_lammps.lammps import lammps # use this for the pip install of pre-built package
lammps_import_comment = "from mlmod_lammps.lammps import lammps";  

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

def write_mlmod_params(filename,params):
  model_type = params['model_type'];
  model_data = params['model_data'];
  
  # xml file
  f = open(filename,'w');
  f.write('<?xml version="1.0" encoding="UTF-8"?>\n');
  f.write('<MLMOD>\n');
  f.write('\n');
  f.write('<model_data type="' + model_type + '">\n');
  f.write('  <M_ii_filename value="' + model_data['M_ii_filename'] + '"/>\n');
  f.write('  <M_ij_filename value="' + model_data['M_ij_filename'] + '"/>\n');
  f.write('</model_data>\n');
  f.write('\n');
  f.write('</MLMOD>\n');
  f.close();

  # pickle file
  f = open(filename + '.pickle','wb'); pickle.dump(params,f); f.close();


def compute_M_ii_oseen1(x,params):
  dd = params;
  num_dim = 3; a = dd['a']; eta = dd['eta']; 
  M_ii = np.eye(num_dim)*(1.0/(6.0*np.pi*eta*a));

  return M_ii;

def compute_M_ij_oseen1(x,params):
  dd = params;
  epsilon = dd['epsilon']; eta = dd['eta']; eps1 = dd['eps1']; num_dim = 3; 

  vec_r_ij = x[0,0:num_dim] - x[1,0:num_dim]; # avoiding tensorizing batch, just one pair assumed
  r_ij_sq = np.sum(np.power(vec_r_ij,2));
  r_ij = np.sqrt(r_ij_sq);

  prefactor = 1.0/(8.0*np.pi*eta*(r_ij + epsilon));
  M_ij = prefactor*(np.eye(num_dim) + (np.outer(vec_r_ij,vec_r_ij)/(r_ij_sq + eps1)));
  
  return M_ij; 

def compute_M_ii_rpy1(x,params):
  dd = params;
  num_dim = 3; a = dd['a']; eta = dd['eta']; 

  c0 = 1.0/(6.0*np.pi*eta*a);
  M_ii = c0*np.eye(num_dim);

  return M_ii;

def compute_M_ij_rpy1(x,params):
  
  dd = params;
  a = dd['a']; epsilon = dd['epsilon']; eta = dd['eta']; eps1 = dd['eps1']; num_dim = 3; 
    
  vec_r_ij = x[0,0:num_dim] - x[1,0:num_dim]; # avoiding tensorizing batch, just one pair assumed
  r_ij_sq = np.sum(np.power(vec_r_ij,2));
  r_ij = np.sqrt(r_ij_sq);
  r_ij_p = r_ij + eps1;
  r_ij_p_sq = r_ij_sq + 2*r_ij_p*eps1 + eps1*eps1; 
  r_ij_p_cub = r_ij_p_sq*r_ij_p;

  c0 = 1.0/(6.0*np.pi*eta*a);
  c1 = 3.0*a/(4.0*r_ij_p);
  M_ij = c1*(np.eye(num_dim) + (np.outer(vec_r_ij,vec_r_ij)/r_ij_p_sq));
  c2 = a*a*a/(2.0*r_ij_p_cub)
  M_ij = M_ij + c2*(np.eye(num_dim) - (np.outer(vec_r_ij,vec_r_ij)/r_ij_p_sq));
  M_ij = c0*M_ij;

  return M_ij; 

def compute_M_ij(x,params):
  model_name = params['model_name'];
  if model_name == 'oseen1':
    M_ij = compute_M_ij_oseen1(x,params);
  elif model_name == 'rpy1':
    M_ij = compute_M_ij_rpy1(x,params);
  else:
    raise Exception("No recognized, model_name = " + model_name);

  return M_ij;

def compute_M_ii(x,params):
  model_name = params['model_name'];
  if model_name == 'oseen1':
    M_ii = compute_M_ii_oseen1(x,params);
  elif model_name == 'rpy1':
    M_ii = compute_M_ii_rpy1(x,params);
  else:
    raise Exception("No recognized, model_name = " + model_name);

  return M_ii;


def compute_full_response(model_case,v_I):
  print("model_case = " + model_case);

  # @base_dir
  base_dir_output   = '%s/output/%s'%(script_dir,script_base_name);
  create_dir(base_dir_output);

  batch_id = 0;
  dir_run_name = model_case + '_' + '%.2d'%v_I + '_' + 'batch_%.2d'%batch_id;
  base_dir = '%s/%s_test001'%(base_dir_output,dir_run_name);

  # remove all data from dir
  rm_dir(base_dir);

  # setup the directories
  base_dir_fig    = '%s/fig'%base_dir;
  create_dir(base_dir_fig);

  base_dir_vtk    = '%s/vtk'%base_dir;
  create_dir(base_dir_vtk);

  ## print the import comment
  print(lammps_import_comment);

  ## copy the model files to the destination
  src = script_dir + '/' + "template_response1";
  dst = base_dir + '/';
  copytree2(src,dst,symlinks=False,ignore=None);
  #
  ## change directory for running LAMMPS in output
  print("For running LAMMPS changing the current working directory to:\n%s"%base_dir);
  os.chdir(base_dir); # base the current working directory

  # #### Setup LAMMPs
  L = lammps();
  Lc = lambda ss: L.command(ss);  # lammps commands 

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
  Lc("region mybox prism -18 18 -18 18 -18 18 0 0 0"); 
  Lc("boundary p p p");
  Lc("create_box 1 mybox");
  Lc("lattice none 1.0");  # NOTE: Very important both create_box and lattice called before create_atoms
  #Lc("lattice fcc 1.0"); 

  # atom set up
  num_dim = 3; num_atoms = 2;
  xx = np.zeros((num_atoms,num_dim)); xx[:,0] = [-5,5];

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

  print("num_atoms = " + str(atom_x.shape[0]));
  print("atom_type = " + str(atom_type));
  print("atom_x = " + str(atom_x));
  print("atom_v = " + str(atom_v));
  print("atom_id = " + str(atom_id));

  Lc("pair_style none");

  Lc("atom_modify sort 1000 ${neighborSkinDist}");          # setup sort data explicitly since no interactions to set this data. 
  Lc("comm_style tiled");
  Lc("comm_modify mode single cutoff 18.0 vel yes");

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
  write_mlmod_params(filename_mlmod_params,mlmod_params);
  Lc("fix ml_1 all mlmod " + filename_mlmod_params);
  timestep = 0.35;
  Lc("timestep %f"%timestep);

  Lc("group cforce1 id 1"); # create group using id's
  force_f1 = np.array([10.0,0.0,0.0]);
  Lc("fix force_f1 cforce1 addforce \
  %.4e %.4e %.4e"%(force_f1[0],force_f1[1],force_f1[2])); # add force to the cforce group

  Lc("group cforce2 id 2"); # create group using id's
  force_f2 = np.array([0.0,0.0,0.0]);
  Lc("fix force_f2 cforce2 addforce \
  %.4e %.4e %.4e"%(force_f2[0],force_f2[1],force_f2[2])); # add force to the cforce group

  Lc("dump dmp_vtk_mlmod all vtk ${dumpfreq} ./vtk/Particles_mlmod_*.vtp id type vx fx");
  Lc("dump_modify dmp_vtk_mlmod pad 8"); # ensures filenames file_000000.data
  Lc("dump_modify dmp_vtk_mlmod sort id");

  Lc("run 0")

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

  # collect some trajectory data
  atom_xx = []; atom_ff = []; atom_vv = [];
  numsteps=3;
  for i in range(0,numsteps):
    Lc("run 1");
    #Lc("set atom 1 x -5.0 y 0.0 z 0.0");
    #Lc("set atom 2 x 5.0 y 0.0 z 0.0");

    atom_xx.append(np.array(atom_x));
    atom_vv.append(np.array(atom_v));
    atom_ff.append(np.array(atom_f));

  atom_xx = np.array(atom_xx);
  atom_vv = np.array(atom_vv);
  atom_ff = np.array(atom_ff);

  print("atom_xx = " + str(atom_xx));
  print("atom_vv = " + str(atom_vv));
  print("atom_ff = " + str(atom_ff));

  # now validate the results 
  gen_data_dir = base_dir;  # template copied this data to output
  filename = gen_data_dir + '/' + 'M_ij_' + model_case + '_params.pickle';
  f = open(filename,'rb'); M_ij_model_params = pickle.load(f); f.close();
  filename = gen_data_dir + '/' + 'M_ii_' + model_case + '_params.pickle';
  f = open(filename,'rb'); M_ii_model_params = pickle.load(f); f.close();

  timeindex = 1;
  x          = atom_xx[timeindex-1,:,:];
  print("");
  print("x = " + str(x));

  model_type = M_ij_model_params['model_type'];
  if model_type == 'dX_MF_ML1':
    M_ij = compute_M_ij(x,M_ij_model_params);
    v_predict1_ij = M_ij.dot(force_f2); v_predict2_ij = M_ij.dot(force_f1);
    print("");
    print("M_ij = " + str(M_ij));
    print("");
    print("v1_ij = " + str(v_predict1_ij));
    print("v2_ij = " + str(v_predict2_ij));

  model_type = M_ii_model_params['model_type'];
  if model_type == 'dX_MF_ML1':
    M_ii = compute_M_ii(x,M_ii_model_params);
    v_predict1_ii = M_ii.dot(force_f1); v_predict2_ii = M_ii.dot(force_f2);
    print("");
    print("M_ii = " + str(M_ii));
    print("");
    print("atom_vv[1,0,:] = " + str(atom_vv[1,0,:]));
    print("v1_ii = " + str(v_predict1_ii));
    print("");
    print("atom_vv[1,1,:] = " + str(atom_vv[1,0,:]));
    print("v2_ii = " + str(v_predict2_ii));

  # total velocity for each particle 
  v_predict1 = v_predict1_ii + v_predict1_ij;
  v_predict2 = v_predict2_ii + v_predict2_ij;
  print("");
  print("atom_vv[%d,0,:] = "%timeindex + str(atom_vv[timeindex,0,:]));
  print("v_predict1 = " + str(v_predict1));
  print("");
  print("atom_vv[%d,1,:] = "%timeindex + str(atom_vv[timeindex,1,:]));
  print("v_predict2 = " + str(v_predict2));

  results = {'atom_vv':atom_vv,
             'v_predict1':v_predict1,
             'v_predict2':v_predict2};

  return results;

# perform the tests
@pytest.mark.parametrize("response_params, threshold_result", [
    ({'model_case':'oseen1','v_I':1}, 1e-2),
    ({'model_case':'oseen1','v_I':2}, 1e-2),
    ({'model_case':'rpy1','v_I':1}, 1e-2),
    ({'model_case':'rpy1','v_I':2}, 1e-2)
])

def test_full_response(response_params,threshold_result):
 
  # sleep to let LAMMPS output to catch up
  # (or can print out-of-place)
  time.sleep(0.1);
 
  # get case testing
  model_case = response_params['model_case'];
  v_I = response_params['v_I'];

  # compute the results
  results = compute_full_response(model_case,v_I);
 
  # test if the response is correct
  atom_vv = results['atom_vv'];
  v_predict1 = results['v_predict1'];
  v_predict2 = results['v_predict2'];

  err_abs_v1 = np.sum(np.abs(atom_vv[1,0,:] - v_predict1));
  err_abs_v2 = np.sum(np.abs(atom_vv[1,1,:] - v_predict2));
  print("");
  print("err_abs_v1 = %.2e"%err_abs_v1);
  print("err_abs_v2 = %.2e"%err_abs_v2);

  # Construct final error measure
  # use relative error, unless target
  # value is close to zero.   
  epsilon = 1e-11;
  norm_v_predict1 = np.linalg.norm(v_predict1);
  norm_v_predict2 = np.linalg.norm(v_predict2);
  if (norm_v_predict1 > epsilon):
    err_v1 = err_abs_v1/norm_v_predict1;
  else: 
    err_v1 = err_abs_v1;

  if (norm_v_predict2 > epsilon):
    err_v2 = err_abs_v2/norm_v_predict2;
  else: 
    err_v2 = err_abs_v2;

  # test the predictions
  threshold = threshold_result;

  if v_I == 1:
    msg_fail = "Threshold failed.\n"
    msg_fail += "model_case = %s \n"%model_case;
    msg_fail += "v_I = %d \n"%v_I;
    msg_fail += "err_v1 = %.4e"%err_v1;
    
    assert err_v1 < threshold, msg_fail;

  elif v_I == 2:  
    msg_fail = "Threshold failed.\n"
    msg_fail += "model_case = %s \n"%model_case;
    msg_fail += "v_I = %d \n"%v_I;
    msg_fail += "err_v2 = %.4e"%err_v2;
    
    assert err_v2 < threshold, msg_fail;

  else: 
    Exception("Invalid case for v_I = %d"%v_I);


