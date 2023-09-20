#!/usr/bin/env python
# coding: utf-8


# #### Creates a PyTorch model for testing the MLMOD-LAMMPS simulation codes
# Paul J. Atzberger 
# http://atzberger.org
#     
# Generates a hand-crafted model in PyTorch framework to show basic ideas. 
#
# This gives a stub for how to use the package and run simulation with 
# them in LAMMPS
#
# These models can be replaced readily by data-driven / learned models or 
# any other PyTorch model that can be traced and output to .pt torch-format.
#
import os, pickle;
script_base_name = "gen_mlmod_oseen1";
script_dir = os.getcwd();

import torch
import shutil, pickle;
import numpy as np;
import sys,shutil,pickle;

import logging;

def create_dir(dir_name):
  if not os.path.exists(dir_name):
    os.makedirs(dir_name);    

# define the model
class M_ij_Model(torch.nn.Module):
  def __init__(self):
    super(M_ij_Model, self).__init__()
    self.a = 5.1;
    self.epsilon = (6.0/8.0)*self.a; self.eta = 1.0;
    self.eps1 = 1e-12;

  # evaluate
  def forward(self, x):
    num_dim = 3;
    
    vec_r_ij = x[0:num_dim,0] - x[num_dim:2*num_dim,0]; # avoiding tensorizing batch, just one pair assumed
    r_ij_sq = torch.sum(torch.pow(vec_r_ij,2),0);
    r_ij = torch.sqrt(r_ij_sq);

    #epsilon = 6.0/8.0; eta = 1.0;  # can adjust below for coupling strength    
    epsilon = self.epsilon; eta = self.eta; eps1 = self.eps1;
    prefactor = 1.0/(8.0*torch.pi*eta*(r_ij + epsilon));
    M_ij = prefactor*(torch.eye(num_dim) + (torch.outer(vec_r_ij,vec_r_ij)/(r_ij_sq + eps1)));

    z = M_ij.flatten().unsqueeze(1);  #shape=[n,1]

    #pdb.set_trace();
    return z;

# define the model
class M_ii_Model(torch.nn.Module):
  def __init__(self):
    super(M_ii_Model, self).__init__()
    self.a = 5.1; self.eta = 1.0;
        
  # evaluate
  def forward(self, x):
    num_dim = 3;

    a = self.a; eta = self.eta;
    M_ii = torch.eye(num_dim)*(1.0/(6.0*torch.pi*eta*a));
                        
    z = M_ii.flatten().unsqueeze(1);
    
    #pdb.set_trace();
    return z;    

# Initialize model
M_ii_model = M_ii_Model();
M_ij_model = M_ij_Model();

num_dim = 3;

a = 4.1; eta = 1.2; epsilon = (6.0/8.0)*a; eps1 = 1e-11; 
M_ii_model.a = a; M_ii_model.eta = eta;
M_ij_model.a = a; M_ij_model.eta = eta; 
M_ii_model.epsilon = epsilon; M_ij_model.eps1 = eps1; 

#-- save model data for later validation
case_label = 'gen_001'
base_dir = './output/%s/%s'%(script_base_name,case_label);

print("base_dir = " + base_dir);
create_dir(base_dir);

# WARNING: currently hard-coded
model_name='oseen1';
f = open('%s/M_ij_oseen1_params.pickle'%base_dir,'wb');
params_M_ij = {
'model_type':'dX_MF_ML1',
'model_name':model_name,
'epsilon':M_ij_model.epsilon,
'eta':M_ij_model.eta,
'eps1':M_ij_model.eps1
};
pickle.dump(params_M_ij,f);
f.close();

f = open('%s/M_ii_oseen1_params.pickle'%base_dir,'wb');
params_M_ii = {
'model_type':'dX_MF_ML1',
'model_name':model_name,
'a':M_ii_model.a,
'eta':M_ii_model.eta 
};
pickle.dump(params_M_ii,f);
f.close();

#--
model = M_ii_model;
x = torch.zeros((num_dim,1));
traced_M_ii_model = torch.jit.trace(model, (x))

torch_filename = '%s/M_ii_oseen1.pt'%base_dir;
print("torch_filename = " + torch_filename);
traced_M_ii_model.save(torch_filename);
print(traced_M_ii_model.code) # prints trace model code

#--
model = M_ij_model;
x = torch.zeros((2*num_dim,1));
traced_M_ij_model = torch.jit.trace(model, (x))

torch_filename = '%s/M_ij_oseen1.pt'%base_dir;
print("torch_filename = " + torch_filename);
traced_M_ij_model.save(torch_filename);
print(traced_M_ij_model.code) # prints trace model code

cmd = 'cp %s.py %s/archive_%s.py'%(script_base_name,base_dir,script_base_name);
print("cmd = " + cmd);
os.system(cmd);

# #### Show model outputs
x = torch.zeros((2*num_dim,1));
x[0,0] = -5.0;
x[3,0] = 5.0;

print("M_ii_model");
xx = torch.zeros((num_dim,1));
xx[:,0] = x[0:num_dim,0];
y = M_ii_model(xx);
yy = y[:,0].reshape((3,3));
M_ii = yy;
print("x.shape = "+ str(x.shape));
print("x = " + str(x));
print("y.shape = "+ str(y.shape));
print("y = " + str(y));
print("M_ii = " + str(M_ii));

print("M_ij_model");
xx = torch.zeros((2*num_dim,1));
xx[0:3,0] = x[0:3,0];
xx[3:6,0] = x[3:6,0];
y = M_ij_model(xx)
yy = y[:,0].reshape((3,3));
M_ij = yy;
print("x.shape = "+ str(x.shape));
print("x = " + str(x));
print("y.shape = "+ str(y.shape));
print("y = " + str(y));
print("M_ij = " + str(M_ij));

f = open('%s/test_data_oseen1.pickle'%base_dir,'wb');
ss = {'M_ii':M_ii.cpu().numpy(),
      'M_ij':M_ij.cpu().numpy(),
      'x':x.cpu().numpy()};
pickle.dump(ss,f);
f.close();

print("The generated models are located in " + base_dir);

print("Done");




