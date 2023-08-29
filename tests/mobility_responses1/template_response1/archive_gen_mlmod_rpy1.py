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
script_base_name = "gen_mlmod_rpy1";
script_dir = os.getcwd();

import torch
import shutil, pickle;
import numpy as np;
import matplotlib;
import matplotlib.pyplot as plt;
import sys,shutil,pickle,pdb;

import logging;

def create_dir(dir_name):
  if not os.path.exists(dir_name):
    os.makedirs(dir_name);    

# define the models (RPY)
# RPY Tensor
# """ M_ij = \frac{1}{6\pi\eta{a}}\left[
# \frac{3a}{4r}\left(\mathcal{I} + \frac{rr^}{r}\right) + 
# \frac{a^3}{2r^3}\left(\mathcal{I} - \frac{rr^}{r}\right)
# \right]
#
# M_ii = \frac{1}{6\pi\eta{a}}\mathcal{I}
#
class M_ij_Model(torch.nn.Module):
  def __init__(self):
    super(M_ij_Model, self).__init__()
    self.a = 5.1;
    self.epsilon = (6.0/8.0)*self.a; self.eta = 1.0;
    self.eps1 = 1e-12;

  # evaluate
  def forward(self, x):
    num_dim = 3;

    #epsilon = 6.0/8.0; eta = 1.0;  # can adjust below for coupling strength    
    a = self.a; epsilon = self.epsilon; eta = self.eta; eps1 = self.eps1;
    
    vec_r_ij = x[0,0:num_dim] - x[0,num_dim:2*num_dim]; # avoiding tensorizing batch, just one pair assumed
    r_ij_sq = torch.sum(torch.pow(vec_r_ij,2),0);
    r_ij = torch.sqrt(r_ij_sq);
    r_ij_p = r_ij + eps1;
    r_ij_p_sq = r_ij_sq + 2*r_ij_p*eps1 + eps1*eps1; 
    r_ij_p_cub = r_ij_p_sq*r_ij_p;

    vec_r_ij_unit = vec_r_ij/r_ij_p;

    c0 = 1.0/(6.0*torch.pi*eta*a);
    c1 = 3.0*a/(4.0*r_ij_p);
    M_ij = c1*(torch.eye(num_dim) + (torch.outer(vec_r_ij_unit,vec_r_ij_unit)));
    c2 = a*a*a/(2.0*r_ij_p_cub)
    M_ij = M_ij + c2*(torch.eye(num_dim) - 3.0*torch.outer(vec_r_ij_unit,vec_r_ij_unit));
    M_ij = c0*M_ij;

    z = M_ij.flatten().unsqueeze(0);

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
    c0 = 1.0/(6.0*torch.pi*eta*a);
    M_ii = c0*torch.eye(num_dim);
                        
    z = M_ii.flatten().unsqueeze(0);
    
    #pdb.set_trace();
    return z;    

# Initialize model
M_ii_model = M_ii_Model();
M_ij_model = M_ij_Model();

a = 1.1; eta = 1.2; epsilon = (6.0/8.0)*a; eps1 = 1e-11; 
M_ii_model.a = a; M_ii_model.eta = eta;
M_ij_model.a = a; M_ij_model.eta = eta; 
M_ii_model.epsilon = epsilon; M_ij_model.eps1 = eps1; 

#-- save model data for later validation
case_label = 'gen_001'
base_dir = './output/%s/%s'%(script_base_name,case_label);

print("base_dir = " + base_dir);
create_dir(base_dir);

# WARNING: currently hard-coded
model_name='rpy1';
f = open('%s/M_ij_rpy1_params.pickle'%base_dir,'wb');
params_M_ij = {
'model_type':'dX_MF_ML1',
'model_name':model_name,
'a':M_ij_model.a,
'epsilon':M_ij_model.epsilon,
'eta':M_ij_model.eta,
'eps1':M_ij_model.eps1
};
pickle.dump(params_M_ij,f);
f.close();

f = open('%s/M_ii_rpy1_params.pickle'%base_dir,'wb');
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
x = torch.zeros((1,3));
traced_M_ii_model = torch.jit.trace(model, (x))

torch_filename = '%s/M_ii_rpy1.pt'%base_dir;
print("torch_filename = " + torch_filename);
traced_M_ii_model.save(torch_filename);
print(traced_M_ii_model.code) # prints trace model code

#--
model = M_ij_model;
x = torch.zeros((1,6));
traced_M_ij_model = torch.jit.trace(model, (x))

torch_filename = '%s/M_ij_rpy1.pt'%base_dir;
print("torch_filename = " + torch_filename);
traced_M_ij_model.save(torch_filename);
print(traced_M_ij_model.code) # prints trace model code

cmd = 'cp %s.py %s/archive_%s.py'%(script_base_name,base_dir,script_base_name);
print("cmd = " + cmd);
os.system(cmd);

# #### Show model outputs
x = torch.zeros((1,6));
x[0,0] = -5.0;
x[0,3] = 5.0;

print("M_ii_model");
xx = torch.zeros((1,3));
xx[:] = x[0,0:3];
y = M_ii_model(xx);
yy = y[0,:].reshape((3,3));
M_ii = yy;
print("x = " + str(x));
print("y = " + str(y));
print("M_ii = " + str(M_ii));

print("M_ij_model");
xx = torch.zeros((1,6));
xx[0,0:3] = x[0,0:3];
xx[0,3:6] = x[0,3:6];
y = M_ij_model(xx)
yy = y[0,:].reshape((3,3));
M_ij = yy;
print("x = " + str(x));
print("y = " + str(y));
print("M_ij = " + str(M_ij));

f = open('%s/test_data_rpy1.pickle'%base_dir,'wb');
ss = {'M_ii':M_ii.cpu().numpy(),
      'M_ij':M_ij.cpu().numpy(),
      'x':x.cpu().numpy()};
pickle.dump(ss,f);
f.close();

print("The generated models are located in " + base_dir);

print("Done");




