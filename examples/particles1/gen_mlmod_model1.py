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
import os
script_base_name = "gen_mlmod_model1";
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

# define the model
class M_ij_Model(torch.nn.Module):
  def __init__(self):
    super(M_ij_Model, self).__init__()

  # evaluate
  def forward(self, x):
    num_dim = 3;
    
    vec_r_ij = x[0,0:num_dim] - x[0,num_dim:2*num_dim]; # avoiding tensorizing batch, just one pair assumed
    r_ij_sq = torch.sum(torch.pow(vec_r_ij,2),0);
    r_ij = torch.sqrt(r_ij_sq);

    #epsilon = 6.0/8.0; eta = 1.0;  # can adjust below for coupling strength    
    epsilon = 0.01*(6.0/8.0)*1.01; eta = 1.0;
    eps1 = 1e-12;
    prefactor = 1.0/(8.0*np.pi*eta*(r_ij + epsilon));
    M_ij = prefactor*(torch.eye(num_dim) + (torch.ger(vec_r_ij,vec_r_ij)/(r_ij_sq + eps1)));

    z = M_ij.flatten().unsqueeze(0);

    #pdb.set_trace();
    return z;

# define the model
class M_ii_Model(torch.nn.Module):
  def __init__(self):
    super(M_ii_Model, self).__init__()
        
  # evaluate
  def forward(self, x):
    num_dim = 3;
        
    a = 1.1; epsilon = 6.0/8.0; eta = 1.0;
    M_ii = torch.eye(num_dim)*(1.0/(6*np.pi*eta*a));
                        
    z = M_ii.flatten().unsqueeze(0);
    
    #pdb.set_trace();
    return z;    

# Initialize model

#--
M_ii_model = M_ii_Model();
model = M_ii_model;
x = torch.zeros((1,3));
traced_model = torch.jit.trace(model, (x))

case_label = 'eps_001'
base_dir = './output/%s/%s'%(script_base_name,case_label);

print("base_dir = " + base_dir);
create_dir(base_dir);

torch_filename = '%s/M_ii_oseen1.pt'%base_dir;
print("torch_filename = " + torch_filename);
traced_model.save(torch_filename);

#--
M_ij_model = M_ij_Model();
model = M_ij_model;
x = torch.zeros((1,6));
traced_model = torch.jit.trace(model, (x))

torch_filename = '%s/M_ij_oseen1.pt'%base_dir;
print("torch_filename = " + torch_filename);
traced_model.save(torch_filename);

cmd = 'cp %s.ipynb %s/archive_%s.py'%(script_base_name,base_dir,script_base_name);
print("cmd = " + cmd);
os.system(cmd);

# #### Show model outputs
print("M_ii_model");
x = torch.zeros((1,3));
x[0] = 1.0;
y = M_ii_model(x);
yy = y[0,:].reshape((3,3));
print("x = " + str(x));
print("y = " + str(y));
print("M_ii = " + str(yy));

print("M_ij_model");
x = torch.zeros((1,6));
x[0,3] = 1.0;
y = M_ij_model(x)

yy = y[0,:].reshape((3,3));
print("x = " + str(x));
print("y = " + str(y));
print("M_ij = " + str(yy));

print("Done");




