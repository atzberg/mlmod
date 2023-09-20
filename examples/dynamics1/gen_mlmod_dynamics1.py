#!/usr/bin/env python
# coding: utf-8

# #### Creates a PyTorch model for testing the MLMOD-LAMMPS simulation codes
# Paul J. Atzberger 
# http://atzberger.org
#     
# Generates model in PyTorch framework to show basic ideas. 
#
# These models can be replaced readily by almost any PyTorch model that 
# can be traced and output to .pt torch-format (neural networks, gpr, etc...).
#
import os, pickle;
script_base_name = "gen_mlmod_dynamics1";
script_dir = os.getcwd();

import torch
import shutil, pickle;
import numpy as np;
import sys,shutil,pickle;

import logging;

def create_dir(dir_name):
  if not os.path.exists(dir_name):
    os.makedirs(dir_name);    

# define dynamics model
#
class ML_Dynamics1_Model(torch.nn.Module):
  def __init__(self,**params):
    super(ML_Dynamics1_Model, self).__init__()

    # tries to get or sets variable to None
    self.a, self.num_dim = tuple(map(params.get,
                           ['a','num_dim']));

    if self.num_dim is None:
      self.num_dim = 3;

    self.mask_input,self.num_atoms = tuple(map(params.get,
                           ['mask_input','num_atoms']));

    self.list_mask = list_mask = self.mask_input.split();
    self.num_inputs = num_inputs = len(list_mask);

    max_num_inputs = 4;
    mask_input_flag = [False]*max_num_inputs;
       
    # detect which flags are set (organized in order) 
    for i in range(0,self.num_inputs):
      if list_mask[i] == "X":
        mask_input_flag[0] = True;
      elif list_mask[i] == "V":
        mask_input_flag[1] = True;
      elif list_mask[i] == "F":
        mask_input_flag[2] = True;
      elif list_mask[i] == "Type":
        mask_input_flag[3] = True;
      else:
        print("No recognized, list_mask[%d] = %s"%(i,self.list_mask));
    
    self.mask_input_flag = mask_input_flag; 

    # get dimensions of each segment of the data
    # and set index ranges  
    i_dim = []; num_dim = self.num_dim; I0 = 0;     
    if mask_input_flag[0]:  # X
      i_dim.append(num_dim);

    if mask_input_flag[1]:  # V
      i_dim.append(num_dim);

    if mask_input_flag[2]:  # F
      i_dim.append(num_dim);

    if mask_input_flag[3]:  # Type
      i_dim.append(int(1));

    II1 = []; II2 = []; I = 0; # setup ranges
    for k in np.arange(0,num_inputs):    
      d = i_dim[k];
      II1.append(I); I += num_atoms*d; II2.append(I);

    II1 = np.array(II1,dtype=int); II2 = np.array(II2,dtype=int);

    self.i_dim = i_dim; self.II1 = II1; self.II2 = II2;

    nn = 0; ss = [];
    for i in range(0,self.num_inputs):
      nn = nn + i_dim[i]*num_atoms;
      ss.append(i_dim[i]*num_atoms);
    self.nn = nn; self.ss = ss;
 
  # evaluate using F(X,V,F,Type)
  # note, strange bugs can occur in libtorch, if wrong tensor size 
  # i.e. t.shape=[1,n] instead of t.shape[n,1].
  def forward(self, z):
    num_dim = 3;

    a = self.a; num_inputs = self.num_inputs;
    nn = self.nn; ss = self.ss;
    II1 = self.II1; II2 = self.II2;

    #aa = torch.split(z,ss,dim=0);    
    xx = z[II1[0]:II2[0],:];
    x = xx.reshape((num_atoms,num_dim));

    vv = z[II1[1]:II2[1],:];
    v = vv.reshape((num_atoms,num_dim));

    ff = z[II1[2]:II2[2],:];
    f = ff.reshape((num_atoms,num_dim));

    aatype = z[II1[3]:II2[3],:];
    atype = aatype.reshape((num_atoms,1));

    # for testing 
    #out_x = x + v + f + atype*torch.ones((1,num_dim));
    out_x = 0.999*x;
    out_x = out_x.reshape((num_atoms*num_dim,1));  

    out_v = 0.998*v;
    out_v = out_v.reshape((num_atoms*num_dim,1));  
   
    out = torch.vstack((out_x,out_v));
 
    return out;

# define dynamics model
#
class ML_Dynamics2_Model(torch.nn.Module):
  def __init__(self,**params):
    super(ML_Dynamics2_Model, self).__init__()

    # tries to get or sets variable to None
    self.a, self.num_dim = tuple(map(params.get,
                           ['a','num_dim']));

    if self.num_dim is None:
      self.num_dim = 3;

    self.mask_input,self.num_atoms = tuple(map(params.get,
                           ['mask_input','num_atoms']));

    self.list_mask = list_mask = self.mask_input.split();
    self.num_inputs = num_inputs = len(list_mask);

    max_num_inputs = 4;
    mask_input_flag = [False]*max_num_inputs;
       
    # detect which flags are set (organized in order) 
    for i in range(0,self.num_inputs):
      if list_mask[i] == "X":
        mask_input_flag[0] = True;
      elif list_mask[i] == "V":
        mask_input_flag[1] = True;
      elif list_mask[i] == "F":
        mask_input_flag[2] = True;
      elif list_mask[i] == "Type":
        mask_input_flag[3] = True;
      else:
        print("No recognized, list_mask[%d] = %s"%(i,self.list_mask));
    
    self.mask_input_flag = mask_input_flag; 

    # get dimensions of each segment of the data
    # and set index ranges  
    i_dim = []; num_dim = self.num_dim; I0 = 0;     
    if mask_input_flag[0]:  # X
      i_dim.append(num_dim);

    if mask_input_flag[1]:  # V
      i_dim.append(num_dim);

    if mask_input_flag[2]:  # F
      i_dim.append(num_dim);

    if mask_input_flag[3]:  # Type
      i_dim.append(int(1));

    II1 = []; II2 = []; I = 0; # setup ranges
    for k in np.arange(0,num_inputs):    
      d = i_dim[k];
      II1.append(I); I += num_atoms*d; II2.append(I);

    II1 = np.array(II1,dtype=int); II2 = np.array(II2,dtype=int);

    self.i_dim = i_dim; self.II1 = II1; self.II2 = II2;

    nn = 0; ss = [];
    for i in range(0,self.num_inputs):
      nn = nn + i_dim[i]*num_atoms;
      ss.append(i_dim[i]*num_atoms);
    self.nn = nn; self.ss = ss;
 
  # evaluate using F(X,V,F,Type)
  # note, strange bugs can occur in libtorch, if wrong tensor size 
  # i.e. t.shape=[1,n] instead of t.shape[n,1].
  def forward(self, z):
    num_dim = 3;

    a = self.a; num_inputs = self.num_inputs;
    nn = self.nn; ss = self.ss;
    II1 = self.II1; II2 = self.II2;

    #aa = torch.split(z,ss,dim=0);    
    xx = z[II1[0]:II2[0],:];
    x = xx.reshape((num_atoms,num_dim));

    vv = z[II1[1]:II2[1],:];
    v = vv.reshape((num_atoms,num_dim));

    ff = z[II1[2]:II2[2],:];
    f = ff.reshape((num_atoms,num_dim));

    aatype = z[II1[3]:II2[3],:];
    atype = aatype.reshape((num_atoms,1));

    # for testing 
    #out_x = x + v + f + atype*torch.ones((1,num_dim));
    out_x = 0.998*x;
    out_x = out_x.reshape((num_atoms*num_dim,1));  

    out_v = 0.997*v;
    out_v = out_v.reshape((num_atoms*num_dim,1));  
   
    out = torch.vstack((out_x,out_v));
    
    return out;
# Initialize model
mask_input = "X V F Type";
num_inputs = len(mask_input.split());
num_atoms = 5; num_dim = 3; 
#num_atoms = torch.div(z.shape[0], num_inputs, rounding_mode='trunc');
##i_dim = torch.tensor([num_dim,num_dim,num_dim,1],dtype=int);


# create the model (assumes for now fixed number of atoms)
a = 4.1;  
ml_dynamics1_model = ML_Dynamics1_Model(a=a,num_dim=num_dim,
                 mask_input=mask_input,num_atoms=num_atoms);
ml_dynamics1_model.a = a;

# create the model (assumes for now fixed number of atoms)
a = 4.1;  
ml_dynamics2_model = ML_Dynamics2_Model(a=a,num_dim=num_dim,
                 mask_input=mask_input,num_atoms=num_atoms);
ml_dynamics2_model.a = a;

#-- save model data for later validation
case_label = 'gen_001'
base_dir = './output/%s/%s'%(script_base_name,case_label);

print("base_dir = " + base_dir);
create_dir(base_dir);

# WARNING: currently hard-coded
model_name='dynamics1';
f = open('%s/dyn1_%s_params.pickle'%(base_dir,model_name),'wb');
params_force = {
'model_type':'Dyn1_ML1',
'model_name':model_name,
'a':ml_dynamics1_model.a
};
pickle.dump(params_force,f);
f.close();

f = open('%s/dyn2_%s_params.pickle'%(base_dir,model_name),'wb');
params_force = {
'model_type':'Dyn1_ML1',
'model_name':model_name,
'a':ml_dynamics2_model.a
};
pickle.dump(params_force,f);
f.close();

#--
model = ml_dynamics1_model;
nn = ml_dynamics1_model.nn;
z = torch.zeros((nn,1));
traced_ml_dynamics1_model = torch.jit.trace(model, (z))

torch_filename = '%s/dyn1_%s.pt'%(base_dir,model_name);
print("torch_filename = " + torch_filename);
traced_ml_dynamics1_model.save(torch_filename);
print(traced_ml_dynamics1_model.code) # prints trace model code

model = ml_dynamics2_model;
nn = ml_dynamics2_model.nn;
z = torch.zeros((nn,1));
traced_ml_dynamics2_model = torch.jit.trace(model, (z))

torch_filename = '%s/dyn2_%s.pt'%(base_dir,model_name);
print("torch_filename = " + torch_filename);
traced_ml_dynamics2_model.save(torch_filename);
print(traced_ml_dynamics2_model.code) # prints trace model code

cmd = 'cp %s.py %s/archive_%s.py'%(script_base_name,base_dir,script_base_name);
print("cmd = " + cmd);
os.system(cmd);

# #### Show model outputs
nn = ml_dynamics1_model.nn;
z = torch.zeros((nn,1));
z[0*num_dim,0] = 1.0;
z[3*num_dim,0] = 2.0;
z[5*num_dim,0] = 3.0;
z[10*num_dim,0] = 4.0;

print("ml_dynamics1_model");
y = ml_dynamics1_model(z);
print("z.shape = " + str(z.shape));
print("y.shape = " + str(y.shape));

f = open('%s/dyn1_%s_test_data.pickle'%(base_dir,model_name),'wb');
ss = {'model_name':model_name,
      'z':z.cpu().numpy(),
      'y':y.cpu().numpy()};
pickle.dump(ss,f);
f.close();

# #### Show model outputs
nn = ml_dynamics2_model.nn;
z = torch.zeros((nn,1));
z[0*num_dim,0] = 1.0;
z[3*num_dim,0] = 2.0;
z[5*num_dim,0] = 3.0;
z[10*num_dim,0] = 4.0;

print("ml_dynamics2_model");
y = ml_dynamics2_model(z);
print("z.shape = " + str(z.shape));
print("y.shape = " + str(y.shape));

f = open('%s/dyn2_%s_test_data.pickle'%(base_dir,model_name),'wb');
ss = {'model_name':model_name,
      'z':z.cpu().numpy(),
      'y':y.cpu().numpy()};
pickle.dump(ss,f);
f.close();

print("The generated models are located in " + base_dir);

print("Done");


