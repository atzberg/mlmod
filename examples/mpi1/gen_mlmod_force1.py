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
import os, pickle,ipdb;
script_base_name = "gen_mlmod_force1";
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

# define force models
#
class F_ML1_Model(torch.nn.Module):

  def __init__(self,**params):
    super(F_ML1_Model, self).__init__()

    # tries to get or sets variable to None
    self.a, self.num_dim = tuple(map(params.get,
                           ['a','num_dim']));

    if self.num_dim is None:
      self.num_dim = 3;

    self.mask_input,self.num_atoms = tuple(map(params.get,
                           ['mask_input','num_atoms']));

    self.list_mask = list_mask = self.mask_input.split();
    self.num_inputs = num_inputs = len(list_mask);

    self.max_num_inputs = max_num_inputs = 5;
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
      elif list_mask[i] == "Time":
        mask_input_flag[4] = True;
      else:
        print("No recognized, list_mask[%d] = %s"%(i,self.list_mask));
    
    self.mask_input_flag = mask_input_flag; 

    # get dimensions of each segment of the data
    # and set index ranges  
    i_dim = []; num_dim = self.num_dim; I0 = 0;     
    if mask_input_flag[0]:  # X
      i_dim.append(num_dim);
    else:
      i_dim.append(0);

    if mask_input_flag[1]:  # V
      i_dim.append(num_dim);
    else:
      i_dim.append(0);

    if mask_input_flag[2]:  # F
      i_dim.append(num_dim);
    else:
      i_dim.append(0);

    if mask_input_flag[3]:  # Type
      i_dim.append(int(1));
    else:
      i_dim.append(0);

    if mask_input_flag[4]:  # time
      i_dim.append(int(1));
    else:
      i_dim.append(0);

    II1 = []; II2 = []; I = 0; # setup ranges
    for k in np.arange(0,self.max_num_inputs - 1): # X V F Type
      d = i_dim[k];
      II1.append(I); I += num_atoms*d; II2.append(I);

    II1.append(I); I = I + i_dim[4]; II2.append(I); # Time

    II1 = np.array(II1,dtype=int); II2 = np.array(II2,dtype=int);

    self.i_dim = i_dim; self.II1 = II1; self.II2 = II2;

    input_size = 0; input_size_list = [];
    for i in range(0,self.max_num_inputs):
      if i <= 3: # X V F Type
        input_size = input_size + i_dim[i]*num_atoms;
        input_size_list.append(i_dim[i]*num_atoms);  
      elif i == 4: # Time
        input_size = input_size + i_dim[i];
        input_size_list.append(i_dim[i]);  
        
    self.input_size = input_size; self.input_size_list = input_size_list;
 
  # evaluate using F(X,V,F,Type)
  # note, strange bugs can occur in libtorch, if wrong tensor size 
  # i.e. t.shape=[1,n] instead of t.shape[n,1].
  def forward(self, z):
    num_dim = self.num_dim;
    num_atoms = self.num_atoms;

    a = self.a; num_inputs = self.num_inputs;
    input_size = self.input_size; input_size_list = self.input_size_list;
    II1 = self.II1; II2 = self.II2;

    #aa = torch.split(z,ss,dim=0);    
    xx = z[II1[0]:II2[0],:];
    x = xx.reshape((xx.shape[0]//num_dim,num_dim));

    vv = z[II1[1]:II2[1],:];
    v = vv.reshape((vv.shape[0]//num_dim,num_dim));

    ff = z[II1[2]:II2[2],:];
    f = ff.reshape((ff.shape[0]//num_dim,num_dim));

    aatype = z[II1[3]:II2[3],:];
    atype = aatype.reshape((aatype.shape[0],1));

    time = z[II1[4]:II2[4],:];

    # for testing 
    #out = x + v + f + atype*torch.ones((1,num_dim));
    #out = -0.1*x - 0.01*v + 0.1*time;
    out = -0.1*x;
    out = out.reshape((num_atoms*num_dim,1));  
    
    return out;

#
class F_X_ML1_Model(torch.nn.Module):
 
  def __init__(self,**params):
    super(F_X_ML1_Model, self).__init__()

    # tries to get or sets variable to None
    self.a, self.num_dim = tuple(map(params.get,
                           ['a','num_dim']));

    if self.num_dim is None:
      self.num_dim = 3;

    self.mask_input,self.num_atoms = tuple(map(params.get,
                           ['mask_input','num_atoms']));

    self.list_mask = list_mask = self.mask_input.split();
    self.num_inputs = num_inputs = len(list_mask);

    self.max_num_inputs = max_num_inputs = 5;
    mask_input_flag = [False]*max_num_inputs;

    num_atoms = self.num_atoms; 
       
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
      elif list_mask[i] == "Time":
        mask_input_flag[4] = True;
      else:
        print("No recognized, list_mask[%d] = %s"%(i,self.list_mask));
    
    self.mask_input_flag = mask_input_flag; 

    # get dimensions of each segment of the data
    # and set index ranges  
    i_dim = []; num_dim = self.num_dim; I0 = 0;     
    if mask_input_flag[0]:  # X
      i_dim.append(num_dim);
    else:
      i_dim.append(0);

    if mask_input_flag[1]:  # V
      i_dim.append(num_dim);
    else:
      i_dim.append(0);

    if mask_input_flag[2]:  # F
      i_dim.append(num_dim);
    else:
      i_dim.append(0);

    if mask_input_flag[3]:  # Type
      i_dim.append(int(1));
    else:
      i_dim.append(0);

    if mask_input_flag[4]:  # time
      i_dim.append(int(1));
    else:
      i_dim.append(0);

    II1 = []; II2 = []; I = 0; # setup ranges
    for k in np.arange(0,self.max_num_inputs - 1): # X V F Type
      d = i_dim[k];
      II1.append(I); I += num_atoms*d; II2.append(I);

    II1.append(I); I = I + i_dim[4]; II2.append(I); # Time

    II1 = np.array(II1,dtype=int); II2 = np.array(II2,dtype=int);

    self.i_dim = i_dim; self.II1 = II1; self.II2 = II2;

    input_size = 0; input_size_list = [];
    for i in range(0,self.max_num_inputs):
      if i <= 3: # X V F Type
        input_size = input_size + i_dim[i]*num_atoms;
        input_size_list.append(i_dim[i]*num_atoms);  
      elif i == 4: # Time
        input_size = input_size + i_dim[i];
        input_size_list.append(i_dim[i]);  
        
    self.input_size = input_size; self.input_size_list = input_size_list;

  # evaluate using F(X,V,F,Type)
  # note, strange bugs can occur in libtorch, if wrong tensor size 
  # i.e. t.shape=[1,n] instead of t.shape[n,1].
  def forward(self, z):
    num_dim = self.num_dim;
    num_atoms = self.num_atoms;

    a = self.a; num_inputs = self.num_inputs;
    input_size = self.input_size; input_size_list = self.input_size_list;
    II1 = self.II1; II2 = self.II2;

    #aa = torch.split(z,ss,dim=0);    
    xx = z[II1[0]:II2[0],:];
    x = xx.reshape((xx.shape[0]//num_dim,num_dim));

    vv = z[II1[1]:II2[1],:];
    v = vv.reshape((vv.shape[0]//num_dim,num_dim));

    ff = z[II1[2]:II2[2],:];
    f = ff.reshape((ff.shape[0]//num_dim,num_dim));

    aatype = z[II1[3]:II2[3],:];
    atype = aatype.reshape((aatype.shape[0],1));

    time = z[II1[4]:II2[4],:];

    # for testing 
    #out = x + v + f + atype*torch.ones((1,num_dim));
    #out = -0.1*x - 0.01*v + 0.1*time;
    #out = -0.1*x + 0.1*time + 0.1*atype;
    out = -0.1*x - 0.1*v;
    out = out.reshape((num_atoms*num_dim,1));  
    
    return out;

#
class F_Pair_ML1_Model(torch.nn.Module):

  def __init__(self,**params):
   super(F_Pair_ML1_Model, self).__init__()

   # tries to get or sets variable to None
   self.a, self.num_dim = tuple(map(params.get,
			  ['a','num_dim']));

   if self.num_dim is None:
     self.num_dim = 3;

   self.mask_input,self.num_atoms = tuple(map(params.get,
			  ['mask_input','num_atoms']));

   self.list_mask = list_mask = self.mask_input.split();
   self.num_inputs = num_inputs = len(list_mask);

   self.max_num_inputs = max_num_inputs = 5;
   mask_input_flag = [False]*max_num_inputs;

   num_atoms = self.num_atoms; 
 
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
     elif list_mask[i] == "Time":
       mask_input_flag[4] = True;
     else:
       print("No recognized, list_mask[%d] = %s"%(i,self.list_mask));
   
   self.mask_input_flag = mask_input_flag; 

   # get dimensions of each segment of the data
   # and set index ranges  
   i_dim = []; num_dim = self.num_dim; I0 = 0;     
   if mask_input_flag[0]:  # X
     i_dim.append(num_dim);
   else:
     i_dim.append(0);

   if mask_input_flag[1]:  # V
     i_dim.append(num_dim);
   else:
     i_dim.append(0);

   if mask_input_flag[2]:  # F
     i_dim.append(num_dim);
   else:
     i_dim.append(0);

   if mask_input_flag[3]:  # Type
     i_dim.append(int(1));
   else:
     i_dim.append(0);

   if mask_input_flag[4]:  # time
     i_dim.append(int(1));
   else:
     i_dim.append(0);

   II1 = []; II2 = []; I = 0; # setup ranges
   for k in np.arange(0,self.max_num_inputs - 1): # X V F Type
     d = i_dim[k];
     II1.append(I); I += num_atoms*d; II2.append(I);

   II1.append(I); I = I + i_dim[4]; II2.append(I); # Time

   II1 = np.array(II1,dtype=int); II2 = np.array(II2,dtype=int);

   self.i_dim = i_dim; self.II1 = II1; self.II2 = II2;

   input_size = 0; input_size_list = [];
   for i in range(0,self.max_num_inputs):
     if i <= 3: # X V F Type
       input_size = input_size + i_dim[i]*num_atoms;
       input_size_list.append(i_dim[i]*num_atoms);  
     elif i == 4: # Time
       input_size = input_size + i_dim[i];
       input_size_list.append(i_dim[i]);  
       
   self.input_size = input_size; self.input_size_list = input_size_list;
 
  # evaluate using F(X,V,F,Type)
  # note, strange bugs can occur in libtorch, if wrong tensor size 
  # i.e. t.shape=[1,n] instead of t.shape[n,1].
  def forward(self, z):
    num_dim = self.num_dim;
    num_atoms = self.num_atoms;

    a = self.a; num_inputs = self.num_inputs;
    input_size = self.input_size; input_size_list = self.input_size_list;
    II1 = self.II1; II2 = self.II2;

    #aa = torch.split(z,ss,dim=0);    
    xx = z[II1[0]:II2[0],:];
    x = xx.reshape((xx.shape[0]//num_dim,num_dim));

    vv = z[II1[1]:II2[1],:];
    v = vv.reshape((vv.shape[0]//num_dim,num_dim));

    ff = z[II1[2]:II2[2],:];
    f = ff.reshape((ff.shape[0]//num_dim,num_dim));

    aatype = z[II1[3]:II2[3],:];
    atype = aatype.reshape((aatype.shape[0],1));

    time = z[II1[4]:II2[4],:];

    # for testing 
    #out = x + v + f + atype*torch.ones((1,num_dim));
    #out = -0.1*x - 0.01*v + 0.1*time;
    out = -0.1*x;
    out = out.reshape((num_atoms*num_dim,1));  
    
    return out;


#-- save model data for later validation
print("="*80);
print("Generating mlmod models.");
print("-"*80);
case_label = 'gen_001'
base_dir = './output/%s/%s'%(script_base_name,case_label);

print("base_dir = " + base_dir);
create_dir(base_dir);

# copy script
cmd = 'cp %s.py %s/archive_%s.py'%(script_base_name,base_dir,script_base_name);
print("cmd = " + cmd);
os.system(cmd);

#--
# Initialize model
print("-"*80);
model_name='force1';
model_type = 'F_ML1';
print("model_name = " + model_name);
print("model_type = " + model_type);
print("."*80);

#mask_input = "X V F Type Time";
mask_input = "X V Time";
print("mask_input = " + mask_input);
num_inputs = len(mask_input.split());
num_atoms = 5; num_dim = 3; 
#num_atoms = torch.div(z.shape[0], num_inputs, rounding_mode='trunc');
##i_dim = torch.tensor([num_dim,num_dim,num_dim,1],dtype=int);

# create the model (assumes for now fixed number of atoms)
a = 4.1;  
f_ml1_model = F_ML1_Model(a=a,num_dim=num_dim,
                          mask_input=mask_input,num_atoms=num_atoms);
f_ml1_model.a = a;

# WARNING: currently hard-coded
f = open('%s/F_%s_%s_params.pickle'%(base_dir,model_name,model_type),'wb');
params_force = {
'model_type':model_type,
'model_name':model_name,
'mask_input':f_ml1_model.mask_input,
'mask_input_flag':f_ml1_model.mask_input_flag,
'num_atoms':f_ml1_model.num_atoms,
'num_dim':f_ml1_model.num_dim,
'input_size':f_ml1_model.input_size,
'input_size_list':f_ml1_model.input_size_list,
'II1':f_ml1_model.II1,'II2':f_ml1_model.II2,
'a':f_ml1_model.a,
};
pickle.dump(params_force,f);
f.close();

model = f_ml1_model;
input_size = f_ml1_model.input_size;
z = torch.zeros((input_size,1));
traced_f_ml1_model = torch.jit.trace(model, (z))

torch_filename = '%s/F_%s_%s.pt'%(base_dir,model_name,model_type);
print("torch_filename = " + torch_filename);
traced_f_ml1_model.save(torch_filename);
print(traced_f_ml1_model.code) # prints trace model code

# #### Show model outputs
input_size = f_ml1_model.input_size;
II1 = f_ml1_model.II1; II2 = f_ml1_model.II2; 
z = torch.zeros((input_size,1));
if f_ml1_model.mask_input_flag[0]: # X
  z[II1[0],0] = 1.0;
  z[II1[0] + num_dim,0] = 2.0;
  z[II1[0] + 2*num_dim,0] = 3.0;
if f_ml1_model.mask_input_flag[1]: # V
  z[II1[1],0] = -1.0;
if f_ml1_model.mask_input_flag[2]: # F
  z[II1[2],0] = 0.11;
if f_ml1_model.mask_input_flag[3]: # Type
  z[II1[3],0] = 2.0
if f_ml1_model.mask_input_flag[4]: # Time
  z[II1[4],0] = 1.1; # Time

print("f_ml1_model");
y = f_ml1_model(z);
print("z.shape = " + str(z.shape));
print("y.shape = " + str(y.shape));

f = open('%s/F_%s_%s_test_data.pickle'%(base_dir,model_name,model_type),'wb');
ss = {'model_name':model_name,'model_type':model_type,
      'z':z.cpu().numpy(),
      'y':y.cpu().numpy()};
pickle.dump(ss,f);
f.close();
  
#--
# Initialize model
print("-"*80);
model_name='force1';
model_type = 'F_X_ML1';
print("model_name = " + model_name);
print("model_type = " + model_type);
print("."*80);

#mask_input = "X V F Type Time";
mask_input = "X V Type Time";
print("mask_input = " + mask_input);
num_inputs = len(mask_input.split());
num_atoms = 1; num_dim = 3; 
#num_atoms = torch.div(z.shape[0], num_inputs, rounding_mode='trunc');
##i_dim = torch.tensor([num_dim,num_dim,num_dim,1],dtype=int);

# create the model (assumes for now fixed number of atoms)
a = 4.1;  
f_ml1_model = F_X_ML1_Model(a=a,num_dim=num_dim,
                            mask_input=mask_input,num_atoms=1);
f_ml1_model.a = a;

# WARNING: currently hard-coded
f = open('%s/F_%s_%s_params.pickle'%(base_dir,model_name,model_type),'wb');
params_force = {
'model_type':model_type,
'model_name':model_name,
'mask_input':f_ml1_model.mask_input,
'mask_input_flag':f_ml1_model.mask_input_flag,
'num_atoms':f_ml1_model.num_atoms,
'num_dim':f_ml1_model.num_dim,
'input_size':f_ml1_model.input_size,
'input_size_list':f_ml1_model.input_size_list,
'II1':f_ml1_model.II1,'II2':f_ml1_model.II2,
'a':f_ml1_model.a,
};
pickle.dump(params_force,f);
f.close();

model = f_ml1_model;
input_size = f_ml1_model.input_size;
z = torch.zeros((input_size,1));
traced_f_ml1_model = torch.jit.trace(model, (z))

torch_filename = '%s/F_%s_%s.pt'%(base_dir,model_name,model_type);
print("torch_filename = " + torch_filename);
traced_f_ml1_model.save(torch_filename);
print(traced_f_ml1_model.code) # prints trace model code

# #### Show model outputs
input_size = f_ml1_model.input_size;
z = torch.zeros((input_size,1));
z[0*num_dim,0] = 1.0;
z[-1,0] = 1.1;

print("f_ml1_model");
y = f_ml1_model(z);
print("z.shape = " + str(z.shape));
print("y.shape = " + str(y.shape));

f = open('%s/F_%s_%s_test_data.pickle'%(base_dir,model_name,model_type),'wb');
ss = {'model_name':model_name,'model_type':model_type,
      'z':z.cpu().numpy(),
      'y':y.cpu().numpy()};
pickle.dump(ss,f);
f.close();


#--
# Initialize model
print("-"*80);
model_name='force1';
model_type = 'F_Pair_ML1';
print("model_name = " + model_name);
print("model_type = " + model_type);
print("."*80);

#mask_input = "X V F Type Time";
mask_input = "X F";
print("mask_input = " + mask_input);
#mask_input = "X Time";
num_inputs = len(mask_input.split());
num_atoms = 5; num_dim = 3; 
#num_atoms = torch.div(z.shape[0], num_inputs, rounding_mode='trunc');
##i_dim = torch.tensor([num_dim,num_dim,num_dim,1],dtype=int);

# create the model (assumes for now fixed number of atoms)
a = 4.1;  
f_ml1_model = F_Pair_ML1_Model(a=a,num_dim=num_dim,
                               mask_input=mask_input,num_atoms=2);
f_ml1_model.a = a;

# WARNING: currently hard-coded
f = open('%s/F_%s_%s_params.pickle'%(base_dir,model_name,model_type),'wb');
params_force = {
'model_type':model_type,
'model_name':model_name,
'mask_input':f_ml1_model.mask_input,
'mask_input_flag':f_ml1_model.mask_input_flag,
'num_atoms':f_ml1_model.num_atoms,
'num_dim':f_ml1_model.num_dim,
'input_size':f_ml1_model.input_size,
'input_size_list':f_ml1_model.input_size_list,
'II1':f_ml1_model.II1,'II2':f_ml1_model.II2,
'a':f_ml1_model.a,
};
pickle.dump(params_force,f);
f.close();

model = f_ml1_model;
input_size = f_ml1_model.input_size;
z = torch.zeros((input_size,1));
traced_f_ml1_model = torch.jit.trace(model, (z))

torch_filename = '%s/F_%s_%s.pt'%(base_dir,model_name,model_type);
print("torch_filename = " + torch_filename);
traced_f_ml1_model.save(torch_filename);
print(traced_f_ml1_model.code) # prints trace model code

# #### Show model outputs
input_size = f_ml1_model.input_size;
z = torch.zeros((input_size,1));
z[0*num_dim,0] = 1.0;
z[3*num_dim,0] = 2.0;
z[-1,0] = 1.1;

print("f_ml1_model");
y = f_ml1_model(z);
print("z.shape = " + str(z.shape));
print("y.shape = " + str(y.shape));

f = open('%s/F_%s_%s_test_data.pickle'%(base_dir,model_name,model_type),'wb');
ss = {'model_name':model_name,'model_type':model_type,
      'z':z.cpu().numpy(),
      'y':y.cpu().numpy()};
pickle.dump(ss,f);
f.close();

print("The generated models are located in " + base_dir);

print("Done");

print("="*80);



