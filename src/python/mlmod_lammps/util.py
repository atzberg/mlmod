r"""Utility routines.  Paul J. Atzberger, http://atzberger.org/ """;

import os,sys,shutil,pickle,logging;
import numpy as np;

# setup default print_log statement
global print_log;
print_log = print;

# lammps related
def print_version_info(L,print_log=print):
  #L.command("info all out overwrite tmp.txt"); # dump all current info to file.
  L.command("info all out overwrite tmp.txt"); # dump all current info to file.
  f = open("tmp.txt"); ss = f.read(); f.close(); # load a display for the user
  results = [];
  for line in ss.splitlines():
    if "version" in line:
      print_log(line);   

# filesystem management
def create_dir(dir_name):
  if not os.path.exists(dir_name):
    os.makedirs(dir_name);    
    
def rm_dir(dir_name):
  if os.path.exists(dir_name):    
    shutil.rmtree(dir_name);
  else: 
    print_log("WARNING: rm_dir(): The directory does not exist, dir_name = " + dir_name);    

def copytree2(src, dst, symlinks=False, ignore=None):
  for ff in os.listdir(src):
    s = os.path.join(src, ff); d = os.path.join(dst, ff);
    if os.path.isdir(s):
      shutil.copytree(s, d, symlinks, ignore);
    else:
      shutil.copy2(s, d);
    
# plotting
def save_fig(base_filename,extraLabel='',flag_verbose=True,dpi_set=200,flag_pdf=False,bbox_inches=None,print_log=print):
    
  import matplotlib;
  import matplotlib.pyplot as plt;

  flag_simple = False;
  if flag_simple: # avoids flashing, just uses screen resolution (ignores dpi_set)
    if flag_pdf:    
      save_filename = '%s%s.pdf'%(base_filename,extraLabel);
      if flag_verbose:
        print_log('save_filename = %s'%save_filename);
      #plt.savefig(save_filename, format='pdf',dpi=dpi_set,facecolor=(1,1,1,1),alpha=1.0);
      plt.savefig(save_filename, format='pdf',bbox_inches=bbox_inches);

    save_filename = '%s%s.png'%(base_filename,extraLabel);
    if flag_verbose:
      print_log('save_filename = %s'%save_filename);
    #plt.savefig(save_filename, format='png',dpi=dpi_set,facecolor=(1,1,1,1),alpha=1.0);
    plt.savefig(save_filename, format='png',bbox_inches=bbox_inches);
  else:  # NOTE: Can also get same below likely by setting facecolor='white'.
    fig = plt.gcf();  
    fig.patch.set_alpha(1.0);
    fig.patch.set_facecolor((1.0,1.0,1.0,1.0));
    
    if flag_pdf:
      save_filename = '%s%s.pdf'%(base_filename,extraLabel);
      if flag_verbose:
        print_log('save_filename = %s'%save_filename);
      plt.savefig(save_filename, format='pdf',dpi=dpi_set,facecolor=(1,1,1,1),bbox_inches=bbox_inches);
      #plt.savefig(save_filename, format='pdf',dpi=dpi_set);

    save_filename = '%s%s.png'%(base_filename,extraLabel);
    if flag_verbose:
      print_log('save_filename = %s'%save_filename);
    plt.savefig(save_filename, format='png',dpi=dpi_set,facecolor=(1,1,1,1),bbox_inches=bbox_inches);
    #plt.savefig(save_filename, format='png',dpi=dpi_set);

    
# logging 
# Setup for mapping print --> to logging (only works for print in the notebook)
class AtzLogging():

    def __init__(self,print_handle,base_dir,level=logging.DEBUG):
      self.flag_log_init = False;
      self.logger = None; self.log_print_handle = None; # default

      self.setup_log(print_handle,base_dir,level=logging.DEBUG);

    def print_log(self,str): 
      if self.flag_log_init: 
        self.logger.info(str);
        #log_print_handle(str);    
      else:
        print(str);

    def print_log_debug(self,str):
      if self.flag_log_init:  
        self.logger.debug(str);
        #log_print_handle(str);
      else:
        print(str);

    def setup_log(self,print_handle,base_dir,level=logging.DEBUG):
      #global self.flag_log_init, self.logger, self.log_print_handle;

      self.log_print_handle = print_handle;
      # logging.basicConfig(filename='%s/main.log'%base_dir,level=logging.DEBUG);
      filename = '%s/main.log'%base_dir;
      print("Setting up log file in filename = " + filename);
      #logging.basicConfig(filename=filename,level=level,format='%(levelname)s:%(message)s');
      #logging.basicConfig(filename=filename,level=level,filemode='w',format='%(message)s'); # filemode overwrite
      #logging.basicConfig(filename=filename,level=level,format='%(message)s'); # filemode overwrite
      logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[logging.FileHandler(filename=filename),
                  logging.StreamHandler(sys.stdout)]);
      self.logger = logging.getLogger('LOGGER_NAME');

      self.flag_log_init = True;
      
      print_log = print_handle; # set global print_log function
    
# generic
def atz_find_name(l,s):
  I = 0;  I0 = None;
  for ll in l:
    if s == ll:
      I0 = I;
    I+=1;
    
  return I0;


#lammps 

def write_read_data(**params):
    filename = params['filename'];    
    #filename = "Polymer.LAMMPS_read_data";
    
    #iif 'print_log' in params:
    #  print_log = params['print_log'];
    #else:
    #  print_log = print;
    
    if 'flag_verbose' in params:
      flag_verbose = params['flag_verbose'];
    else:
      flag_verbose = 0;    
    
    box = params['box'];    
    atom_types,bond_types,angle_types = tuple(map(params.get,['atom_types','bond_types','angle_types']));
    atom_list,atom_mass_list,atom_mol_list,atom_id_list = tuple(map(params.get,['atom_list','atom_mass_list','atom_mol_list','atom_id_list']));
    bond_list,bond_id_list,bond_coeff_list = tuple(map(params.get,['bond_list','bond_id_list','bond_coeff_list']));
    angle_list,angle_id_list,angle_coeff_list = tuple(map(params.get,['angle_list','angle_id_list','angle_coeff_list']));
    
    num_atoms = 0;
    for I_type in atom_types:
      x = atom_list[I_type - 1]; 
      num_atoms += x.shape[0];
        
    num_bonds = 0;
    for I_type in bond_types:
      b = bond_list[I_type - 1]; 
      num_bonds += b.shape[0]; 
        
    num_angles = 0;
    for I_type in angle_types:
      a = angle_list[I_type - 1]; 
      num_angles += a.shape[0];         
            
    if flag_verbose > 0:
      print_log("Writing:")
      print_log("filename = " + filename);
    
    f = open(filename,"w");

    # note the triple quotes use literal formatting, so avoid indents
    s = """
# =========================================================================
# LAMMPS file for 'read_data' command                                      
# 
# Generated by mlmod python scripts by Paul J. Atzberger.
# 
# =========================================================================
""";
    f.write(s);

    s = """
# =========================================================================
# Description:
# -------------------------------------------------------------------------
# 
# SELM_Lagrangian = SELM_Lagrangian_LAMMPS_ATOM_ANGLE_STYLE
# LagrangianName = Points
# LagrangianTypeStr = LAMMPS_ATOM_ANGLE_STYLE
# 
# SELM_Eulerian   = SELM_Eulerian_LAMMPS_SHEAR_UNIFORM1_FFTW3
#
# atom_type = angle_type
#
# =========================================================================
""";
    f.write(s);

    s = """
# =========================================================================
# Header information:
# -------------------------------------------------------------------------
%d atoms
%d bonds
%d angles

%d atom types
%d bond types
%d angle types
# =========================================================================
"""%(num_atoms,num_bonds,num_angles,atom_types.shape[0],bond_types.shape[0],angle_types.shape[0]);
    f.write(s);

    s = """
# =========================================================================
# Domain Size Specification:
# -------------------------------------------------------------------------
%d %d xlo xhi
%d %d ylo yhi
%d %d zlo zhi
0.0 0.0 0.0 xy xz yz
# =========================================================================
"""%tuple(box.flatten());
    f.write(s);

    s = """
# =========================================================================
# Mass Specification:
#
# Gives for each atom the following:
#    type-ID | mass
# -------------------------------------------------------------------------
# Atom Location Specification:
#
# Gives for atom angle_type the following:
#    atom-ID | molecule-ID | type-ID | x | y | z
# -------------------------------------------------------------------------
# Bond Specification:
#
# Gives for atom angle_type the following:
#    bond-ID | type-ID | atom1-ID | atom2-ID
# -------------------------------------------------------------------------
# Angle Specification:
#
# Gives for atom angle_type the following:
#    angle-ID | type-ID | atom1-ID | atom2-ID | atom3-ID
# -------------------------------------------------------------------------
# WARNING: atom-ID, type-ID, molecule-ID must be between 1 - N             
# -------------------------------------------------------------------------
""";
    f.write(s);

    s = """
Masses

""";
    for I_type in atom_types:
      m0 = atom_mass_list[I_type - 1];
      s += "%d %.7f\n"%(I_type,m0);

    f.write(s);

    s = """
Atoms

""";
    for I_type in atom_types:
      x = atom_list[I_type - 1]; 
      atom_id = atom_id_list[I_type - 1];
      atom_mol = atom_mol_list[I_type - 1];
      for I in range(0,x.shape[0]):
        I_id = atom_id[I]; I_mol = atom_mol[I];
        s += "%d %d %d "%(I_id,I_mol,I_type);
        s += "%.7f %.7f %.7f\n"%tuple(x[I,:]);    
    f.write(s);

    if num_bonds > 0:
      # note below does not violate scope, but needed for formatting
      s = """
Bond Coeffs

""";
      f.write(s);

      s = "";
      for I_type in bond_types:
        bond_coeff = bond_coeff_list[I_type - 1];
        s += "%d %s\n"%(I_type,bond_coeff);

      f.write(s);

      # note below does not violate scope, but needed for formatting
      s = """
Bonds

""";
      f.write(s);

      s = "";
      for I_type in bond_types:  
        bonds = bond_list[I_type - 1]; bond_id = bond_id_list[I_type - 1];
        for I in range(0,bonds.shape[0]):
          I_id = bond_id[I];
          s += "%d %d %d %d\n"%(I_id,I_type,bonds[I,0],bonds[I,1]);

      f.write(s);

    if num_angles > 0:
      # note below does not violate scope, but needed for formatting  
      s = """
Angle Coeffs

""";
      f.write(s);

      for I_type in angle_types:
        s = "";  
        angle_coeff = angle_coeff_list[I_type - 1];
        s += "%d %s\n"%(I_type,angle_coeff);
        f.write(s);

    if num_angles > 0:  
      # note below does not violate scope, but needed for formatting  
      s = """
Angles

""";
      f.write(s);    

      s = "";
      for I_type in angle_types:
        angles = angle_list[I_type - 1]; angle_id = angle_id_list[I_type - 1];
        for I in range(0,angles.shape[0]):
          I_id = angle_id[I];
          s += "%d %d %d %d %d\n"%(I_id,I_type,angles[I,0],angles[I,1],angles[I,2]);

      f.write(s);

      #if (angle_types.shape[0] > 0):
      #  print("WARNING: Need to also add writing bond angles.");

    f.close();

def write_mlmod_F_ML1_params(filename,params):
  model_type = params['model_type']; model_data = params['model_data'];
  base_name, base_dir, mask_fix, mask_input, F_filename = tuple(map(model_data.get,\
                                  ['base_name', 'base_dir', 'mask_fix', 
                                   'mask_input', 'F_filename']));
  
  # xml file
  f = open(filename,'w');
  f.write('<?xml version="1.0" encoding="UTF-8"?>\n');
  f.write('<MLMOD>\n');
  f.write('<model_data type="' + model_type + '">\n');
  f.write('<base_name value="' + base_name + '"/>\n');
  f.write('<base_dir value="' + base_dir + '"/>\n');
  f.write('<mask_fix value="' + mask_fix + '"/>\n');
  f.write('<mask_input value="' + mask_input + '"/>\n');
  f.write('<F_filename value="' + F_filename + '"/>\n');
  f.write('</model_data>\n');
  f.write('</MLMOD>\n');
  f.close();

  # pickle file
  f = open(filename + '.pickle','wb'); pickle.dump(params,f); f.close();

def write_mlmod_F_X_ML1_params(filename,params):
  model_type = params['model_type']; model_data = params['model_data'];
  base_name, base_dir, mask_fix, mask_input, F_filename = tuple(map(model_data.get,\
                                  ['base_name', 'base_dir', 'mask_fix', 
                                   'mask_input', 'F_filename']));
  
  # xml file
  f = open(filename,'w');
  f.write('<?xml version="1.0" encoding="UTF-8"?>\n');
  f.write('<MLMOD>\n');
  f.write('<model_data type="' + model_type + '">\n');
  f.write('<base_name value="' + base_name + '"/>\n');
  f.write('<base_dir value="' + base_dir + '"/>\n');
  f.write('<mask_fix value="' + mask_fix + '"/>\n');
  f.write('<mask_input value="' + mask_input + '"/>\n');
  f.write('<F_filename value="' + F_filename + '"/>\n');
  f.write('</model_data>\n');
  f.write('</MLMOD>\n');
  f.close();

  # pickle file
  f = open(filename + '.pickle','wb'); pickle.dump(params,f); f.close();

def write_mlmod_F_Pair_ML1_params(filename,params):
  model_type = params['model_type']; model_data = params['model_data'];
  base_name, base_dir, mask_fix, mask_input, F_filename = tuple(map(model_data.get,\
                                  ['base_name', 'base_dir', 'mask_fix', 
                                   'mask_input', 'F_filename']));
  
  # xml file
  f = open(filename,'w');
  f.write('<?xml version="1.0" encoding="UTF-8"?>\n');
  f.write('<MLMOD>\n');
  f.write('<model_data type="' + model_type + '">\n');
  f.write('<base_name value="' + base_name + '"/>\n');
  f.write('<base_dir value="' + base_dir + '"/>\n');
  f.write('<mask_fix value="' + mask_fix + '"/>\n');
  f.write('<mask_input value="' + mask_input + '"/>\n');
  f.write('<F_filename value="' + F_filename + '"/>\n');
  f.write('</model_data>\n');
  f.write('</MLMOD>\n');
  f.close();

  # pickle file
  f = open(filename + '.pickle','wb'); pickle.dump(params,f); f.close();


def write_mlmod_Dyn_ML1_params(filename,params):
  model_type = params['model_type']; model_data = params['model_data'];
  base_name, base_dir, mask_fix, mask_input, dyn1_filename, dyn2_filename = tuple(map(model_data.get,\
                                  ['base_name', 'base_dir', 'mask_fix', 
                                   'mask_input', 'dyn1_filename', 'dyn2_filename']));
  
  # xml file
  f = open(filename,'w');
  f.write('<?xml version="1.0" encoding="UTF-8"?>\n');
  f.write('<MLMOD>\n');
  f.write('<model_data type="' + model_type + '">\n');
  f.write('  <base_name value="' + base_name + '"/>\n');
  f.write('  <base_dir value="' + base_dir + '"/>\n');
  f.write('  <mask_fix value="' + mask_fix + '"/>\n');
  f.write('  <mask_input value="' + mask_input + '"/>\n');
  f.write('  <dyn1_filename value="' + dyn1_filename + '"/>\n');
  f.write('  <dyn2_filename value="' + dyn2_filename + '"/>\n');
  f.write('</model_data>\n');
  f.write('</MLMOD>\n');
  f.close();

  # pickle file
  f = open(filename + '.pickle','wb'); pickle.dump(params,f); f.close();

def write_mlmod_QoI_ML1_params(filename,params):
  model_type = params['model_type']; model_data = params['model_data'];
  base_name, base_dir, mask_fix, mask_input, skip_step, qoi_filename = tuple(map(model_data.get,\
                                  ['base_name', 'base_dir', 'mask_fix', 
                                   'mask_input', 'skip_step', 'qoi_filename']));
  
  # xml file
  f = open(filename,'w');
  f.write('<?xml version="1.0" encoding="UTF-8"?>\n');
  f.write('<MLMOD>\n');
  f.write('<model_data type="' + model_type + '">\n');
  f.write('  <base_name value="' + base_name + '"/>\n');
  f.write('  <base_dir value="' + base_dir + '"/>\n');
  f.write('  <mask_fix value="' + mask_fix + '"/>\n');
  f.write('  <mask_input value="' + mask_input + '"/>\n');
  f.write('  <skip_step value="%d"'%skip_step + '/>\n');
  f.write('  <qoi_filename value="' + qoi_filename + '"/>\n');
  f.write('</model_data>\n');
  f.write('</MLMOD>\n');
  f.close();

  # pickle file
  f = open(filename + '.pickle','wb'); pickle.dump(params,f); f.close();

def write_mlmod_dX_MF_ML1_params(filename,params):
  model_type = params['model_type'];
  model_data = params['model_data'];
  
  # xml file
  f = open(filename,'w');
  f.write('<?xml version="1.0" encoding="UTF-8"?>\n');
  f.write('<MLMOD>\n');
  f.write('<model_data type="' + model_type + '">\n');
  f.write('  <M_ii_filename value="' + model_data['M_ii_filename'] + '"/>\n');
  f.write('  <M_ij_filename value="' + model_data['M_ij_filename'] + '"/>\n');
  f.write('</model_data>\n');
  f.write('</MLMOD>\n');
  f.close();

  # pickle file
  f = open(filename + '.pickle','wb'); pickle.dump(params,f); f.close();

def write_mlmod_params(filename,params):
  if params['model_type'] == 'F_ML1': 
    write_mlmod_F_ML1_params(filename,params);
  elif params['model_type'] == 'F_X_ML1': 
    write_mlmod_F_X_ML1_params(filename,params);
  elif params['model_type'] == 'F_Pair_ML1': 
    write_mlmod_F_Pair_ML1_params(filename,params);
  elif params['model_type'] == 'Dyn_ML1': 
    write_mlmod_Dyn_ML1_params(filename,params);
  elif params['model_type'] == 'QoI_ML1': 
    write_mlmod_QoI_ML1_params(filename,params);
  elif params['model_type'] == 'dX_MF_ML1': 
    write_mlmod_dX_MF_ML1_params(filename,params);
  else:
    raise Exception("No recongnized model_type = " + str(model_type));
  
def find_closest_pt(x0,xx):
  dist_sq = np.sum(np.power(xx - np.expand_dims(x0,0),2),1);
  ii0 = np.argmin(dist_sq);
  return ii0;

def Lc_print(ss,L=None):
  # typical usage is to wrap as: 
  #   Lc = lambda ss: m_util.wrap_lammps_Lc(ss,L); 
  print(ss);  # show the command about to run
  L.command(ss); 

# wraps the lammps L object to augment behaviors
# such as further display or logging
def wrap_L(L,wrap_func):
  Lc = lambda ss: wrap_func(ss,L);  # lammps commands 
  return Lc;

