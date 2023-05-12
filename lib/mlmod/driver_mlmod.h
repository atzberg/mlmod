/* -----------------------------------------------------------------------------

  MLMOD: Machine Learning (ML) Modeling (MOD) and Simulation Package 

  Paul J. Atzberger
  http://atzberger.org/
  
  Please cite the follow paper when referencing this package
  
  "MLMOD Package: Machine Learning Methods for Data-Driven Modeling in LAMMPS",
  Atzberger, P. J., arXiv:2107.14362, 2021.
  
  @article{mlmod_atzberger,
    author  = {Atzberger, P. J.},
    journal = {arxiv},
    title   = {MLMOD Package: Machine Learning Methods for Data-Driven Modeling in LAMMPS},
    year    = {2021},
    note    = {http://atzberger.org},
    doi     = {10.48550/arXiv.2107.14362},
    url     = {https://arxiv.org/abs/2107.14362},
  }
    
  For latest releases, examples, and additional information see 
  http://atzberger.org/
 
--------------------------------------------------------------------------------
*/

#ifndef LMP_DRIVER_MLMOD_H
#define LMP_DRIVER_MLMOD_H

#ifndef FFT_FFTW /* determine if FFTW package specified */
  #warning "No FFTW package specified for MLMOD codes.  Some MLMOD functionality may be disabled."
#else
  #define USE_PACKAGE_FFTW3
#endif

#include "fix_mlmod.h"

// PyTorch C/C++ interface
#include <torch/torch.h>

#include "tinyxml2.h"

//#include "mpi.h"
#include "fix.h"
#include "lammps.h"
#include "random_mars.h"

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cstddef>

using namespace std;
using namespace tinyxml2;
using namespace LAMMPS_NS;

namespace USER_MLMOD {

struct DriverMLMOD {

 public:

  /* ================= function prototypes ================= */
  DriverMLMOD(class FixMLMOD *, class LAMMPS *, int, char **);

  DriverMLMOD(); /* used to construct empty object,
                primarily for testing purposes */
  ~DriverMLMOD();

  // =========================== Fix Related Function Calls ==================== */
  int          setmask();
  virtual void init();
  void         setup(int vflag);
  virtual void initial_integrate(int);
  virtual void final_integrate();  
  void         reset_dt();
  void         post_force(int vflag);
  
  // additional methods/hooks
  void pre_exchange();
  void end_of_step();

  /* =========================== MLMOD Function Calls =========================== */
  void init_attributes();
  void init_from_fix();
  void constr_input_masked(at::TensorAccessor<float,2,at::DefaultPtrTraits,long> input_m_a,
                           int num_indices,int *indices,
                           int mask_input, int mask_list_n, int *mask_list,
                           double **x,double **v,double **f, int *type);
  void parse_xml_params(char *filename);
  
  /* write the MLMOD simulation data to disk */
  //void writeAllSimulationData(int timeIndex);
  void writeInfo();
  void writeFinalInfo();

  /* case specific functions */ 
  void initial_integrate_dX_MF_Q1_ML1_Pair();  
  void initial_integrate_dX_MF_Q1_ML1_N2N();  
  void initial_integrate_F_ML1();  
  void initial_integrate_QoI_ML1();  
  void initial_integrate_Dyn_ML1();  

  void final_integrate_dX_MF();  
  void final_integrate_dX_MF_ML1();  
  void final_integrate_dX_MF_Q1_ML1_Pair();  
  void final_integrate_dX_MF_Q1_ML1_N2N();  
  void final_integrate_F_ML1();  
  void final_integrate_QoI_ML1();  
  void final_integrate_Dyn_ML1();  

  void parse_xml_params_dX_MF(XMLElement *model_data_element); 
  void parse_xml_params_dX_MF_ML1(XMLElement *model_data_element);
  void parse_xml_params_dX_MF_Q1_ML1_Pair(XMLElement *model_data_element);
  void parse_xml_params_dX_MF_Q1_ML1_N2N(XMLElement *model_data_element);
  void parse_xml_params_F_ML1(XMLElement *model_data_element);
  void parse_xml_params_QoI_ML1(XMLElement *model_data_element);
  void parse_xml_params_Dyn_ML1(XMLElement *model_data_element);

  void parse_mask_input_str(const char *mask_input_str,int *mask_input_ptr,
                            int *mask_list_n_ptr,int **mask_list);
  
  void parse_mask_fix_str(const char *mask_str,int *mask_ptr);

  void print_tensor_2d(const char *name, at::Tensor t);
  void write_array_txt_1D(const char *filename,int n,double *v);
  void write_array_txt_2D(const char *filename,int m,int n,double **A);

  /* =========================== Variables =========================== */
  LAMMPS                              *lammps; /* lammps data */
  friend class FixMLMOD;   
  /* reference to the fix directly */  // friend class so can access all members  
  FixMLMOD                            *fixMLMOD; 
  
  int                                  mlmod_seed;
  RanMars                             *random; /* random number generator */

  int  model_type;
  string model_type_str;  // @@ current length limit 
  void *model_data;

  int  MODEL_TYPE_NULL;
  string MODEL_TYPE_STR_NULL;

  int  MODEL_TYPE_dX_MF;
  string MODEL_TYPE_STR_dX_MF;

  int  MODEL_TYPE_dX_MF_ML1;
  string MODEL_TYPE_STR_dX_MF_ML1;
  
  int MODEL_TYPE_dX_MF_Q1_ML1_Pair;
  string MODEL_TYPE_STR_dX_MF_Q1_ML1_Pair;

  int MODEL_TYPE_dX_MF_Q1_ML1_N2N;
  string MODEL_TYPE_STR_dX_MF_Q1_ML1_N2N;

  int MODEL_TYPE_F_ML1;
  string MODEL_TYPE_STR_F_ML1;

  int MODEL_TYPE_QoI_ML1;
  string MODEL_TYPE_STR_QoI_ML1;

  int MODEL_TYPE_Dyn_ML1;
  string MODEL_TYPE_STR_Dyn_ML1;

  typedef struct {
    double eta;
    double a;
    double epsilon;
  } ModelData_dX_MF_Type;

  typedef struct { // note make object references pointers (setup later)
    string *M_ii_filename;
    string *M_ij_filename;
    torch::jit::script::Module *M_ii_model;
    torch::jit::script::Module *M_ij_model;
  } ModelData_dX_MF_ML1_Type;
 
  typedef struct { // note make object references pointers (setup later)
    string *M_ii_filename;
    string *M_ij_filename;

    torch::jit::script::Module *M_ii_model;
    torch::jit::script::Module *M_ij_model;
    
    int    flag_stochastic;
    double KBT;

    int    flag_thm_drift;
    double delta;

    // --
    // for saving needed info between integration steps
    int      flag_init;

    int      num_II;
    int      II_size;
    int*     II;

    double*  xi_block;
    double** xi;
    int      xi_size;

    double*  Q_xi_block;
    double** Q_xi;
    int      Q_xi_size;

    double*  prev_v_block;
    double** prev_v;
    int      prev_v_size;
    
    double*  store_x_block;
    double** store_x;
    int      store_x_size;
    
    double*  avg_diff_M_block;
    double** avg_diff_M;
    int      avg_diff_M_size[2];
    
    double*  M_block;
    double** M;
    int      M_size[2];
    
    double*  Q_block;
    double** Q;
    int      Q_size[2];

    double*  A;
    int      A_size;
    
  } ModelData_dX_MF_Q1_ML1_Pair_Type;

  typedef struct { // note make object references pointers (setup later)
    string *M_filename;
    torch::jit::script::Module *M_model;
    
    int    flag_stochastic;
    double KBT;

    int    flag_thm_drift;
    double delta;

    // --
    // for saving needed info between integration steps
    int      flag_init;

    int      num_II;
    int      II_size;
    int*     II;

    double*  xi_block;
    double** xi;
    int      xi_size;

    double*  Q_xi_block;
    double** Q_xi;
    int      Q_xi_size;

    double*  prev_v_block;
    double** prev_v;
    int      prev_v_size;
    
    double*  store_x_block;
    double** store_x;
    int      store_x_size;
    
    double*  avg_diff_M_block;
    double** avg_diff_M;
    int      avg_diff_M_size[2];
    
    double*  M_block;
    double** M;
    int      M_size[2];
    
    double*  Q_block;
    double** Q;
    int      Q_size[2];

    double*  A;
    int      A_size;
    
  } ModelData_dX_MF_Q1_ML1_N2N_Type;

  typedef struct { // note make object references pointers (setup later)
    string *base_dir;
    string *base_name;

    string *F_filename;
    torch::jit::script::Module *F_model;

    string *mask_input_str; 
    
    string *mask_fix_str;

    // --
    // for saving needed info between integration steps
    int      flag_init;

    int      num_II;
    int      II_size;
    int*     II;

    int  mask_input;
    int  mask_list_n;
    int *mask_list;
    int  input_size;

    int mask_fix;

    int IN_X,IN_V,IN_F,IN_Type;

  } ModelData_F_ML1_Type;

  typedef struct { // note make object references pointers (setup later)
    string *base_dir;
    string *base_name;

    string *QoI_filename;
    torch::jit::script::Module *QoI_model;

    string *mask_input_str; 

    string *mask_fix_str;

    int skip_step; // how often to compute 

    // --
    // for saving needed info between integration steps
    int      flag_init;

    int      num_II;
    int      II_size;
    int*     II;

    int  mask_input;
    int  mask_list_n;
    int *mask_list;
    int  input_size;

    int mask_fix;

    int IN_X,IN_V,IN_F,IN_Type;

  } ModelData_QoI_ML1_Type;

  typedef struct { // note make object references pointers (setup later)
    string *base_dir;
    string *base_name;

    string *Dyn1_filename;
    string *Dyn2_filename;

    torch::jit::script::Module *Dyn1_model;
    torch::jit::script::Module *Dyn2_model;

    string *mask_input_str; 

    string *mask_fix_str;

    // --
    // for saving needed info between integration steps
    int      flag_init;

    int      num_II;
    int      II_size;
    int*     II;

    int  mask_input;
    int  mask_list_n;
    int *mask_list;
    int  input_size;

    int mask_fix;

    int IN_X,IN_V,IN_F,IN_Type;

  } ModelData_Dyn_ML1_Type;

};

}

#endif

