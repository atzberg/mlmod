/**
 * @file 
 *
 * @brief MLMOD library routines to drive the machine learning models 
 * and simulations.
 *
 * 
 */

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

#ifndef LMP_DRIVER_SELM_H
#define LMP_DRIVER_SELM_H

#ifndef FFT_FFTW /* determine if FFTW package specified */
  #warning "No FFTW package specified for SELM codes.  The SELM functionality will be disabled."
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

/**
 * @brief The driver of the functionality for the shared library class.  This
 * specifies the prototype for the API interface with lammps.
 * 
 */
struct DriverMLMOD {

 public:

  /* ================= function prototypes ================= */

  /** 
   * @brief Main constructor for mlmod package driver. 
   *
   * @param FixMLMOD*: reference to the mlmod-lammps "fix" interface  
   * @param LAMMPS*: reference to the running lammps instance
   * @param narg: number of command arguments
   * @param args: list of strings for the arguments 
   *
  */
  DriverMLMOD(class FixMLMOD *, class LAMMPS *, int, char **);

  /** 
   * @brief Constructor for empy object which is used primarily for testing. 
  */
  DriverMLMOD(); /* used to construct empty object,
                    primarily for testing purposes */

  /** The deconstructor. */
  ~DriverMLMOD();

  // =========================== Fix Related Function Calls ==================== */
  /** Lammps interface code for signaling the type of simulation. */
  int          setmask();

  /** Used for initializing the driver settings when created. */
  virtual void init();

  /** Used for setting up references in the driver after lammps is initialized. */
  void         setup(int vflag);

  /** Lammps calls this routine at the start of each time-step. */
  virtual void initial_integrate(int);

  /** Lammps calls this routine at the end of each time-step. */
  virtual void final_integrate();  

  /** Lammps calls this routine when the time-step size changes. */
  void         reset_dt();

  /** Lammps calls this routine after it finished computing forces. */
  void         post_force(int vflag);
  
  /** Lammps calls this routine before updating atom tables. */
  void pre_exchange();

  /** Lammps calls this at the end of a simulation step. */
  void end_of_step();

  /* =========================== MLMOD Function Calls =========================== */
  /** This is called from mlmod routines in lammps to initializes the driver
   * class attributes. */
  void init_attributes();

  /** This is called from mlmod routines in lammps when the "fix" is setup.. */
  void init_from_fix();

  /** Creates tensors for positions, velocity, and forces from lammps.  */
  void constr_input_masked(at::TensorAccessor<float,2,at::DefaultPtrTraits,long> input_m_a,
                           int num_indices,int *indices,
                           int mask_input, int mask_list_n, int *mask_list,
                           double **x,double **v,double **f, int *type);

  /** Parses the parameters from the xml files associated with the model_type.*/
  void parse_xml_params(char *filename);
  
  /** Writes additional simulation data to disk each time-step. */
  //void writeAllSimulationData(int timeIndex);

  /** Writes a file giving information about the current simulation for
   * post-processing scripts.*/
  void writeInfo();

  /** Writes a file for the final step of the simulation, see writeInfo().*/
  void writeFinalInfo();

  /* ===== case specific functions ===== */

  /** @brief Dynamic integrator with ML model for the mobility tensor \f$M(\mathbf{X})\f$
    for a pair of particles with <br/>  
    \f$ \left\{
     \begin{array}{lll}
      {d\mathbf{X}}/{dt} & = & \mathbf{M}(\mathbf{X})\mathbf{F} 
      + k_B{T}\nabla_X \cdot \mathbf{M}(\mathbf{X}) + \mathbf{g}, \\
      \langle \mathbf{g}(s) \mathbf{g}(t)^T \rangle & = & 2 k_B{T} \mathbf{M}(\mathbf{X}) \delta(t - s). \\
      \end{array} \right.
    \f$
  **/
  void initial_integrate_dX_MF_Q1_ML1_Pair();  

  /** @brief Dynamic integrator with ML model for the mobility tensor \f$M(\mathbf{X})\f$
    for pairwise interactions for a collection of particles with <br/>  
    \f$ \left\{
     \begin{array}{lll}
      {d\mathbf{X}}/{dt} & = & \mathbf{M}(\mathbf{X})\mathbf{F} 
      + k_B{T}\nabla_X \cdot \mathbf{M}(\mathbf{X}) + \mathbf{g}, \\
      \langle \mathbf{g}(s) \mathbf{g}(t)^T \rangle & = & 2 k_B{T} \mathbf{M}(\mathbf{X}) \delta(t - s). \\
      \end{array} \right.
    \f$
  **/
  void initial_integrate_dX_MF_Q1_ML1_N2N();  

  /** @brief Computes an ML model for the force.
   *
   **/
  void initial_integrate_F_ML1();  

  /** @brief Computes an ML model for for a quantity of interest (QoI).
   *
   **/
  void initial_integrate_QoI_ML1();  

  /** @brief Computes an ML model for particle dynamics. 
   *
   **/
  void initial_integrate_Dyn_ML1();  

  /** @brief Called at end of integration steps. */ 
  void final_integrate_dX_MF();  /**< See initial_integrate_dX_MF(). */
  void final_integrate_dX_MF_ML1();  /**< See initial_integrate_dX_MF_ML1(). */
  void final_integrate_dX_MF_Q1_ML1_Pair(); /**< See initial_integrate_dX_MF_Q1_ML1_Pair(). */
  void final_integrate_dX_MF_Q1_ML1_N2N();   /**< See initial_integrate_dX_MF_Q1_ML1_N2N(). */
  void final_integrate_F_ML1();   /**< See initial_integrate_F_ML1(). */
  void final_integrate_QoI_ML1();   /**< See initial_integrate_QoI_ML1(). */
  void final_integrate_Dyn_ML1();   /**< See initial_integrate_Dyn_ML1(). */

  /** @brief Parsers for the specific model cases. */ 
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

  /** @brief Display information.  */ 
  void print_tensor_2d(const char *name, at::Tensor t);

  /** @brief Writing tensor arrays to disk for processing and debugging.  */ 
  void write_array_txt_1D(const char *filename,int n,double *v);
  void write_array_txt_2D(const char *filename,int m,int n,double **A);

  /* =========================== Variables =========================== */
  LAMMPS                              *lammps;     /**< lammps instance */

  /** Declaration allows access to internal members for our lammps "fix" interface */
  friend class FixMLMOD; 

  FixMLMOD                            *fixMLMOD;   /**< our lammps "fix"
                                                     interface instance  */
  
  int                                  mlmod_seed; /**< seed for random number
                                                     generator  */
  RanMars                             *random;     /**< random number generator
                                                     */

  int  model_type;                                 /**< integer flag for the model type to use */
  string model_type_str;                           /**< name of the model type
                                                     for setting up integer
                                                     flags and displaying
                                                     information (note may be
                                                     current length limit) */
  void *model_data;                                /**< general model specific data */

  /** @brief Model types (constants) */
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

  /** @brief Data for hydrodynamic model case (used primarily for testing) */
  typedef struct {
    double eta;
    double a;
    double epsilon;
  } ModelData_dX_MF_Type;

  /** @brief Data for a basic ML mobility-based simulation  */
  typedef struct { // note make object references pointers (setup later)
    string *M_ii_filename; /**< filename for the ML model for the self-mobility tensor components */
    string *M_ij_filename; /**< filename for the ML model for pair-mobility tensor components */
    torch::jit::script::Module *M_ii_model; /**< torch ML model for self-mobility */
    torch::jit::script::Module *M_ij_model; /**< torch ML model for pair-mobility */
  } ModelData_dX_MF_ML1_Type;

  /** @brief Data for an overdamped stochastic ML mobility-based simulation, 
    see initial_integrate_dX_MF_Q1_ML1_Pair() and
    final_integrate_dX_MF_Q1_ML1_Pair().
    */
  typedef struct { // note make object references pointers (setup later)
    string *M_ii_filename;
    string *M_ij_filename;

    torch::jit::script::Module *M_ii_model;
    torch::jit::script::Module *M_ij_model;
    
    int    flag_stochastic; /*< flag for fluctuations */
    double KBT;             /*< themal energy */

    int    flag_thm_drift;  /*< flag for drift term */
    double delta;           /*< \f$\delta\f$ for drift calculation */

    // --
    // for saving needed info between integration steps
    
    /** internal variables */
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

  /** @brief State information for specified simulation modality. */
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

  /** @brief State information for specified simulation modality. */
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

  /** @brief State information for specified simulation modality. */
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

  /** @brief State information for specified simulation modality. */
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

