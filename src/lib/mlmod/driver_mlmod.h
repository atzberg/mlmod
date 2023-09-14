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
  P.J. Atzberger, Journal of Open Source Software, 8(89), 5620, (2023) 

  @article{mlmod_atzberger,
    author    = {Paul J. Atzberger},
    journal   = {Journal of Open Source Software}, 
    title     = {MLMOD: Machine Learning Methods for Data-Driven 
                 Modeling in LAMMPS},
    year      = {2023},  
    publisher = {The Open Journal},
    volume    = {8},
    number    = {89},
    pages     = {5620},
    note      = {http://atzberger.org},
    doi       = {10.21105/joss.05620},
    url       = {https://doi.org/10.21105/joss.05620}
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
  void destructor_dX_MF_Q1_ML1_Pair();

  // =========================== Fix Related Function Calls ==================== */
  /** Lammps interface code for signaling the type of simulation. */
  int          setmask();

  /** Used for initializing the driver settings when created. */
  virtual void init();

  /** Used for setting up references in the driver after lammps is initialized. */
  void         setup(int vflag);
  void setup_QoI_ML1(int vflag);

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
  void end_of_step_QoI_ML1();

  /** Lammps calls this to extract an array of data from the fix. */
  double compute_array(int i, int j);
  double compute_array_QoI_ML1(int i, int j);

  /* =========================== MLMOD Function Calls =========================== */
  /** This is called from mlmod routines in lammps to initializes the driver 
      class attributes. 
  **/
  void init_attributes();

  /** This is called from mlmod routines in lammps when the "fix" is setup. **/
  void init_from_fix();


  /** Parses the parameters from the xml files associated with the model_type. **/
  void parse_xml_params(char *filename);
  
  /** Parses the parameters from the xml files associated with the model_type. **/
  void params_parse(int narg, char **arg);

  /** Writes additional simulation data to disk each time-step. **/
  //void writeAllSimulationData(int timeIndex);

  /** Writes a file giving information about the current simulation for
      post-processing scripts. **/
  void write_info();

  /** Writes a file for the final step of the simulation, see write_info(). **/
  void write_final_info();

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

  /** @brief Computes an ML model for a force determined and applied collectively for all the particles 
      in the designated group \f$ \mathbf{X} = \{\mathbf{X}_i\}_{i \in \mathcal{G}} \f$, where
      \f$ F(\mathbf{X},\mathbf{V},\mathbf{F},\mathbf{I_T},t) \f$.
   **/
  void post_force_F_ML1();  

  /** @brief Computes an ML model for a force applied individually to each particle \f$ \mathbf{X}_i \f$,
      with 
      \f$ F_i(\mathbf{X}_i,\mathbf{V}_i,\mathbf{F}_i,\mathbf{I_T}_i,t) \f$.
      To compute forces collectively over the particles, see post_force_F_ML1().
   **/
  void post_force_F_X_ML1();  

  /** @brief Computes an ML model for a force applied on pairs of particles within the designated group, 
      \f$ \mathbf{X}_{ij} = (\mathbf{X}_i,\mathbf{X}_j) \f$ for $i,j \in \mathcal{G}$, with 
      \f$ F_{ij}(\mathbf{X}_{ij},\mathbf{V}_{ij},\mathbf{F}_{ij},\mathbf{I_T}_{ij},t) \f$.
      To compute forces collectively over the particles, see post_force_F_ML1().
   **/
  void post_force_F_Pair_ML1();  

  /** @brief Computes an ML model for a quantity of interest (QoI),
       \f$ Q(\mathbf{X},\mathbf{V},\mathbf{F}) \f$.
   **/
  void initial_integrate_QoI_ML1();  

  /** @brief Computes an ML model for the particle dynamics,
      \f$ \mathbf{X}^{n+1} = \Gamma(\mathbf{X}^n,\mathbf{V}^n,\mathbf{F}^n,t_n) \f$.
      Depending on the mask_fix settings this can be a single time-step or 
      Verlet-style integrator.  The initial step uses the machine learning model 
      specified in dyn1_filenmae and the final step the model in dyn2_filename.  
      See the documentation pages and examples for more details.  
   **/
  void initial_integrate_Dyn_ML1();  

  /** @brief Dynamic integrator with ML model for the mobility tensor \f$M(\mathbf{X})\f$
    for pairwise interactions for a collection of particles with <br/>  
    \f$ 
      {d\mathbf{X}}/{dt} = \mathbf{M}(\mathbf{X})\mathbf{F}. 
    \f$
  **/
  void final_integrate_dX_MF_ML1(); 

  /** @brief Dynamic integrator for a test mobility tensor \f$M(\mathbf{X})\f$
    for pairwise interactions for a collection of particles with <br/>  
    \f$ 
      {d\mathbf{X}}/{dt} = \mathbf{M}(\mathbf{X})\mathbf{F}. 
    \f$
  **/
  void final_integrate_dX_MF(); 

  void final_integrate_dX_MF_Q1_ML1_Pair(); /**< See initial_integrate_dX_MF_Q1_ML1_Pair(). **/
  void final_integrate_dX_MF_Q1_ML1_N2N();   /**< See initial_integrate_dX_MF_Q1_ML1_N2N(). **/

  void final_integrate_QoI_ML1();   /**< See initial_integrate_QoI_ML1(). **/

  /** @brief Dynamic integrator based on specified machine learning model.
    For a Verlet-style integrator this calls the second half of the integration 
    step using the model specified by dyn2_filenmae.   For more details, 
    see initial_integrate_Dyn_ML1().  
  **/
  void final_integrate_Dyn_ML1();


  /** @brief Parsers for the specific model cases. **/ 
  void parse_xml_params_dX_MF(XMLElement *model_data_element); 
  void parse_xml_params_dX_MF_ML1(XMLElement *model_data_element);
  void parse_xml_params_dX_MF_Q1_ML1_Pair(XMLElement *model_data_element);
  void parse_xml_params_dX_MF_Q1_ML1_N2N(XMLElement *model_data_element);
  void parse_xml_params_F_ML1(XMLElement *model_data_element);
  void parse_xml_params_F_X_ML1(XMLElement *model_data_element);
  void parse_xml_params_F_Pair_ML1(XMLElement *model_data_element);
  void parse_xml_params_QoI_ML1(XMLElement *model_data_element);
  void parse_xml_params_Dyn_ML1(XMLElement *model_data_element);

  void parse_mask_input_str(const char *mask_input_str,int *mask_input_ptr,
                            int *mask_list_n_ptr,int **mask_list);
  
  void parse_mask_fix_str(const char *mask_str,int *mask_ptr);

  /** 
   @brief Creates a vector array x[][] for storing double precision floating point values.

   @param num_cols: number of dimensions
   @param num_rows: number of indices
   @param db_vec_block: block of memory for vector (set to zero to allocate)
   @param db_vec_size: size of the block of memory allocated

   @return (double **) vector array with shape=[num_rows,num_cols].
   
  **/
  double** create_db_vec(int num_rows, int num_cols, 
                         double **db_vec_block, 
                         int *db_vec_size);

  /** 
   @brief Sets a vector array x[][] from torch tensor data a[:][0]. 
   
   This implements a copy of the form
   x[i][d] = a[I + offset][0], where I = i*num_cols + d
   x.shape=[n,num_cols], a.shape=[n*num_cols,1]

   @param double**: vector array, (typical shape=[n,num_cols])
   @param output_m: torch tensor 
   @param num_cols: number of dimensions
   @param num_rows: number of indices
   @param rowII: subset of indices to set (rowII=NULL, use all)
   @param offset_tensor: for tensor array, offset for start of values
   
  **/
  void set_db_vec_to_tensor_full(double **z,
                                 at::Tensor output_m,
                                 int num_rows, int num_cols, int *rowII, 
                                 int offset_tensor);

  /** 
   @brief Sets a vector array x[][] from torch tensor data a[:][0]. 
   
   This implements a copy of the form
   x[i][d] += a[I + offset][0], where I = i*num_cols + d
   x.shape=[n,num_cols], a.shape=[n*num_cols,1]

   @param double**: vector array, (typical shape=[n,num_cols])
   @param output_m: torch tensor 
   @param num_cols: number of dimensions
   @param num_rows: number of indices
   @param rowII: subset of indices to set (rowII=NULL, use all)
   @param offset_tensor: for tensor array, offset for start of values
   
  **/
  void add_tensor_full_to_db_vec(double **z,
                                 at::Tensor output_m,
                                 int num_rows, int num_cols, int *rowII, 
                                 int offset_tensor);

  /** 
   @brief Sets a vector array x[][] from torch tensor data a[:][0]
   which has the same overall size as x. 
   
   This implements a copy of the form
   x[iii][d] += a[I + offset][0], where I = iii*num_cols + d,
   iii = rowII[i], x.shape=[num_rows,num_cols], a.shape=[num_rows*num_cols,1].
   

   @param double**: vector array, (typical shape=[n,num_cols])
   @param output_m: torch tensor 
   @param num_cols: number of dimensions
   @param num_rows: number of indices
   @param rowII: subset of indices to set (rowII=NULL, use all)
   @param offset_tensor: for tensor array, offset for start of values
   
  **/
  void add_tensor_seg_to_db_vec(double **z,
                                at::Tensor output_m,
                                int num_rows, int num_cols, int *rowII, 
                                int offset_tensor);

  /** 
   @brief Sets a vector array x[][] to zero. 
   
   This implements a copy of the form
   x[i][d] = 0.0
   x.shape=[n,num_cols], a.shape=[n*num_cols,1]

   @param double**: vector array, (typical shape=[n,num_cols])
   @param num_cols: number of dimensions
   @param num_rows: number of indices
   @param rowII: subset of indices to set (rowII=NULL, use all)
   
  **/
  void set_to_zero_db_vec(double **z,
                          int num_rows, int num_cols, int *rowII);

  /** Get tensor size given the masks and groups from lammps.  **/
  int get_tensor_input_size(int num_II,int *II, int num_dim, 
                            int mask_input, int mask_list_n, int *mask_list,
                            int groupbit, int *mask_atom);

  /** Creates tensors for positions, velocity, and forces from lammps.  **/
  void build_tensor_from_masks(at::TensorAccessor<float,2,at::DefaultPtrTraits,long> input_m_a,
                               int num_indices,int *indices,
                               int mask_input, int mask_list_n, int *mask_list,
                               int groupbit, int *mask_atom,
                               double **x,double **v,double **f, int *type,double time);

  /** Get indices for group from lammps.  **/
  void get_indices_for_group(int groupbit,int *mask_atom, int nlocal, 
                             int *II_size_ptr, int **II_ptr, int *num_II_ptr);

  /** @brief Display information.  **/ 
  void print_tensor_2d(const char *name, at::Tensor t);

  /** @brief Writing tensor arrays to disk for processing and debugging.  **/ 
  void write_array_txt_1D(const char *filename,int n,double *v);
  void write_array_txt_2D(const char *filename,int m,int n,double **A);

  /** @brief Test codes helpful in debugging.  **/ 
  void test1();  
  void test2();
 
  /* =========================== Variables =========================== */
  LAMMPS                              *lammps;     /**< lammps instance */

  /** Declaration allows access to internal members for our lammps "fix" interface */
  friend class FixMLMOD; 
  friend class Fix; 
  friend class Update; 

  FixMLMOD                            *fixMLMOD;   /**< our lammps "fix"
                                                     interface instance  */
  
  int                                  mlmod_seed; /**< seed for random number
                                                     generator  */
  int                                  flag_verbose; /** level of output to print */
  string                               params_filename; /* parameter filename */

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

  int MODEL_TYPE_F_X_ML1;
  string MODEL_TYPE_STR_F_X_ML1;

  int MODEL_TYPE_F_Pair_ML1;
  string MODEL_TYPE_STR_F_Pair_ML1;

  int MODEL_TYPE_QoI_ML1;
  string MODEL_TYPE_STR_QoI_ML1;

  int MODEL_TYPE_Dyn_ML1;
  string MODEL_TYPE_STR_Dyn_ML1;

  int IN_X;
  string IN_X_str;

  int IN_V;
  string IN_V_str;

  int IN_F;
  string IN_F_str;

  int IN_Type;
  string IN_Type_str;

  int IN_Time;
  string IN_Time_str;

  int IN_num_types;

  /** @brief Data for hydrodynamic model case (used primarily for testing) */
  typedef struct {
    double eta;
    double a;
    double epsilon;

    /** internal variables */
    int      flag_init;   

    string *mask_input_str; 
    string *mask_fix_str;

    int  mask_input;
    int  mask_list_n;
    int *mask_list;
    int  input_size;

    int mask_fix;

  } ModelData_dX_MF_Type;

  /** @brief Data for a basic ML mobility-based simulation  */
  typedef struct { // note make object references pointers (setup later)
    string *M_ii_filename; /**< filename for the ML model for the self-mobility tensor components */
    string *M_ij_filename; /**< filename for the ML model for pair-mobility tensor components */

    torch::jit::Module *M_ii_model; /**< torch ML model for self-mobility */
    torch::jit::Module *M_ij_model; /**< torch ML model for pair-mobility */

    /** internal variables */
    int      flag_init;   

    string *mask_input_str; 
    string *mask_fix_str;

    int  mask_input;
    int  mask_list_n;
    int *mask_list;
    int  input_size;

    int mask_fix;

  } ModelData_dX_MF_ML1_Type;

  /** @brief Data for an overdamped stochastic ML mobility-based simulation, 
    see initial_integrate_dX_MF_Q1_ML1_Pair() and
    final_integrate_dX_MF_Q1_ML1_Pair().
    */
  typedef struct { // note make object references pointers (setup later)
    string *M_ii_filename;
    string *M_ij_filename;

    torch::jit::Module *M_ii_model;
    torch::jit::Module *M_ij_model;
    
    int    flag_stochastic; /*< flag for fluctuations */
    double KBT;             /*< themal energy */

    int    flag_thm_drift;  /*< flag for drift term */
    double delta;           /*< \f$\delta\f$ for drift calculation */


    // --
    // for saving needed info between integration steps
    /** internal variables */
    int      flag_init;   

    string *mask_input_str; 
    string *mask_fix_str;

    int  mask_input;
    int  mask_list_n;
    int *mask_list;
    int  input_size;

    int mask_fix;

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
    torch::jit::Module *M_model;
    
    int    flag_stochastic;
    double KBT;

    int    flag_thm_drift;
    double delta;

    // --
    // for saving needed info between integration steps
    /** internal variables */
    int      flag_init;   

    string *mask_input_str; 
    string *mask_fix_str;

    int  mask_input;
    int  mask_list_n;
    int *mask_list;
    int  input_size;

    int mask_fix;

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
    torch::jit::Module *F_model;

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

  } ModelData_F_ML1_Type;

  /** @brief State information for specified simulation modality. */
  typedef struct { // note make object references pointers (setup later)
    string *base_dir;
    string *base_name;

    string *F_filename;
    torch::jit::Module *F_model;

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

  } ModelData_F_X_ML1_Type;

  /** @brief State information for specified simulation modality. */
  typedef struct { // note make object references pointers (setup later)
    string *base_dir;
    string *base_name;

    string *F_filename;
    torch::jit::Module *F_model;

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

  } ModelData_F_Pair_ML1_Type;

  /** @brief State information for specified simulation modality. */
  typedef struct { // note make object references pointers (setup later)
    string *base_dir;
    string *base_name;

    string *qoi_filename;
    torch::jit::Module *QoI_model;

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

    int      flag_val_qoi;
    int      qoi_num_rows;
    int      qoi_num_cols;
    double*  val_qoi_block;
    double** val_qoi;
    int      val_qoi_size;

  } ModelData_QoI_ML1_Type;

  /** @brief State information for specified simulation modality. */
  typedef struct { // note make object references pointers (setup later)
    string *base_dir;
    string *base_name;

    string *dyn1_filename;
    string *dyn2_filename;

    torch::jit::Module *Dyn1_model;
    torch::jit::Module *Dyn2_model;

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

  } ModelData_Dyn_ML1_Type;

};

}

#endif

