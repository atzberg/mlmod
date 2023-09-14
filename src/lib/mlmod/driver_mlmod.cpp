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


  Note: lapack required version >= 3.7 (based on debian9 builds)

--------------------------------------------------------------------------------
*/

/* SELM_includes */
#include "driver_mlmod.h"

#include "tinyxml2.h"

// PyTorch C/C++ interface
#include <torch/torch.h>
#include <torch/script.h> // for loading models

// LaPack interface
#include <lapacke.h>  // C interface (avoid memory alloc issue)
//#include <lapacke/lapacke.h>  // C interface (avoid memory alloc issue) (Centos)

/* LAMMPS includes */
#include "atom.h"
#include "force.h"
#include "update.h"
#include "respa.h"
#include "error.h"
#include "comm.h"
#include "universe.h"
#include "version.h" 
#include "random_mars.h"
#include "citeme.h"

/* C/C++ includes */
#include <cstdlib>
#include <cstddef>
#include <cstdio>
#include <cstring>

#include <sys/stat.h>

using namespace LAMMPS_NS;
using namespace USER_MLMOD;
using namespace FixConst;
using namespace std;
using namespace tinyxml2;

static const char cite_mlmod_str[] =
  "MLMOD Package paper: https://doi.org/10.21105/joss.05620 \n"
  "@article{mlmod_atzberger,\n"
  "author    = {Paul J. Atzberger},\n"
  "journal   = {Journal of Open Source Software}, \n"
  "title     = {MLMOD: Machine Learning Methods for "
               "Data-Driven Modeling in LAMMPS},\n"
  "year      = {2023},  \n"
  "publisher = {The Open Journal},\n"
  "volume    = {8},\n"
  "number    = {89},\n"
  "pages     = {5620},\n"
  "note      = {http://atzberger.org},\n"
  "doi       = {10.21105/joss.05620},\n"
  "url       = {https://doi.org/10.21105/joss.05620}\n"
  "}\n\n";

/* =========================== Class definitions =========================== */
DriverMLMOD::DriverMLMOD() {

  cout << "-----------------------------------" << endl;
  cout << "Empty constructor case ()" << endl;
  cout << "MLMOD: Testing Torch routines." << endl;
  torch::Tensor tensor = torch::rand({2, 3});
  cout << "Tensor = " << tensor << endl;              
  cout << "-----------------------------------" << endl;

}

/* =========================== Class definitions =========================== */
DriverMLMOD::DriverMLMOD(FixMLMOD *fixMLMOD, LAMMPS *lmp, int narg, char **arg) {
  int II, flag;

  /* ================= init ================= */

  /* parse the fix parameters */
  params_parse(narg,arg);

  /* display setup message */
  if (flag_verbose >= 2) { 
    cout << "-----------------------------------" << endl; 
    cout << "Setting up an MLMOD model." << endl; 
    cout << "params_filename = " << params_filename << endl; 
  }

  /* add to citation collection */
  if (lmp->citeme) lmp->citeme->add(cite_mlmod_str);

  this->fixMLMOD = fixMLMOD;
  this->lammps = lmp;

  this->mlmod_seed = 1;
  this->random = new RanMars(lammps,this->mlmod_seed);  // WARNING: Be careful for MPI, since each seed the same (use comm->me) 

  // init constants
  II = 0;

  MODEL_TYPE_NULL = II; II++;
  MODEL_TYPE_STR_NULL = "NULL";

  MODEL_TYPE_dX_MF = II; II++;
  MODEL_TYPE_STR_dX_MF = "dX_MF";

  MODEL_TYPE_dX_MF_ML1 = II; II++;
  MODEL_TYPE_STR_dX_MF_ML1 = "dX_MF_ML1";

  MODEL_TYPE_dX_MF_Q1_ML1_Pair = II; II++;
  MODEL_TYPE_STR_dX_MF_Q1_ML1_Pair = "dX_MF_Q1_ML1_Pair";

  MODEL_TYPE_dX_MF_Q1_ML1_N2N = II; II++;
  MODEL_TYPE_STR_dX_MF_Q1_ML1_N2N = "dX_MF_Q1_ML1_N2N";
  
  MODEL_TYPE_F_ML1 = II; II++;
  MODEL_TYPE_STR_F_ML1 = "F_ML1";

  MODEL_TYPE_F_X_ML1 = II; II++;
  MODEL_TYPE_STR_F_X_ML1 = "F_X_ML1";

  MODEL_TYPE_F_Pair_ML1 = II; II++;
  MODEL_TYPE_STR_F_Pair_ML1 = "F_Pair_ML1";
  
  MODEL_TYPE_QoI_ML1 = II; II++;
  MODEL_TYPE_STR_QoI_ML1 = "QoI_ML1";
  
  MODEL_TYPE_Dyn_ML1 = II; II++;
  MODEL_TYPE_STR_Dyn_ML1 = "Dyn_ML1";

  this->model_type = MODEL_TYPE_NULL; 
  this->model_type_str = MODEL_TYPE_STR_NULL;

  /* constants for assumed ordering of mask_input */
  II = 0;

  IN_X_str = "X"; IN_X = II; II++;
  IN_V_str = "V"; IN_V = II; II++;
  IN_F_str = "F"; IN_F = II; II++;
  IN_Type_str = "Type"; IN_Type = II; II++;
  IN_Time_str = "Time"; IN_Time = II; II++;

  IN_num_types = II;

  /* run some basic tests (if activated) */
  flag = 0;  if (flag == 1) { test1(); }
  flag = 0;  if (flag == 1) { test2(); }

  /* parse the XML fiele */
  parse_xml_params((char *)params_filename.c_str());

  /* write some data to XML info file */
  write_info();

  /* how often fix called (default) */
  fixMLMOD->nevery = 1;

  if (flag_verbose >= 1) { 
    cout << "-----------------------------------" << endl;
  }

}

/* destructor */
DriverMLMOD::~DriverMLMOD() {

  if (model_type == MODEL_TYPE_dX_MF_Q1_ML1_Pair) {
    destructor_dX_MF_Q1_ML1_Pair();
  }

}

void DriverMLMOD::params_parse(int narg, char**arg) {

  /* check enough input parameters */
  if (narg < 4) {
    cout << "DriverMLMOD::params_parse(): Not enough parameters specified, narg = " << narg << endl;
    cout << endl;
    cout << "Usage of mlmod via a lammps fix:" << endl;
    cout << "fix <fix_id> <atom_group> mlmod <params_filename> (optional: verbose <level>)" << endl;
    cout << endl;
    cout << "For documentation, examples, and more information, see http://atzberger.org." << endl;
    exit(1);
  }

  /* get the parameter filename */
  params_filename = arg[3];

  /* check if flag_verbose set */
  if ((narg >= 6) && (strcmp(arg[4],"verbose") == 0)) {
    flag_verbose = atoi(arg[5]); /* arg[4]==verbose, arg[5]==<level> */ 
  } else {
    flag_verbose = 1;
  }

}

void DriverMLMOD::setup(int vflag) { /* lammps setup */

  // lammps calls this when setting up the integrator

  if (this->model_type == MODEL_TYPE_dX_MF_Q1_ML1_Pair) { 
  } else if (this->model_type == MODEL_TYPE_dX_MF_Q1_ML1_N2N) { 
  } else if (this->model_type == MODEL_TYPE_F_ML1) { 
  } else if (this->model_type == MODEL_TYPE_F_X_ML1) { 
  } else if (this->model_type == MODEL_TYPE_F_Pair_ML1) { 
  } else if (this->model_type == MODEL_TYPE_QoI_ML1) {
    setup_QoI_ML1(vflag);
  } else if (this->model_type == MODEL_TYPE_Dyn_ML1) { 
  } else {
    printf("DriverMLMOD::setup(), No recognized model_type = %d\n",model_type); 
  }

}

void DriverMLMOD::destructor_dX_MF_Q1_ML1_Pair() {

    ModelData_dX_MF_Q1_ML1_Pair_Type *m_data 
      = (ModelData_dX_MF_Q1_ML1_Pair_Type *) this->model_data;
    
    if (m_data->II_size > 0) {
      free(m_data->II);
    }
    
    if (m_data->store_x_size > 0) {
      free(m_data->store_x_block);
      free(m_data->store_x);
    }
    
    if (m_data->prev_v_size > 0) {
      free(m_data->prev_v_block);
      free(m_data->prev_v);
    }
    
    if (m_data->xi_size > 0) {
      free(m_data->xi_block);
      free(m_data->xi);
    }

    if (m_data->Q_xi_size > 0) {
      free(m_data->Q_xi_block);
      free(m_data->Q_xi);
    }

    if (m_data->avg_diff_M_size[0] > 0) {
      free(m_data->avg_diff_M_block);
      free(m_data->avg_diff_M);
    }

    if (m_data->M_size[0] > 0) {
      free(m_data->M_block);
      free(m_data->M);
    }

    if (m_data->Q_size[0] > 0) {
      free(m_data->Q_block);
      free(m_data->Q);
    }

    if (m_data->A_size > 0) {
      free(m_data->A);
    }

    free(m_data);
}


void DriverMLMOD::setup_QoI_ML1(int vflag) { 

    ModelData_QoI_ML1_Type *m_data 
      = (ModelData_QoI_ML1_Type *) this->model_data;

    // indicate compute 
    fixMLMOD->array_flag = 1;               // 0/1 if compute_array() function exists
    fixMLMOD->size_array_rows = 0;          // rows in global array
    fixMLMOD->size_array_cols = 1;          // columns in global array
    fixMLMOD->size_array_rows_variable = 0; // 1 if array rows is unknown in advance

    fixMLMOD->global_freq = 1;          // frequency s/v data is available at
    fixMLMOD->peratom_flag = 0;         // 0/1 if per-atom data is stored
    fixMLMOD->size_peratom_cols = 1;    // 0 = vector, N = columns in peratom array
    fixMLMOD->peratom_freq = 1;         // frequency per-atom data is available at

    fixMLMOD->local_flag = 0;           // 0/1 if local data is stored
    fixMLMOD->size_local_rows = 0;      // rows in local vector or array
    fixMLMOD->size_local_cols = 1;      // 0 = vector, N = columns in local array

    fixMLMOD->array_atom = NULL;        // computed per-atom array
    fixMLMOD->array_local = NULL;       // computed local array

    // initialize
    m_data->val_qoi = NULL;
    m_data->val_qoi_block = NULL;
    m_data->val_qoi_size = 0;

    // nvalid = next step on which end_of_step does something
    // add nvalid to all computes that store invocation times
    // since don't know a priori which are invoked by this fix
    // once in end_of_step() can set timestep for ones actually invoked
    //nvalid = nextvalid();
    //fixMLMOD->modify->addstep_compute_all(update->ntimestep);
}

void DriverMLMOD::end_of_step()
{
  int i,j;

  // lammps calls this after the integrator
  if (model_type == MODEL_TYPE_QoI_ML1) {
    end_of_step_QoI_ML1();
  } else {
    printf("DriverMLMOD::setup(), No recognized model_type = %d\n",model_type); 
  }

}

void DriverMLMOD::end_of_step_QoI_ML1()
{
  int i,j;

  ModelData_QoI_ML1_Type *m_data 
    = (ModelData_QoI_ML1_Type *) this->model_data;

  // skip if not step which requires doing something
  //bigint ntimestep = fixMLMOD->update->ntimestep;
  //if (ntimestep != nvalid) return;
  //nvalid_last = nvalid;

  // accumulate results of computes,fixes,variables to origin
  // compute/fix/variable may invoke computes so wrap with clear/add
  //modify->clearstep_compute();

  // fistindex = index in values ring of earliest time sample
  // nsample = number of time samples in values ring

  // nvalid += fixMLMOD->nevery;
  //modify->addstep_compute(nvalid);

}


/* ---------------------------------------------------------------------- */
int DriverMLMOD::setmask()
{
  /*
    mask |= INITIAL_INTEGRATE;
    mask |= FINAL_INTEGRATE;
   */

  //SELM_integrator_mask = INITIAL_INTEGRATE | FINAL_INTEGRATE;

  //return SELM_integrator_mask; /* syncronize the SELM mask with that returned to LAMMPS 

  //int mask = INITIAL_INTEGRATE | FINAL_INTEGRATE;
  int mask = 0;

  if (model_type == MODEL_TYPE_dX_MF) {
    mask = FINAL_INTEGRATE;
  } else if (model_type == MODEL_TYPE_dX_MF_ML1) {
    mask = FINAL_INTEGRATE;
  } else if (model_type == MODEL_TYPE_dX_MF_Q1_ML1_Pair) {
    mask = INITIAL_INTEGRATE | FINAL_INTEGRATE;
  } else if (model_type == MODEL_TYPE_dX_MF_Q1_ML1_N2N) {
    mask = INITIAL_INTEGRATE | FINAL_INTEGRATE;
  } else if (model_type == MODEL_TYPE_F_ML1) {
    ModelData_F_ML1_Type *m_data 
      = (ModelData_F_ML1_Type *) this->model_data;
    mask = POST_FORCE;
    //mask = m_data->mask_fix;
  } else if (model_type == MODEL_TYPE_F_X_ML1) {
    ModelData_F_X_ML1_Type *m_data 
      = (ModelData_F_X_ML1_Type *) this->model_data;
    mask = POST_FORCE;
    //mask = m_data->mask_fix;
  } else if (model_type == MODEL_TYPE_F_Pair_ML1) {
    ModelData_F_Pair_ML1_Type *m_data 
      = (ModelData_F_Pair_ML1_Type *) this->model_data;
    mask = POST_FORCE;
    //mask = m_data->mask_fix;
  } else if (model_type == MODEL_TYPE_QoI_ML1) {
    ModelData_QoI_ML1_Type *m_data 
      = (ModelData_QoI_ML1_Type *) this->model_data;
    mask = m_data->mask_fix;
  } else if (model_type == MODEL_TYPE_Dyn_ML1) {
    ModelData_Dyn_ML1_Type *m_data 
      = (ModelData_Dyn_ML1_Type *) this->model_data;
    mask = m_data->mask_fix;
  } else {
    // default case 
    mask = FINAL_INTEGRATE;
    printf("DriverMLMOD: setmask(), default, mask = %d\n",mask); 
  }

  if (flag_verbose >= 2) { 
    printf("MLMOD: setmask(), mask = %d\n",mask);
  }

  return mask;

}

/* ---------------------------------------------------------------------- */
void DriverMLMOD::pre_exchange()
{

}

/* ---------------------------------------------------------------------- */
double DriverMLMOD::compute_array(int i, int j)
{
  double val = -1.0;

  if (model_type == MODEL_TYPE_dX_MF) {
  } else if (model_type == MODEL_TYPE_dX_MF_ML1) {
  } else if (model_type == MODEL_TYPE_dX_MF_Q1_ML1_Pair) {
  } else if (model_type == MODEL_TYPE_dX_MF_Q1_ML1_N2N) {
  } else if (model_type == MODEL_TYPE_F_ML1) {
  } else if (model_type == MODEL_TYPE_F_X_ML1) {
  } else if (model_type == MODEL_TYPE_F_Pair_ML1) {
  } else if (model_type == MODEL_TYPE_QoI_ML1) {
    val = compute_array_QoI_ML1(i,j);
  } else if (model_type == MODEL_TYPE_Dyn_ML1) {
  } else {
    if (flag_verbose >= 1) {
      printf("DriverMLMOD::compute_array(): No recognized model_type = %d\n",model_type); 
    }
  }

  return val;

}

double DriverMLMOD::compute_array_QoI_ML1(int i, int j)
{

  double val;

  // get model data
  ModelData_QoI_ML1_Type *m_data 
    = (ModelData_QoI_ML1_Type *) this->model_data;

  val = m_data->val_qoi[i][j];

  return val; 

}

/* ---------------------------------------------------------------------- */

void DriverMLMOD::init()
{
  // WARNING: lammps does not seem to call this
 
  int b;

  b = 1;

}


void DriverMLMOD::initial_integrate(int vflag)
{

  if (this->model_type == MODEL_TYPE_dX_MF_Q1_ML1_Pair) { 
    initial_integrate_dX_MF_Q1_ML1_Pair();
  } else if (this->model_type == MODEL_TYPE_dX_MF_Q1_ML1_N2N) { 
    initial_integrate_dX_MF_Q1_ML1_N2N();
  } else if (this->model_type == MODEL_TYPE_F_ML1) { 
    //initial_integrate_F_ML1();
  } else if (this->model_type == MODEL_TYPE_F_X_ML1) { 
    //initial_integrate_F_X_ML1();
  } else if (this->model_type == MODEL_TYPE_F_Pair_ML1) { 
    //initial_integrate_F_Pair_ML1();
  } else if (this->model_type == MODEL_TYPE_QoI_ML1) { 
    initial_integrate_QoI_ML1();
  } else if (this->model_type == MODEL_TYPE_Dyn_ML1) { 
    initial_integrate_Dyn_ML1();
  } else {
    printf("WARNING: Not recognized, this->model_type_str = %s, this->model_type = %d \n",
    this->model_type_str.c_str(),this->model_type);
  }

}

/* ---------------------------------------------------------------------- */

void DriverMLMOD::final_integrate()
{

  if (this->model_type == MODEL_TYPE_dX_MF) {
    final_integrate_dX_MF();
  } else if (this->model_type == MODEL_TYPE_dX_MF_ML1) { 
    final_integrate_dX_MF_ML1();
  } else if (this->model_type == MODEL_TYPE_dX_MF_Q1_ML1_Pair) { 
    final_integrate_dX_MF_Q1_ML1_Pair();
  } else if (this->model_type == MODEL_TYPE_dX_MF_Q1_ML1_N2N) { 
    final_integrate_dX_MF_Q1_ML1_N2N();
  } else if (this->model_type == MODEL_TYPE_F_ML1) { 
    //final_integrate_F_ML1();
  } else if (this->model_type == MODEL_TYPE_F_X_ML1) { 
    //final_integrate_F_X_ML1();
  } else if (this->model_type == MODEL_TYPE_F_Pair_ML1) { 
    //final_integrate_F_Pair_ML1();
  } else if (this->model_type == MODEL_TYPE_QoI_ML1) { 
    final_integrate_QoI_ML1();
  } else if (this->model_type == MODEL_TYPE_Dyn_ML1) { 
    final_integrate_Dyn_ML1();
  } else {
    printf("WARNING: Not recognized, this->model_type_str = %s, this->model_type = %d \n",
    this->model_type_str.c_str(),this->model_type);
  }

}

/* ---------------------------------------------------------------------- */
void DriverMLMOD::final_integrate_dX_MF()
{

  // get atom info v
  Atom *atom = this->lammps->atom;
  Update *update = this->lammps->update;
  Force *force = this->lammps->force;

  int igroup = this->fixMLMOD->igroup; // may need to reference back to fixMLMOD
   
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask_atom = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  int groupbit = this->fixMLMOD->groupbit;

  //printf("this->model_type = %d \n",this->model_type);
  ModelData_dX_MF_Type *m_data = (ModelData_dX_MF_Type *) this->model_data;

  /* compute the particle motion using, dX/dt = V = M(X)*F */
  
  /* loop over atoms with specified mask */

  /* compute the velocity V = M*F step */

  /* update the atom location X^{n+1} = X^{n} + dt*V */

  /* double dtf = 0.5*update->dt*force->ftm2v; */
  //double dtf = update->dt*force->ftm2v;

  /* collect the list of indices matching the mask */

  /* loop over the index pairs to compute all-to-all interactions,
     ideally do this with a oct-tree search similar to pair 
     interactions in LAMPPS */

  /* for now we just loop over all pairs */

  if (rmass) {
    double m;
    m = 1;     
  } else {
    double m;
    m = 1;
  }

  double vecX[3],r_ij,r_ij_sq,M_ij[3][3];
  double delta_kd;
  double dt = update->dt;

  int num_dim = 3; 
 
  double pi,eta,a,epsilon;
  pi = M_PI; eta = m_data->eta; a = m_data->a; // @@@ put in correct constants
  epsilon = m_data->epsilon;
  double prefactor,M_ij_s;

  // initialize the velocity to zero
  for (int i = 0; i < nlocal; i++) {
    for (int d = 0; d < num_dim; d++) {
      v[i][d] = 0.0;   
    }
  }

  // compute the mobility contributions
  for (int i = 0; i < nlocal; i++) {
    for (int j = i; j < nlocal; j++) {

      if ((mask_atom[i] & groupbit) && (mask_atom[j] & groupbit)) {

        if (i == j) {
          M_ij_s = 1.0/(6.0*pi*eta*a);
          for (int d = 0; d < num_dim; d++) {
            // self interaction
            v[i][d] += M_ij_s*f[i][d];
          }

        } else { // i != j 
               
          // construct M_ij(X) block which is 6x6
          r_ij_sq = 0.0;
          for (int d = 0; d < num_dim; d++) {
            vecX[d] = x[i][d] - x[j][d];
            r_ij_sq += vecX[d]*vecX[d];
          }

          r_ij = sqrt(r_ij_sq);
          //r_ij_eps = r_ij + epsilon;
          //r_ij_eps_sq = r_ij_eps*r_ij_eps;
        
          prefactor = 1.0/(8.0*eta*(r_ij + epsilon));

          for (int d = 0; d < num_dim; d++) {

            // self interaction (already handled for i == j case)
            //M_ij[d][d] += 1.0/(6.0*pi*eta*a);  
            //M_ij[num_dim + d][num_dim + d] +== 1.0/(6.0*pi*eta*a); 

            // lateral interactions (Oseen tensor)
            for (int k = 0; k < num_dim; k++) {
              if (k == d) {delta_kd = 1.0;} else {delta_kd = 0.0;}
              M_ij[d][k] = prefactor*(delta_kd + (vecX[d]*vecX[k]/(r_ij_sq + 1e-12)));
            }

          }

          // compute contributions to V_i, V_j            
          for (int d = 0; d < num_dim; d++) {
            for (int k = 0; k < num_dim; k++) {
              v[i][d] += M_ij[d][k]*f[j][k];
              v[j][d] += M_ij[d][k]*f[i][k];
            }
          }
          
        } // i != j

      } // check mask
    } // loop j
  } // loop i

  // update the position                  
  for (int i = 0; i < nlocal; i++) {
    if (mask_atom[i] & groupbit) { 
      x[i][0] += dt*v[i][0];
      x[i][1] += dt*v[i][1];
      x[i][2] += dt*v[i][2];
    }
  }

}

/* ---------------------------------------------------------------------- */
void DriverMLMOD::final_integrate_dX_MF_ML1()
{

  // get atom info v
  Atom *atom = this->lammps->atom;
  Update *update = this->lammps->update;
  Force *force = this->lammps->force;

  int igroup = this->fixMLMOD->igroup; // may need to reference back to fixMLMOD
   
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask_atom = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;
  int num_dim = 3;

  double dt = update->dt;

  int groupbit = this->fixMLMOD->groupbit;

  double M_ii[3][3];
  double M_ij[3][3];

  // create tensors
  at::Tensor input1_m = torch::zeros({3,1});
  auto input1_m_a = input1_m.accessor<float,2>(); 
  std::vector<torch::jit::IValue> input1_m_i{input1_m};
  //auto input1_m_i = std::vector<torch::jit::IValue>{torch::jit::IValue::from(input1_m)};
  // std::vector<torch::jit::IValue>{torch::jit::IValue::from(my_array.data())};

  at::Tensor input2_m = torch::zeros({6,1});
  auto input2_m_a = input2_m.accessor<float,2>(); 
  std::vector<torch::jit::IValue> input2_m_i{input2_m};
  //auto input2_m_i = std::vector<torch::jit::IValue>{torch::jit::IValue::from(input2_m)};

  at::Tensor output_m; // leave un-initialized

  // use .pt model from PyTorch for ML learned model for M(X)
  // get model data
  ModelData_dX_MF_ML1_Type *m_data 
    = (ModelData_dX_MF_ML1_Type *) this->model_data;

  // initialize the velocity to zero
  for (int i = 0; i < nlocal; i++) {
    if (mask_atom[i] & groupbit) {
      for (int d = 0; d < num_dim; d++) {
        v[i][d] = 0.0;   
      }
    }
  }

  // loop over the points and apply the tensor
  
  // compute the mobility contributions
  for (int i = 0; i < nlocal; i++) {
    for (int j = i; j < nlocal; j++) {

      if ((mask_atom[i] & groupbit) && (mask_atom[j] & groupbit)) {

        if (i == j) { // self-mobility response

          // set the xi tensor 
          for (int d = 0; d < num_dim; d++) {
            input1_m_a[d][0] = x[i][d];
            //input1_m.index_put_({0,d},x[i][d]);
          }

          // evaluate the model M(X) and obtain tensor output
          output_m = m_data->M_ii_model->forward(input1_m_i).toTensor();
          // accessor for the 2D tensor 1x18
          auto output_m_a = output_m.accessor<float,2>();

          // reshape into the needed M tensor components
          for (int d = 0; d < num_dim; d++) {
            for (int k = 0; k < num_dim; k++) {
              int I1 = num_dim*d + k; 
              M_ii[d][k] = output_m_a[I1][0]; 
            }
          }

          // compute contributions to V_i, V_j
          for (int d = 0; d < num_dim; d++) {
            for (int k = 0; k < num_dim; k++) {
              v[i][d] += M_ii[d][k]*f[i][k];
            }
          }

        } else { // pair-mobility response

          // set the xi and xj tensors 
          for (int d = 0; d < num_dim; d++) {
            input2_m_a[0][d] = x[i][d];
            input2_m_a[0][num_dim + d] = x[j][d];
            //input2_m.index_put_({0,d},x[i][d]);
            //input2_m.index_put_({0,num_dim + d},x[j][d]);
          }

          // evaluate the model M(X) and obtain tensor output
          output_m = m_data->M_ij_model->forward(input2_m_i).toTensor();
          // accessor for the 2D tensor 1x18
          auto output_m_a = output_m.accessor<float,2>(); 

          // reshape into the needed M tensor components
          for (int d = 0; d < num_dim; d++) {
            for (int k = 0; k < num_dim; k++) {
              int I1 = num_dim*d + k; 
              M_ij[d][k] = output_m_a[I1][0]; 
            }
          }

          // compute contributions to V_i, V_j
          // assumes M_ij = M_{ji}^T symmetric
          for (int d = 0; d < num_dim; d++) {
            for (int k = 0; k < num_dim; k++) {
              v[i][d] += M_ij[d][k]*f[j][k];
              v[j][d] += M_ij[d][k]*f[i][k]; 
            }
          }

        } // end else

      } // mask check

    } // j loop
  } // i loop 

  // update the position                  
  for (int i = 0; i < nlocal; i++) {
    if (mask_atom[i] & groupbit) { 
      x[i][0] += dt*v[i][0];
      x[i][1] += dt*v[i][1];
      x[i][2] += dt*v[i][2];
    }
  }
    
}

/* ---------------------------------------------------------------------- */
void DriverMLMOD::initial_integrate_dX_MF_Q1_ML1_Pair() {

  // get atom info v
  Atom *atom = this->lammps->atom;
  Update *update = this->lammps->update;
  Force *force = this->lammps->force;

  int igroup = this->fixMLMOD->igroup; // may need to reference back to fixMLMOD
   
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask_atom = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;
  int num_dim = 3;

  double dt = update->dt;

  int groupbit = this->fixMLMOD->groupbit;

  int flag = 0;

  // use .pt model from PyTorch for ML learned model for M(X)
  // get model data
  ModelData_dX_MF_Q1_ML1_Pair_Type *m_data 
    = (ModelData_dX_MF_Q1_ML1_Pair_Type *) this->model_data;

  // initialize
  if (m_data->flag_init == 0) {

    // determine index list controlled by this fix 
    // get the II indices for the group
    m_data->II_size = 0; m_data->II = NULL; 
    get_indices_for_group(groupbit,mask_atom,nlocal,&m_data->II_size,&m_data->II,&m_data->num_II);

    // get tensor input size given masks
    m_data->input_size = get_tensor_input_size(m_data->num_II,m_data->II,num_dim,
                                               m_data->mask_input, m_data->mask_list_n, m_data->mask_list,
                                               groupbit, mask_atom);

    // alloc data for A matrix for lapack use
    if (m_data->A_size == 0) {
      m_data->A 
        = (double *)malloc(sizeof(double)*m_data->num_II
                                         *num_dim*m_data->num_II*num_dim);
      m_data->A_size = m_data->num_II*num_dim*m_data->num_II*num_dim;
    }

    // setup data structure for saving Q\xi 
    // if not yet allocated 
    if (m_data->prev_v_size == 0) {
      m_data->prev_v_block 
        = (double *)malloc(sizeof(double)*m_data->num_II*num_dim);
      m_data->prev_v = (double **)malloc(sizeof(double)*m_data->num_II);
      for (int i = 0; i < m_data->num_II; i++) {
        m_data->prev_v[i] = m_data->prev_v_block 
                          + i*num_dim + 0; // row-major blocking
        for (int d = 0; d < num_dim; d++) {
          m_data->prev_v[i][d] = 0.0;
        }
      }
      m_data->prev_v_size = m_data->num_II;
    }

    if (m_data->store_x_size == 0) {
      m_data->store_x_block 
        = (double *)malloc(sizeof(double)*m_data->num_II*num_dim);
      m_data->store_x = (double **)malloc(sizeof(double)*m_data->num_II);
      for (int i = 0; i < m_data->num_II; i++) {
        m_data->store_x[i] = m_data->store_x_block 
                           + i*num_dim + 0; // row-major blocking
        for (int d = 0; d < num_dim; d++) {
          m_data->store_x[i][d] = 0.0;
        }
      }
      m_data->store_x_size = m_data->num_II;
    }

    if ((m_data->xi_size == 0) && (m_data->flag_stochastic)) {
      m_data->xi_block 
        = (double *)malloc(sizeof(double)*m_data->num_II*num_dim);
      m_data->xi = (double **)malloc(sizeof(double)*m_data->num_II);
      for (int i = 0; i < m_data->num_II; i++) {
        m_data->xi[i] = m_data->xi_block 
                      + i*num_dim + 0; // row-major blocking
        for (int d = 0; d < num_dim; d++) {
          m_data->xi[i][d] = 0.0;
        }
      }
      m_data->xi_size = m_data->num_II;
    }

    if ((m_data->Q_xi_size == 0) && (m_data->flag_stochastic)) {
      m_data->Q_xi_block 
        = (double *)malloc(sizeof(double)*m_data->num_II*num_dim);
      m_data->Q_xi = (double **)malloc(sizeof(double)*m_data->num_II);
      for (int i = 0; i < m_data->num_II; i++) {
        m_data->Q_xi[i] = m_data->Q_xi_block 
                        + i*num_dim + 0; // row-major blocking
        for (int d = 0; d < num_dim; d++) {
          m_data->Q_xi[i][d] = 0.0;
        }
      }
      m_data->Q_xi_size = m_data->num_II;
    }

    if (((m_data->avg_diff_M_size[0] == 0) || 
         (m_data->avg_diff_M_size[1] == 0)) &&
         (m_data->flag_stochastic) && (m_data->flag_thm_drift)) {
      m_data->avg_diff_M_block 
        = (double *)malloc(sizeof(double)*m_data->num_II*num_dim
                                         *m_data->num_II*num_dim);
      m_data->avg_diff_M
        = (double **)malloc(sizeof(double)*m_data->num_II*num_dim);
      for (int i = 0; i < m_data->num_II*num_dim; i++) {
        m_data->avg_diff_M[i] 
          = m_data->avg_diff_M_block 
          + i*m_data->num_II*num_dim + 0; // row-major blocking
        for (int d = 0; d < m_data->num_II*num_dim; d++) {
          m_data->avg_diff_M[i][d] = 0.0;
        }
      }
      m_data->avg_diff_M_size[0] = m_data->num_II*num_dim; 
      m_data->avg_diff_M_size[1] = m_data->num_II*num_dim;
    }

    if ((m_data->M_size[0] == 0) || (m_data->M_size[1] == 0)) {
      m_data->M_block 
        = (double *)malloc(sizeof(double)*m_data->num_II*num_dim
                                         *m_data->num_II*num_dim);
      m_data->M = (double **)malloc(sizeof(double)*m_data->num_II*num_dim);
      for (int i = 0; i < m_data->num_II*num_dim; i++) {
        m_data->M[i] 
          = m_data->M_block 
          + i*m_data->num_II*num_dim + 0; // row-major blocking
        for (int d = 0; d < m_data->num_II*num_dim; d++) {
          m_data->M[i][d] = 0.0;
        }
      }
      m_data->M_size[0] = m_data->num_II*num_dim; 
      m_data->M_size[1] = m_data->num_II*num_dim;
    }

    if (((m_data->Q_size[0] == 0) || (m_data->Q_size[1] == 0)) 
          && (m_data->flag_stochastic)) {
      m_data->Q_block 
        = (double *)malloc(sizeof(double)*m_data->num_II*num_dim
                                         *m_data->num_II*num_dim);
      m_data->Q = (double **)malloc(sizeof(double)*m_data->num_II*num_dim);
      for (int i = 0; i < m_data->num_II*num_dim; i++) {
        m_data->Q[i] 
          = m_data->Q_block + i*m_data->num_II*num_dim + 0; // row-major blocking
        for (int d = 0; d < m_data->num_II*num_dim; d++) {
          m_data->Q[i][d] = 0.0;
        }
      }
      m_data->Q_size[0] = m_data->num_II*num_dim; 
      m_data->Q_size[1] = m_data->num_II*num_dim;
    }

    if (m_data->flag_thm_drift) {
      // divergence KBT*div*M(X) treated as zero for now
      // (assumes small enough variations in X for now)
      cout << "WARNING: KB*T*Div(M) term treated as 0.0 for now.\n"; 
    }

    m_data->flag_init = 1;

  } // end flag_init

  // create tensors
  at::Tensor input1_m = torch::zeros({num_dim,1});
  auto input1_m_a = input1_m.accessor<float,2>(); 
  std::vector<torch::jit::IValue> input1_m_i{input1_m};

  at::Tensor input2_m = torch::zeros({2*num_dim,1});
  auto input2_m_a = input2_m.accessor<float,2>(); 
  std::vector<torch::jit::IValue> input2_m_i{input2_m};

  at::Tensor output_m; // leave un-initialized

  // initialize the velocity to zero
  set_to_zero_db_vec(v,m_data->num_II,num_dim,m_data->II);
  //for (int i = 0; i < m_data->num_II; i++) {
  //  int iii = m_data->II[i];
  //  for (int d = 0; d < num_dim; d++) {
  //    v[iii][d] = 0.0;   
  //  }
  //}


  // construct the mobility tensor M (@optimize later using neigh list thresholds)
  for (int i = 0; i < m_data->num_II; i++) {
    int iii = m_data->II[i];
    for (int j = 0; j < m_data->num_II; j++) {
      int jjj = m_data->II[j];

      if (i == j) { // self-mobility response

        // set the xi tensor 
        for (int d = 0; d < num_dim; d++) {
          input1_m_a[d][0] = x[iii][d];
          //input1_m.index_put_({0,d},x[i][d]);
        }

        // evaluate the model M(X) and obtain tensor output
        output_m = m_data->M_ii_model->forward(input1_m_i).toTensor();
        // accessor for the 2D tensor 1x18
        auto output_m_a = output_m.accessor<float,2>();

        // reshape into the needed M tensor components
        for (int d = 0; d < num_dim; d++) {
          for (int k = 0; k < num_dim; k++) {
            int I1 = num_dim*d + k; 
            m_data->M[i*num_dim + d][j*num_dim + k] = output_m_a[I1][0]; 
          }
        }

      } else { // i != j

        // set the xi and xj tensors 
        for (int d = 0; d < num_dim; d++) {
          input2_m_a[d][0] = x[iii][d];
          input2_m_a[num_dim + d][0] = x[jjj][d];
          //input2_m.index_put_({0,d},x[i][d]);
          //input2_m.index_put_({0,num_dim + d},x[j][d]);
        }

        // evaluate the model M(X) and obtain tensor output
        output_m = m_data->M_ij_model->forward(input2_m_i).toTensor();
        // accessor for the 2D tensor 1x18 
        auto output_m_a = output_m.accessor<float,2>(); 

        // reshape into the needed M tensor components
        for (int d = 0; d < num_dim; d++) {
          for (int k = 0; k < num_dim; k++) {
            int I1 = num_dim*d + k; 
            m_data->M[i*num_dim + d][j*num_dim + k] = output_m_a[I1][0]; 
          }
        }

      } // end else
      
    } // j
  } // i


  // compute the cholesky factor QQ^T = 2*KB*T*M/dt
  // @optimize for only subset of particles in group  
  if (m_data->flag_stochastic) {
    int factor_type = 1;
    if (factor_type == 1) {
      int nn = m_data->num_II*num_dim; char uplo = 'L'; 
      int info = 0; int lda = m_data->num_II*num_dim;
      double *A; double **M; double **Q;
      double two_KBT_o_dt = 2.0*m_data->KBT/dt;
      A = m_data->A; M = m_data->M; Q = m_data->Q;

      for (int ii = 0; ii < m_data->num_II*num_dim; ii++) {
        for (int jj = 0; jj < m_data->num_II*num_dim; jj++) {
          A[jj*m_data->num_II*num_dim + ii] 
            = two_KBT_o_dt*M[ii][jj]; // put in column-major format
        }
      }

      // lapack cholesky factorization (done in-place, returns only lower triangular part)
      // dpotrf_(&uplo,&nn,A,&lda,&info);
      info = LAPACKE_dpotrf2(LAPACK_COL_MAJOR,uplo,nn,A,lda);

      // test for error
      assert(info == 0);

      // convert to Q (assumes Q lower tri and upper already initialized to zero) 
      for (int ii = 0; ii < m_data->num_II*num_dim; ii++) {
        for (int jj = 0; jj <= ii; jj++) {
          Q[ii][jj] = A[jj*m_data->num_II*num_dim + ii]; // A is column-major format
        }
      }

    } // factor_type

    // compute Q*xi (assumes only lower triangular part is non-zero) 
    for (int ii = 0; ii < m_data->num_II*num_dim; ii++) {
      m_data->xi_block[ii] = (double) this->random->gaussian(); // normal random number 
    }
    for (int ii = 0; ii < m_data->num_II*num_dim; ii++) {
      m_data->Q_xi_block[ii] = 0.0;
      // generate random xi and multiply by Q
      for (int jj = 0; jj <= ii; jj++) {
        m_data->Q_xi_block[ii] += m_data->Q[ii][jj]*m_data->xi_block[jj];
      }
    }

  } // flag_stochastic

  // compute the velocity update
  // compute contributions to V_i
  for (int i = 0; i < m_data->num_II; i++) {
    int iii = m_data->II[i];

    // particle j force contributions
    for (int j = 0; j < m_data->num_II; j++) {
      int jjj = m_data->II[j];
      // compute V_i = M_ij*f_j 
      for (int d = 0; d < num_dim; d++) {
        for (int k = 0; k < num_dim; k++) {
          v[iii][d] += m_data->M[i*num_dim + d][j*num_dim + k]*f[jjj][k]; 
        }
      }
    } // j

    // stochastic contributions
    if (m_data->flag_stochastic) {
      int iii = m_data->II[i];
      for (int d = 0; d < num_dim; d++) {
        v[iii][d] += m_data->Q_xi[i][d];
      }
    } // flag_stochastic

  } // i

  if ((m_data->flag_stochastic) && (m_data->flag_thm_drift)) {
    cout << "WARNING: flag_thm_drift not yet implemented in prototype code.\n";
    // compute the perturbations avg_diff_M
    // @@@
    // for (int ii = 0; ii < m_data->num_II*num_dim; ii++) {
    //  m_data->Q_xi_block[ii] = 0.0;
    //  // generate random xi and multiply by Q
    //  for (int jj = 0; jj <= ii; jj++) {
    //    m_data->xi_block[jj] 
    //      = (double) this->random->gaussian(); // normal random number 
    //    m_data->Q_xi_block[ii] += m_data->Q[ii][jj]*m_data->xi_block[jj];
    //  }
    //}
  }

  // save to update from the initial position x^n to x^{n+1}
  for (int i = 0; i < m_data->num_II; i++) {
    int iii = m_data->II[i];

    // save the baseline x^n 
    m_data->store_x[i][0] = x[iii][0];
    m_data->store_x[i][1] = x[iii][1];
    m_data->store_x[i][2] = x[iii][2];

    // compute the tilde_x^{n+1} for intermediate force calculations
    x[iii][0] += dt*v[iii][0];
    x[iii][1] += dt*v[iii][1];
    x[iii][2] += dt*v[iii][2];
    
  }
    
  // save the velocity for next step of integrator
  for (int i = 0; i < m_data->num_II; i++) {
    int iii = m_data->II[i];
    
    for (int d = 0; d < num_dim; d++) {
      m_data->prev_v[i][d] = v[iii][d];   
    }
    
  } // i

  // debug
  flag = 0; 
  if (flag) {
    write_array_txt_2D("M.txt",
                       m_data->M_size[0],m_data->M_size[1],m_data->M);
    write_array_txt_2D("Q.txt",
                       m_data->Q_size[0],m_data->Q_size[1],m_data->Q);
    write_array_txt_2D("store_x.txt",nlocal,num_dim,m_data->store_x);
    write_array_txt_2D("v.txt",nlocal,num_dim,v);
    write_array_txt_2D("f.txt",nlocal,num_dim,f);
  }
}

/* ---------------------------------------------------------------------- */
void DriverMLMOD::final_integrate_dX_MF_Q1_ML1_Pair()
{

  // get atom info v
  Atom *atom = this->lammps->atom;
  Update *update = this->lammps->update;
  Force *force = this->lammps->force;

  int igroup = this->fixMLMOD->igroup; // may need to reference back to fixMLMOD
   
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask_atom = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;
  int num_dim = 3;

  double dt = update->dt;

  int groupbit = this->fixMLMOD->groupbit;

  // create tensors
  at::Tensor input1_m = torch::zeros({num_dim,1});
  auto input1_m_a = input1_m.accessor<float,2>(); 
  std::vector<torch::jit::IValue> input1_m_i{input1_m};

  at::Tensor input2_m = torch::zeros({2*num_dim,1});
  auto input2_m_a = input2_m.accessor<float,2>(); 
  std::vector<torch::jit::IValue> input2_m_i{input2_m};

  at::Tensor output_m; // leave un-initialized

  // use .pt model from PyTorch for ML learned model for M(X)
  // get model data
  ModelData_dX_MF_Q1_ML1_Pair_Type *m_data 
    = (ModelData_dX_MF_Q1_ML1_Pair_Type *) this->model_data;

  // initialize the velocity to zero
  for (int i = 0; i < m_data->num_II; i++) {
    int iii = m_data->II[i];
    for (int d = 0; d < num_dim; d++) {
      v[iii][d] = 0.0;   
    }
  }

  // construct the mobility tensor M (@optimize later to neighbor subset)
  for (int i = 0; i < m_data->num_II; i++) {
    int iii = m_data->II[i];
    for (int j = 0; j < m_data->num_II; j++) {
      int jjj = m_data->II[j];

      if (i == j) { // self-mobility response

        // set the xi tensor 
        for (int d = 0; d < num_dim; d++) {
          input1_m_a[d][0] = x[iii][d];
          //input1_m.index_put_({0,d},x[i][d]);
        }

        // evaluate the model M(X) and obtain tensor output
        output_m = m_data->M_ii_model->forward(input1_m_i).toTensor();
        // accessor for the 2D tensor 1x18
        auto output_m_a = output_m.accessor<float,2>();

        // reshape into the needed M tensor components
        for (int d = 0; d < num_dim; d++) {
          for (int k = 0; k < num_dim; k++) {
            int I1 = num_dim*d + k; 
            m_data->M[i*num_dim + d][j*num_dim + k] = output_m_a[I1][0]; 
          }
        }

      } else { // i != j

        // set the xi and xj tensors 
        for (int d = 0; d < num_dim; d++) {
          input2_m_a[d][0] = x[iii][d];
          input2_m_a[num_dim + d][0] = x[jjj][d];
          //input2_m.index_put_({0,d},x[i][d]);
          //input2_m.index_put_({0,num_dim + d},x[j][d]);
        }

        // evaluate the model M(X) and obtain tensor output
        output_m = m_data->M_ij_model->forward(input2_m_i).toTensor();
        // accessor for the 2D tensor 1x18
        auto output_m_a = output_m.accessor<float,2>(); 

        // reshape into the needed M tensor components
        for (int d = 0; d < num_dim; d++) {
          for (int k = 0; k < num_dim; k++) {
            int I1 = num_dim*d + k; 
            m_data->M[i*num_dim + d][j*num_dim + k] = output_m_a[I1][0]; 
          }
        }

      } // end else
    
    } // j
  } // i

  // compute the velocity update
  // compute contributions to V_i
  for (int i = 0; i < m_data->num_II; i++) {
    int iii = m_data->II[i];

    // particle j contributions
    for (int j = 0; j < m_data->num_II; j++) {
      int jjj = m_data->II[j];

      // compute V_i = M_ij*f_j
      for (int d = 0; d < num_dim; d++) {
        for (int k = 0; k < num_dim; k++) {          
          v[iii][d] += m_data->M[i*num_dim + d][j*num_dim + k]*f[jjj][k]; 
        }          
      }

    } // j 

    // stochastic contributions
    if (m_data->flag_stochastic) {
      int iii = m_data->II[i];
      for (int d = 0; d < num_dim; d++) {
        v[iii][d] += m_data->Q_xi[i][d]; // using previously stored value        
      }
    } // flag_stochastic

  } // i

  // update the position from x^n to x^{n+1} 
  double KBT = m_data->KBT; double delta = m_data->delta;
  double pre_KBT_dt_o_delta = KBT*dt/delta;
  for (int i = 0; i < m_data->num_II; i++) {
    int iii = m_data->II[i];

    // update the position from x^n to x^{n+1}
    x[iii][0] = m_data->store_x[i][0];
    x[iii][1] = m_data->store_x[i][1];
    x[iii][2] = m_data->store_x[i][2];
    
    x[iii][0] += 0.5*dt*(v[iii][0] + m_data->prev_v[i][0]);
    x[iii][1] += 0.5*dt*(v[iii][1] + m_data->prev_v[i][1]);
    x[iii][2] += 0.5*dt*(v[iii][2] + m_data->prev_v[i][2]);

    if ((m_data->flag_stochastic) && (m_data->flag_thm_drift)) {
      x[iii][0] += pre_KBT_dt_o_delta*m_data->avg_diff_M[i][0];
      x[iii][1] += pre_KBT_dt_o_delta*m_data->avg_diff_M[i][1];
      x[iii][2] += pre_KBT_dt_o_delta*m_data->avg_diff_M[i][2];
    }

  } // i
    
}

/* ---------------------------------------------------------------------- */
void DriverMLMOD::initial_integrate_dX_MF_Q1_ML1_N2N()
{

  // get atom info v
  Atom *atom = this->lammps->atom;
  Update *update = this->lammps->update;
  Force *force = this->lammps->force;

  int igroup = this->fixMLMOD->igroup; // may need to reference back to fixMLMOD
   
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask_atom = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;
  int num_dim = 3;

  double dt = update->dt;

  int groupbit = this->fixMLMOD->groupbit;

  // use .pt model from PyTorch for ML learned model for M(X)
  // get model data
  ModelData_dX_MF_Q1_ML1_N2N_Type *m_data 
    = (ModelData_dX_MF_Q1_ML1_N2N_Type *) this->model_data;

  // initialize
  if (m_data->flag_init == 0) {

    // determine index list controlled by this fix 
    // get the II indices for the group
    m_data->II_size = 0; m_data->II = NULL; 
    get_indices_for_group(groupbit,mask_atom,nlocal,&m_data->II_size,&m_data->II,&m_data->num_II);

    // get tensor input size given masks
    m_data->input_size = get_tensor_input_size(m_data->num_II,m_data->II,num_dim,
                                               m_data->mask_input, m_data->mask_list_n, m_data->mask_list,
                                               groupbit, mask_atom);

    // alloc data for A matrix for lapack use
    if (m_data->A_size == 0) {
      m_data->A 
        = (double *)malloc(sizeof(double)*m_data->num_II
                                         *num_dim*m_data->num_II*num_dim);
      m_data->A_size = m_data->num_II*num_dim*m_data->num_II*num_dim;
    }

    // setup data structure for saving Q\xi 
    // if not yet allocated 
    if (m_data->prev_v_size == 0) {
      m_data->prev_v_block 
        = (double *)malloc(sizeof(double)*m_data->num_II*num_dim);
      m_data->prev_v = (double **)malloc(sizeof(double)*m_data->num_II);
      for (int i = 0; i < m_data->num_II; i++) {
        m_data->prev_v[i] = m_data->prev_v_block 
                          + i*num_dim + 0; // row-major blocking
        for (int d = 0; d < num_dim; d++) {
          m_data->prev_v[i][d] = 0.0;
        }
      }
      m_data->prev_v_size = m_data->num_II;
    }

    if (m_data->store_x_size == 0) {
      m_data->store_x_block 
        = (double *)malloc(sizeof(double)*m_data->num_II*num_dim);
      m_data->store_x = (double **)malloc(sizeof(double)*m_data->num_II);
      for (int i = 0; i < m_data->num_II; i++) {
        m_data->store_x[i] = m_data->store_x_block 
                           + i*num_dim + 0; // row-major blocking
        for (int d = 0; d < num_dim; d++) {
          m_data->store_x[i][d] = 0.0;
        }
      }
      m_data->store_x_size = m_data->num_II;
    }

    if ((m_data->xi_size == 0) && (m_data->flag_stochastic)) {
      m_data->xi_block 
        = (double *)malloc(sizeof(double)*m_data->num_II*num_dim);
      m_data->xi = (double **)malloc(sizeof(double)*m_data->num_II);
      for (int i = 0; i < m_data->num_II; i++) {
        m_data->xi[i] = m_data->xi_block 
                      + i*num_dim + 0; // row-major blocking
        for (int d = 0; d < num_dim; d++) {
          m_data->xi[i][d] = 0.0;
        }
      }
      m_data->xi_size = m_data->num_II;
    }

    if ((m_data->Q_xi_size == 0) && (m_data->flag_stochastic)) {
      m_data->Q_xi_block 
        = (double *)malloc(sizeof(double)*m_data->num_II*num_dim);
      m_data->Q_xi = (double **)malloc(sizeof(double)*m_data->num_II);
      for (int i = 0; i < m_data->num_II; i++) {
        m_data->Q_xi[i] = m_data->Q_xi_block 
                        + i*num_dim + 0; // row-major blocking
        for (int d = 0; d < num_dim; d++) {
          m_data->Q_xi[i][d] = 0.0;
        }
      }
      m_data->Q_xi_size = m_data->num_II;
    }

    if (((m_data->avg_diff_M_size[0] == 0) || 
         (m_data->avg_diff_M_size[1] == 0)) &&
         (m_data->flag_stochastic) && (m_data->flag_thm_drift)) {
      m_data->avg_diff_M_block 
        = (double *)malloc(sizeof(double)*m_data->num_II*num_dim
                                         *m_data->num_II*num_dim);
      m_data->avg_diff_M
        = (double **)malloc(sizeof(double)*m_data->num_II*num_dim);
      for (int i = 0; i < m_data->num_II*num_dim; i++) {
        m_data->avg_diff_M[i] 
          = m_data->avg_diff_M_block 
          + i*m_data->num_II*num_dim + 0; // row-major blocking
        for (int d = 0; d < m_data->num_II*num_dim; d++) {
          m_data->avg_diff_M[i][d] = 0.0;
        }
      }
      m_data->avg_diff_M_size[0] = m_data->num_II*num_dim; 
      m_data->avg_diff_M_size[1] = m_data->num_II*num_dim;
    }

    if ((m_data->M_size[0] == 0) || (m_data->M_size[1] == 0)) {
      m_data->M_block 
        = (double *)malloc(sizeof(double)*m_data->num_II*num_dim
                                         *m_data->num_II*num_dim);
      m_data->M = (double **)malloc(sizeof(double)*m_data->num_II*num_dim);
      for (int i = 0; i < m_data->num_II*num_dim; i++) {
        m_data->M[i] 
          = m_data->M_block 
          + i*m_data->num_II*num_dim + 0; // row-major blocking
        for (int d = 0; d < m_data->num_II*num_dim; d++) {
          m_data->M[i][d] = 0.0;
        }
      }
      m_data->M_size[0] = m_data->num_II*num_dim; 
      m_data->M_size[1] = m_data->num_II*num_dim;
    }

    if (((m_data->Q_size[0] == 0) || (m_data->Q_size[1] == 0)) 
          && (m_data->flag_stochastic)) {
      m_data->Q_block 
        = (double *)malloc(sizeof(double)*m_data->num_II*num_dim
                                         *m_data->num_II*num_dim);
      m_data->Q = (double **)malloc(sizeof(double)*m_data->num_II*num_dim);
      for (int i = 0; i < m_data->num_II*num_dim; i++) {
        m_data->Q[i] 
          = m_data->Q_block + i*m_data->num_II*num_dim + 0; // row-major blocking
        for (int d = 0; d < m_data->num_II*num_dim; d++) {
          m_data->Q[i][d] = 0.0;
        }
      }
      m_data->Q_size[0] = m_data->num_II*num_dim; 
      m_data->Q_size[1] = m_data->num_II*num_dim;
    }

    if (m_data->flag_thm_drift) {
      // divergence KBT*div*M(X) treated as zero for now
      // (assumes small enough variations in X for now)
      cout << "WARNING: KB*T*Div(M) term treated as 0.0 for now.\n"; 
    }

    m_data->flag_init = 1;

  } // end flag_init

  // create tensors
  at::Tensor input2_m = torch::zeros({m_data->num_II*num_dim,1});
  //auto input2_m_a = input2_m.accessor<float,2>(); 
  at::TensorAccessor<float, 2, at::DefaultPtrTraits, long> input2_m_a = input2_m.accessor<float,2>(); 
  std::vector<torch::jit::IValue> input2_m_i{input2_m};

  at::Tensor output_m; // leave un-initialized

  // initialize the velocity to zero
  for (int i = 0; i < m_data->num_II; i++) {
    int iii = m_data->II[i];
    for (int d = 0; d < num_dim; d++) {
      v[iii][d] = 0.0;   
    }
  }

  // construct the mobility tensor M 
  // (@optimize later using neigh list thresholds)
  for (int i = 0; i < m_data->num_II; i++) {
    int iii = m_data->II[i];
    for (int d = 0; d < num_dim; d++) {
      input2_m_a[i*num_dim + d][0] = x[iii][d];
    }
  }

  // evaluate the model M(X) and obtain tensor output
  output_m = m_data->M_model->forward(input2_m_i).toTensor();
  // accessor for the 2D tensor 
  // 1 x m_data->num_II*num_dim*m_data->num_II*num_dim
  auto output_m_a = output_m.accessor<float,2>(); 

  // reshape into the needed M tensor components
  for (int i = 0; i < m_data->num_II; i++) {
    int iii = m_data->II[i];
    for (int j = 0; j < m_data->num_II; j++) {
      int jjj = m_data->II[j];

      for (int d = 0; d < num_dim; d++) {
        for (int k = 0; k < num_dim; k++) {
          int I1 = (i*num_dim + d)*m_data->num_II*num_dim + j*num_dim + k; 
          m_data->M[i*num_dim + d][j*num_dim + k] = output_m_a[I1][0]; 
        }
      }
      
    } // j
  } // i

  // compute the cholesky factor QQ^T = 2*KB*T*M/dt
  // @optimize for only subset of particles in group  
  if (m_data->flag_stochastic) {
    int factor_type = 1;
    if (factor_type == 1) {
      int nn = m_data->num_II*num_dim; char uplo = 'L'; 
      int info = 0; int lda = m_data->num_II*num_dim;
      double *A; double **M; double **Q;
      double two_KBT_o_dt = 2.0*m_data->KBT/dt;
      A = m_data->A; M = m_data->M; Q = m_data->Q;

      for (int ii = 0; ii < m_data->num_II*num_dim; ii++) {
        for (int jj = 0; jj < m_data->num_II*num_dim; jj++) {
          A[jj*m_data->num_II*num_dim + ii] 
            = two_KBT_o_dt*M[ii][jj]; // put in column-major format
        }
      }

      // lapack cholesky factorization 
      // (done in-place, returns only lower triangular part)
      // dpotrf_(&uplo,&nn,A,&lda,&info);
      info = LAPACKE_dpotrf2(LAPACK_COL_MAJOR,uplo,nn,A,lda);

      // test for error
      assert(info == 0);

      // convert to Q 
      // (assumes Q lower tri and upper already initialized to zero) 
      for (int ii = 0; ii < m_data->num_II*num_dim; ii++) {
        for (int jj = 0; jj <= ii; jj++) {
          Q[ii][jj] 
            = A[jj*m_data->num_II*num_dim + ii]; // A is column-major format
        }
      }

    } // factor_type

    // compute Q*xi (assumes only lower triangular part is non-zero) 
    for (int ii = 0; ii < m_data->num_II*num_dim; ii++) {
      m_data->xi_block[ii] 
        = (double) this->random->gaussian(); // normal random number 
    }
    for (int ii = 0; ii < m_data->num_II*num_dim; ii++) {
      m_data->Q_xi_block[ii] = 0.0;
      // generate random xi and multiply by Q
      for (int jj = 0; jj <= ii; jj++) {
        m_data->Q_xi_block[ii] 
          += m_data->Q[ii][jj]*m_data->xi_block[jj];
      }
    }

  } // flag_stochastic

  // compute the velocity update
  // compute contributions to V_i
  for (int i = 0; i < m_data->num_II; i++) {
    int iii = m_data->II[i];
    // particle j force contributions
    for (int j = 0; j < m_data->num_II; j++) {
      int jjj = m_data->II[j];
      // compute V_i = M_ij*f_j 
      for (int d = 0; d < num_dim; d++) {
        for (int k = 0; k < num_dim; k++) {
          v[iii][d] += m_data->M[i*num_dim + d][j*num_dim + k]*f[jjj][k]; 
        }
      }
    } // j

    // stochastic contributions
    if (m_data->flag_stochastic) {
      for (int d = 0; d < num_dim; d++) {
        v[iii][d] += m_data->Q_xi[i][d];
      }
    } // flag_stochastic

  } // i

  if ((m_data->flag_stochastic) && (m_data->flag_thm_drift)) {
    cout << "WARNING: flag_thm_drift not yet implemented in prototype code.\n";
    // compute the perturbations avg_diff_M
    // @@@
    // for (int ii = 0; ii < m_data->num_II*num_dim; ii++) {
    //  m_data->Q_xi_block[ii] = 0.0;
    //  // generate random xi and multiply by Q
    //  for (int jj = 0; jj <= ii; jj++) {
    //    m_data->xi_block[jj] 
    //    = (double) this->random->gaussian(); // normal random number 
    //    m_data->Q_xi_block[ii] += m_data->Q[ii][jj]*m_data->xi_block[jj];
    //  }
    //}
  }

  // save to update from the initial position x^n to x^{n+1}
  for (int i = 0; i < m_data->num_II; i++) {
    int iii = m_data->II[i];

    // save the baseline x^n 
    m_data->store_x[i][0] = x[iii][0];
    m_data->store_x[i][1] = x[iii][1];
    m_data->store_x[i][2] = x[iii][2];

    // compute the tilde_x^{n+1} for intermediate force calculations
    x[iii][0] += dt*v[iii][0];
    x[iii][1] += dt*v[iii][1];
    x[iii][2] += dt*v[iii][2];
  }
    
  // save the velocity for next step of integrator
  for (int i = 0; i < m_data->num_II; i++) {
    int iii = m_data->II[i];

    for (int d = 0; d < num_dim; d++) {
      m_data->prev_v[i][d] = v[iii][d];   
    }
    
  } // i

}

/* ---------------------------------------------------------------------- */
void DriverMLMOD::final_integrate_dX_MF_Q1_ML1_N2N()
{

  // get atom info v
  Atom *atom = this->lammps->atom;
  Update *update = this->lammps->update;
  Force *force = this->lammps->force;

  int igroup = this->fixMLMOD->igroup; // may need to reference back to fixMLMOD
   
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask_atom = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;
  int num_dim = 3;

  double dt = update->dt;

  int groupbit = this->fixMLMOD->groupbit;

  // use .pt model from PyTorch for ML learned model for M(X)
  // get model data
  ModelData_dX_MF_Q1_ML1_N2N_Type *m_data 
    = (ModelData_dX_MF_Q1_ML1_N2N_Type *) this->model_data;

  // create tensors
  at::Tensor input_m = torch::zeros({m_data->num_II*num_dim,1});
  auto input_m_a = input_m.accessor<float,2>(); 
  std::vector<torch::jit::IValue> input_m_i{input_m};

  at::Tensor output_m; // leave un-initialized

  // initialize the velocity to zero
  for (int i = 0; i < m_data->num_II; i++) {
    int iii = m_data->II[i];
    for (int d = 0; d < num_dim; d++) {
      v[iii][d] = 0.0;   
    }
  }

  // construct the mobility tensor M
  // (@optimize later using neigh list thresholds)
  for (int i = 0; i < m_data->num_II; i++) {
    for (int d = 0; d < num_dim; d++) {
      int iii = m_data->II[i];
      input_m_a[i*num_dim + d][0] = x[iii][d];
    }
  }

  // evaluate the model M(X) and obtain tensor output
  output_m = m_data->M_model->forward(input_m_i).toTensor();
  // accessor for the 2D tensor 
  // 1 x m_data->num_II*num_dim*m_data->num_II*num_dim
  auto output_m_a = output_m.accessor<float,2>(); 

  // reshape into the needed M tensor components
  for (int i = 0; i < m_data->num_II; i++) {
    int iii = m_data->II[i];
    for (int j = 0; j < m_data->num_II; j++) {
      int jjj = m_data->II[j];
      for (int d = 0; d < num_dim; d++) {
        for (int k = 0; k < num_dim; k++) {
          int I1 = (i*num_dim + d)*m_data->num_II*num_dim + j*num_dim + k; 
          m_data->M[i*num_dim + d][j*num_dim + k] = output_m_a[I1][0]; 
        }
      }
    } // j
  } // i

  // compute the velocity update
  // compute contributions to V_i
  for (int i = 0; i < m_data->num_II; i++) {
    int iii = m_data->II[i];

    // particle j contributions
    for (int j = 0; j < m_data->num_II; j++) {
      int jjj = m_data->II[j];

      // compute V_i = M_ij*f_j
      for (int d = 0; d < num_dim; d++) {
        for (int k = 0; k < num_dim; k++) {          
          v[iii][d] += m_data->M[i*num_dim + d][j*num_dim + k]*f[jjj][k]; 
        }          
      }

    } // j 

    // stochastic contributions
    if (m_data->flag_stochastic) {
      int iii = m_data->II[i];
      for (int d = 0; d < num_dim; d++) {
        v[iii][d] += m_data->Q_xi[i][d]; // using previously stored value        
      }
    } // flag_stochastic

  } // i

  // update the position from x^n to x^{n+1} 
  double KBT = m_data->KBT; double delta = m_data->delta;
  double pre_KBT_dt_o_delta = KBT*dt/delta;
  for (int i = 0; i < m_data->num_II; i++) {
    int iii = m_data->II[i];

    if (mask_atom[i] & groupbit) {
      // update the position from x^n to x^{n+1}
      x[iii][0] = m_data->store_x[i][0];
      x[iii][1] = m_data->store_x[i][1];
      x[iii][2] = m_data->store_x[i][2];
      
      x[iii][0] += 0.5*dt*(v[iii][0] + m_data->prev_v[i][0]);
      x[iii][1] += 0.5*dt*(v[iii][1] + m_data->prev_v[i][1]);
      x[iii][2] += 0.5*dt*(v[iii][2] + m_data->prev_v[i][2]);

      if ((m_data->flag_stochastic) && (m_data->flag_thm_drift)) {
        x[iii][0] += pre_KBT_dt_o_delta*m_data->avg_diff_M[i][0];
        x[iii][1] += pre_KBT_dt_o_delta*m_data->avg_diff_M[i][1];
        x[iii][2] += pre_KBT_dt_o_delta*m_data->avg_diff_M[i][2];
      }
    } // mask

  } // i
    
}

/* ---------------------------------------------------------------------- */
void DriverMLMOD::post_force_F_ML1()
{

  // get atom info v
  Atom *atom = this->lammps->atom;
  Update *update = this->lammps->update;
  Force *force = this->lammps->force;

  int igroup = this->fixMLMOD->igroup; // may need to reference back to fixMLMOD
   
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask_atom = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;
  int num_dim = 3;
  int offset_tensor = 0;

  double dt = update->dt;

  double time = update->dt*update->ntimestep;

  int groupbit = this->fixMLMOD->groupbit;

  // use .pt model from PyTorch for ML learned model for M(X)
  // get model data
  ModelData_F_ML1_Type *m_data 
    = (ModelData_F_ML1_Type *) this->model_data;

  // initialize
  if (m_data->flag_init == 0) {

    // determine index list controlled by this fix 
    // get the II indices for the group
    m_data->II_size = 0; m_data->II = NULL; 
    get_indices_for_group(groupbit,mask_atom,nlocal,&m_data->II_size,&m_data->II,&m_data->num_II);

    // get tensor input size given masks
    m_data->input_size = get_tensor_input_size(m_data->num_II,m_data->II,num_dim,
                                               m_data->mask_input, m_data->mask_list_n, m_data->mask_list,
                                               groupbit, mask_atom);

    m_data->flag_init = 1;

  } // end flag_init

  // get the II indices for the group 
  get_indices_for_group(groupbit,mask_atom,nlocal,&m_data->II_size,&m_data->II,&m_data->num_II);

  // get tensor input size given masks
  m_data->input_size = get_tensor_input_size(m_data->num_II,m_data->II,num_dim,
                                             m_data->mask_input, m_data->mask_list_n, m_data->mask_list,
                                             groupbit, mask_atom);

  // create tensors
  at::Tensor input2_m = torch::zeros({m_data->input_size,1}); // PJA: change to column from row
  //auto input2_m_a = input2_m.accessor<float,2>(); 
  at::TensorAccessor<float,2,at::DefaultPtrTraits,long> input2_m_a = input2_m.accessor<float,2>(); 
  std::vector<torch::jit::IValue> input2_m_i{input2_m};

  at::Tensor output_m; // leave un-initialized

  // construct the input for force F() 
  build_tensor_from_masks(input2_m_a,m_data->num_II,m_data->II,m_data->mask_input,
                          m_data->mask_list_n,m_data->mask_list,
                          groupbit,mask_atom,
                          x,v,f,type,time);

  // evaluate the model F(X) and obtain tensor output
  // 1 x m_data->num_II*num_dim*m_data->num_II*num_dim
  output_m = m_data->F_model->forward(input2_m_i).toTensor();

  // accessor for the 2D tensor 
  //auto output_m_a = output_m.accessor<float,2>(); 

  // update the force
  offset_tensor = 0;
  add_tensor_full_to_db_vec(f,output_m,m_data->num_II,num_dim,m_data->II,offset_tensor);

}

/* ---------------------------------------------------------------------- */
void DriverMLMOD::post_force_F_X_ML1()
{

  // get atom info v
  Atom *atom = this->lammps->atom;
  Update *update = this->lammps->update;
  Force *force = this->lammps->force;

  int igroup = this->fixMLMOD->igroup; // may need to reference back to fixMLMOD
   
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask_atom = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;
  int num_dim = 3;
  int offset_tensor = 0;

  double dt = update->dt;

  int groupbit = this->fixMLMOD->groupbit;

  int count;

  double time = update->dt*update->ntimestep;

  // use .pt model from PyTorch for ML learned model for M(X)
  // get model data
  ModelData_F_X_ML1_Type *m_data 
    = (ModelData_F_X_ML1_Type *) this->model_data;

  // initialize
  if (m_data->flag_init == 0) {

    // determine index list controlled by this fix 
    m_data->num_II = 1; m_data->II_size = m_data->num_II;
    m_data->II = (int *)malloc(sizeof(int)*m_data->II_size);
    m_data->II[0] = 0; 

    // get tensor input size given masks
    m_data->input_size = get_tensor_input_size(m_data->num_II,m_data->II,num_dim,
                                               m_data->mask_input, m_data->mask_list_n, m_data->mask_list,
                                               0, NULL);

    m_data->flag_init = 1;

  } // end flag_init

  // create tensors
  at::Tensor input2_m = torch::zeros({m_data->input_size,1}); 
  //auto input2_m_a = input2_m.accessor<float,2>(); 
  at::TensorAccessor<float,2,at::DefaultPtrTraits,long> input2_m_a = input2_m.accessor<float,2>(); 
  std::vector<torch::jit::IValue> input2_m_i{input2_m};

  at::Tensor output_m; // leave un-initialized

  for (int i = 0; i < nlocal; i++) {

    if (groupbit & mask_atom[i]) {
      // construct the input for force F(X_i,V_i,F_i,IT_i,t_n)
      m_data->II[0] = i;       
      build_tensor_from_masks(input2_m_a,m_data->num_II,m_data->II,m_data->mask_input,
			      m_data->mask_list_n,m_data->mask_list,
			      0,NULL,
			      x,v,f,type,time);

      // evaluate the model F(X) and obtain tensor output
      output_m = m_data->F_model->forward(input2_m_i).toTensor();

      // update the force
      offset_tensor = 0;
      add_tensor_seg_to_db_vec(f,output_m,m_data->num_II,num_dim,m_data->II,offset_tensor);
    }

  } 

}

/* ---------------------------------------------------------------------- */
void DriverMLMOD::post_force_F_Pair_ML1()
{

  // get atom info v
  Atom *atom = this->lammps->atom;
  Update *update = this->lammps->update;
  Force *force = this->lammps->force;

  int igroup = this->fixMLMOD->igroup; // may need to reference back to fixMLMOD
   
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask_atom = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;
  int num_dim = 3;
  int offset_tensor = 0;

  double dt = update->dt;

  int groupbit = this->fixMLMOD->groupbit;

  double time = update->dt*update->ntimestep;

  int count;

  // use .pt model from PyTorch for ML learned model for M(X)
  // get model data
  ModelData_F_Pair_ML1_Type *m_data 
    = (ModelData_F_Pair_ML1_Type *) this->model_data;

  // initialize
  if (m_data->flag_init == 0) {

    // determine index list controlled by this fix 
    m_data->num_II = 2; m_data->II_size = m_data->num_II;
    m_data->II = (int *)malloc(sizeof(int)*m_data->II_size);
    m_data->II[0] = 0; m_data->II[1] = 1; 

    // get tensor input size given masks
    m_data->input_size = get_tensor_input_size(m_data->num_II,m_data->II,num_dim,
                                               m_data->mask_input, m_data->mask_list_n, m_data->mask_list,
                                               0, NULL);

    m_data->flag_init = 1;

  } // end flag_init

  // create tensors
  at::Tensor input2_m = torch::zeros({m_data->input_size,1}); 
  //auto input2_m_a = input2_m.accessor<float,2>(); 
  at::TensorAccessor<float,2,at::DefaultPtrTraits,long> input2_m_a = input2_m.accessor<float,2>(); 
  std::vector<torch::jit::IValue> input2_m_i{input2_m};

  at::Tensor output_m; // leave un-initialized

  // loop over unique pairs with i < j 
  for (int i = 0; i < nlocal - 1; i++) {
    for (int j = i + 1; j < nlocal; j++) {

      if ((groupbit & mask_atom[i]) && (groupbit & mask_atom[j])) {
	// construct the input for force F(X_i,V_i,F_i,IT_i,t_n)
	m_data->II[0] = i; m_data->II[1] = j;
	build_tensor_from_masks(input2_m_a,m_data->num_II,m_data->II,m_data->mask_input,
				m_data->mask_list_n,m_data->mask_list,
				0,NULL,
				x,v,f,type,time);

	// evaluate the model F(X) and obtain tensor output
	output_m = m_data->F_model->forward(input2_m_i).toTensor();

        // update the force
	offset_tensor = 0;
	add_tensor_seg_to_db_vec(f,output_m,m_data->num_II,num_dim,m_data->II,offset_tensor);
      }

    } // j
  } // i 

}

/* ---------------------------------------------------------------------- */
void DriverMLMOD::initial_integrate_QoI_ML1()
{

  // get atom info v
  Atom *atom = this->lammps->atom;
  Update *update = this->lammps->update;
  Force *force = this->lammps->force;

  int igroup = this->fixMLMOD->igroup; // may need to reference back to fixMLMOD
   
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask_atom = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;
  int num_dim = 3;

  double dt = update->dt;
  double time = update->dt*update->ntimestep;

  int groupbit = this->fixMLMOD->groupbit;

  // use .pt model from PyTorch for ML learned model for M(X)
  // get model data
  ModelData_QoI_ML1_Type *m_data 
    = (ModelData_QoI_ML1_Type *) this->model_data;

  // initialize
  if (m_data->flag_init == 0) {

    // determine index list controlled by this fix 
    // get the II indices for the group
    m_data->II_size = 0; m_data->II = NULL; 
    get_indices_for_group(groupbit,mask_atom,nlocal,&m_data->II_size,&m_data->II,&m_data->num_II);

    // get tensor input size given masks
    m_data->input_size = get_tensor_input_size(m_data->num_II,m_data->II,num_dim,
                                               m_data->mask_input, m_data->mask_list_n, m_data->mask_list,
                                               groupbit, mask_atom);

    m_data->flag_init = 1;

  }

  // only compute every skip_step (syncronized with nevery in lammps)
  if (lammps->update->ntimestep % m_data->skip_step == 0) { 

    // create tensors (@optimize should do this just once)
    at::Tensor input2_m = torch::zeros({m_data->input_size,1});
    // auto input2_m_a = input2_m.accessor<float,2>(); 
    at::TensorAccessor<float,2,at::DefaultPtrTraits,long> input2_m_a = input2_m.accessor<float,2>(); 
    std::vector<torch::jit::IValue> input2_m_i{input2_m};

    at::Tensor output_m; // leave un-initialized

    // construct the input for force QoI() 
    build_tensor_from_masks(input2_m_a,m_data->num_II,m_data->II,m_data->mask_input,
                            m_data->mask_list_n,m_data->mask_list,
                            groupbit,mask_atom,
                            x,v,f,type,time);

    // evaluate the model F(X) and obtain tensor output
    output_m = m_data->QoI_model->forward(input2_m_i).toTensor();
    c10::IntArrayRef output_m_shape = output_m.sizes();

   // initialization (first pass) 
   // check if we need to setup the QoI array 
   // array for qoi (so can be used in other lammps work-flow)
   m_data->flag_val_qoi = 1;
   if ((m_data->val_qoi == 0) && (m_data->flag_val_qoi)) {
     m_data->qoi_num_rows = output_m_shape[0];
     m_data->qoi_num_cols = output_m_shape[1];
     m_data->val_qoi_block = 0; m_data->val_qoi_size = -1;
     m_data->val_qoi = create_db_vec(m_data->qoi_num_rows,
                                     m_data->qoi_num_cols,
                                     &m_data->val_qoi_block,
                                     &m_data->val_qoi_size);
    
     // setup the array sizes for lammps
     fixMLMOD->array_flag = 1;                         // 0/1 if compute_array() function exists
     fixMLMOD->size_array_rows_variable = 0;           // 1 if array rows is unknown in advance
     fixMLMOD->size_array_rows = m_data->qoi_num_rows; // rows in global array
     fixMLMOD->size_array_cols = m_data->qoi_num_cols; // columns in global array

   } // end of initialization

   // copy data into the qoi array
   int num_rows = m_data->qoi_num_rows;
   int num_cols = m_data->qoi_num_cols;
   set_db_vec_to_tensor_full(m_data->val_qoi,output_m,num_rows,num_cols,NULL,0);

    // write the output 
    int flag_write_output = 1;
    if (flag_write_output != 0) {
      // accessor for the 2D tensor 
      // 1 x m_data->num_II*num_dim*m_data->num_II*num_dim
      auto output_m_a = output_m.accessor<float,2>();

      const char *base_name = m_data->base_name->c_str();
      const char* base_dir = m_data->base_dir->c_str();

      char *filename = new char[strlen(base_name) + strlen(base_dir) + 17 + 100];
      sprintf(filename,"%s/%s__initial_integrate__0_%.8ld.xml",
	      base_dir,base_name,lammps->update->ntimestep);
      ofstream fid(filename);
      // fid.precision(6)  

      if (fid.is_open()) {
	fid << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
	fid << "<base_name value=\"" << base_name << "\"/>\n";
	fid << "<num_rows value=\"" << m_data->qoi_num_rows << "\"/>\n";
	fid << "<num_cols value=\"" << m_data->qoi_num_cols << "\"/>\n";

	fid << "<data>\n";
	for (int i = 0; i < m_data->qoi_num_rows; i++) {
	  for (int j = 0; j < m_data->qoi_num_cols; j++) {
	    fid << m_data->val_qoi[i][j] << " "; 
	  }
        }

	fid << "\n</data>\n";
        fid.close();
        delete[] filename;
      }

    }

  }

}

/* ---------------------------------------------------------------------- */
void DriverMLMOD::final_integrate_QoI_ML1()
{

  // get atom info 
  Atom *atom = this->lammps->atom;
  Update *update = this->lammps->update;
  Force *force = this->lammps->force;

  int igroup = this->fixMLMOD->igroup; // may need to reference back to fixMLMOD
   
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask_atom = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;
  int num_dim = 3;

  double dt = update->dt;
  double time = update->dt*update->ntimestep;

  int groupbit = this->fixMLMOD->groupbit;

  // use .pt model from PyTorch for ML learned model for M(X)
  // get model data
  ModelData_QoI_ML1_Type *m_data 
    = (ModelData_QoI_ML1_Type *) this->model_data;

  // initialize
  if (m_data->flag_init == 0) {

    // determine index list controlled by this fix 
    // get the II indices for the group
    m_data->II_size = 0; m_data->II = NULL; 
    get_indices_for_group(groupbit,mask_atom,nlocal,&m_data->II_size,&m_data->II,&m_data->num_II);

    // get tensor input size given masks
    m_data->input_size = get_tensor_input_size(m_data->num_II,m_data->II,num_dim,
                                               m_data->mask_input, m_data->mask_list_n, m_data->mask_list,
                                               groupbit, mask_atom);

    m_data->flag_init = 1;

  } // end flag_init

  // only compute every skip_step (syncronized with nevery in lammps)
  if (lammps->update->ntimestep % m_data->skip_step == 0) { 

    // create tensors (@optimize should do this just once)
    at::Tensor input_m = torch::zeros({m_data->input_size,1});
    //auto input_m_a = input_m.accessor<float,2>(); 
    at::TensorAccessor<float,2,at::DefaultPtrTraits,long> input_m_a = input_m.accessor<float,2>(); 
    std::vector<torch::jit::IValue> input_m_i{input_m};

    at::Tensor output_m; // leave un-initialized

    // construct the input for force QoI() 
    build_tensor_from_masks(input_m_a,m_data->num_II,m_data->II,m_data->mask_input,
                            m_data->mask_list_n,m_data->mask_list,
                            groupbit,mask_atom,
                            x,v,f,type,time);

    // evaluate the model QoI() and obtain tensor output
    // output 2D tensor assumed shape=[qoi_size,1]
    output_m = m_data->QoI_model->forward(input_m_i).toTensor();
    c10::IntArrayRef output_m_shape = output_m.sizes();

   // initialization (first pass) 
   // check if we need to setup the QoI array 
   // array for qoi (so can be used in other lammps work-flow)
   m_data->flag_val_qoi = 1;
   if ((m_data->val_qoi == 0) && (m_data->flag_val_qoi)) {
     m_data->qoi_num_rows = output_m_shape[0];
     m_data->qoi_num_cols = output_m_shape[1];
     m_data->val_qoi_block = 0; m_data->val_qoi_size = -1;
     m_data->val_qoi = create_db_vec(m_data->qoi_num_rows,
                                     m_data->qoi_num_cols,
                                     &m_data->val_qoi_block,
                                     &m_data->val_qoi_size);
    
     // setup the array sizes for lammps
     fixMLMOD->array_flag = 1;                         // 0/1 if compute_array() function exists
     fixMLMOD->size_array_rows_variable = 0;           // 1 if array rows is unknown in advance
     fixMLMOD->size_array_rows = m_data->qoi_num_rows; // rows in global array
     fixMLMOD->size_array_cols = m_data->qoi_num_cols; // columns in global array

   } // end of initialization

   // copy data into the qoi array
   int num_rows = m_data->qoi_num_rows;
   int num_cols = m_data->qoi_num_cols;
   set_db_vec_to_tensor_full(m_data->val_qoi,output_m,num_rows,num_cols,NULL,0);

    // write the output 
    int flag_write_output = 1;
    if (flag_write_output != 0) {
      // accessor for the 2D tensor 
      // 1 x m_data->num_II*num_dim*m_data->num_II*num_dim
      auto output_m_a = output_m.accessor<float,2>();

      const char *base_name = m_data->base_name->c_str();
      const char* base_dir = m_data->base_dir->c_str();

      char *filename = new char[strlen(base_name) + strlen(base_dir) + 17 + 100];
      sprintf(filename,"%s/%s__final_integrate__0_%.8ld.xml",
	      base_dir,base_name,lammps->update->ntimestep);
      ofstream fid(filename);
      // fid.precision(6)  

      if (fid.is_open()) {
	fid << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
	fid << "<base_name value=\"" << base_name << "\"/>\n";
	fid << "<num_rows value=\"" << m_data->qoi_num_rows << "\"/>\n";
	fid << "<num_cols value=\"" << m_data->qoi_num_cols << "\"/>\n";

	fid << "<data>\n";
	for (int i = 0; i < m_data->qoi_num_rows; i++) {
	  for (int j = 0; j < m_data->qoi_num_cols; j++) {
	    fid << m_data->val_qoi[i][j] << " "; 
	  }
        }

	fid << "\n</data>\n";
	fid.close();
        delete[] filename;
      }

    }

  }

}

/* ---------------------------------------------------------------------- */
void DriverMLMOD::initial_integrate_Dyn_ML1()
{

  // get atom info v
  Atom *atom = this->lammps->atom;
  Update *update = this->lammps->update;
  Force *force = this->lammps->force;

  int igroup = this->fixMLMOD->igroup; // may need to reference back to fixMLMOD
   
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask_atom = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;
  int num_dim = 3;

  double dt = update->dt;
  
  double time = update->dt*update->ntimestep;

  int groupbit = this->fixMLMOD->groupbit;

  // use .pt model from PyTorch for ML learned model for M(X)
  // get model data
  ModelData_Dyn_ML1_Type *m_data 
    = (ModelData_Dyn_ML1_Type *) this->model_data;

  // initialize
  if (m_data->flag_init == 0) {

    // determine index list controlled by this fix 
    // get the II indices for the group
    m_data->II_size = 0; m_data->II = NULL; 
    get_indices_for_group(groupbit,mask_atom,nlocal,&m_data->II_size,&m_data->II,&m_data->num_II);

    // get tensor input size given masks
    m_data->input_size = get_tensor_input_size(m_data->num_II,m_data->II,num_dim,
                                               m_data->mask_input, m_data->mask_list_n, m_data->mask_list,
                                               groupbit, mask_atom);

    m_data->flag_init = 1;

  } // end flag_init

  // create tensors
  at::Tensor input2_m = torch::zeros({m_data->input_size,1});
  //auto input2_m_a = input2_m.accessor<float,2>(); 
  at::TensorAccessor<float,2,at::DefaultPtrTraits,long> input2_m_a = input2_m.accessor<float,2>(); 
  std::vector<torch::jit::IValue> input2_m_i{input2_m};

  at::Tensor output_m; // leave un-initialized

  // construct the input for update (X,V) = Gamma(X,V,...) 
  // (@optimize later using neigh list thresholds)
  build_tensor_from_masks(input2_m_a,m_data->num_II,m_data->II,m_data->mask_input,
                          m_data->mask_list_n,m_data->mask_list,
                          groupbit,mask_atom,
                          x,v,f,type,time);

  // evaluate the model (X,V) = Gamma(X,V) and obtain tensor output
  // output 2D tensor assumed shape=[2*num_II*num_dim,1]
  output_m = m_data->Dyn1_model->forward(input2_m_i).toTensor();

  int II1 = 0; int II2 = II1 + num_dim*m_data->num_II;
  set_db_vec_to_tensor_full(x,output_m,m_data->num_II,num_dim,m_data->II,II1);
  set_db_vec_to_tensor_full(v,output_m,m_data->num_II,num_dim,m_data->II,II2);
  
}

/* ---------------------------------------------------------------------- */
void DriverMLMOD::final_integrate_Dyn_ML1()
{
  // get atom info v
  Atom *atom = this->lammps->atom;
  Update *update = this->lammps->update;
  Force *force = this->lammps->force;

  int igroup = this->fixMLMOD->igroup; // may need to reference back to fixMLMOD
   
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask_atom = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;
  int num_dim = 3;

  double dt = update->dt;

  double time = update->dt*update->ntimestep;

  int groupbit = this->fixMLMOD->groupbit;

  // use .pt model from PyTorch for ML learned model for 
  // update (X,V) = Gamma(X,V) get model data
  ModelData_Dyn_ML1_Type *m_data 
    = (ModelData_Dyn_ML1_Type *) this->model_data;

  // initialize
  if (m_data->flag_init == 0) {

    // determine index list controlled by this fix 
    m_data->num_II = 2; m_data->II_size = m_data->num_II;
    m_data->II = (int *)malloc(sizeof(int)*m_data->II_size);
    m_data->II[0] = 0; m_data->II[1] = 1; 

    // get tensor input size given masks
    m_data->input_size = get_tensor_input_size(m_data->num_II,m_data->II,num_dim,
                                               m_data->mask_input, m_data->mask_list_n, m_data->mask_list,
                                               0, NULL);

    m_data->flag_init = 1;

  } // end flag_init

  // create tensors
  at::Tensor input_m = torch::zeros({m_data->input_size,1});
  //auto input_m_a = input_m.accessor<float,2>(); 
  at::TensorAccessor<float,2,at::DefaultPtrTraits,long> input_m_a = input_m.accessor<float,2>(); 
  std::vector<torch::jit::IValue> input_m_i{input_m};

  at::Tensor output_m; // leave un-initialized

  // construct the input for update (X,V) = Gamma(X,V,...) 
  build_tensor_from_masks(input_m_a,m_data->num_II,m_data->II,m_data->mask_input,
                          m_data->mask_list_n,m_data->mask_list,
                          groupbit,mask_atom,
                          x,v,f,type,time);

  // evaluate the model (X,V) = Gamma(X,V) and obtain tensor output
  // output 2D tensor assumed shape=[2*num_II*num_dim,1]
  output_m = m_data->Dyn2_model->forward(input_m_i).toTensor();

  int II1 = 0; int II2 = II1 + num_dim*m_data->num_II;
  set_db_vec_to_tensor_full(x,output_m,m_data->num_II,num_dim,m_data->II,II1);
  set_db_vec_to_tensor_full(v,output_m,m_data->num_II,num_dim,m_data->II,II2);
  
}

void DriverMLMOD::reset_dt()
{

}

void DriverMLMOD::post_force(int vflag) {


  if (this->model_type == MODEL_TYPE_dX_MF_Q1_ML1_Pair) { 
  } else if (this->model_type == MODEL_TYPE_dX_MF_Q1_ML1_N2N) { 
  } else if (this->model_type == MODEL_TYPE_F_ML1) { 
    post_force_F_ML1();
  } else if (this->model_type == MODEL_TYPE_F_X_ML1) { 
    post_force_F_X_ML1();
  } else if (this->model_type == MODEL_TYPE_F_Pair_ML1) { 
    post_force_F_Pair_ML1();
  } else if (this->model_type == MODEL_TYPE_QoI_ML1) { 
  } else if (this->model_type == MODEL_TYPE_Dyn_ML1) { 
  } else {
    if (flag_verbose >= 1) {
      printf("WARNING: Not recognized, this->model_type_str = %s, this->model_type = %d \n",
             this->model_type_str.c_str(),this->model_type);
    }
    
  }


}


/*****************************************************************************************/
/* Supporting codes */
/*****************************************************************************************/
void DriverMLMOD::init_attributes() {

}

void DriverMLMOD::write_info() {

}

// pass along the initialization
void DriverMLMOD::init_from_fix() {

  if (flag_verbose >= 1) {
    cout << endl;
    cout << "................................................................................" << endl;
    cout << "MLMOD package info: " << endl;
    cout << "URL = http://atzberger.org" << endl;
    cout << "Compile Date = " << COMPILE_DATE_TIME << endl;
    cout << "SVN = " << SVN_REV << endl;
    cout << "GIT = " << GIT_REV << endl;
    cout << "................................................................................" << endl;
    cout << endl; 
  }

}


double** DriverMLMOD::create_db_vec(int num_rows, int num_cols, 
                                    double **db_vec_block, 
                                    int *db_vec_size) {

  // if block is not already allocated (malloc)
  if ((*(db_vec_block) == 0) || (*(db_vec_size) <= 0)) {
    *(db_vec_block)
      = (double *)malloc(sizeof(double)*num_rows*num_cols);
    *(db_vec_size) = num_rows*num_cols;
  }
  
  // create the db_vec
  double **db_vec = (double **)malloc(sizeof(double *)*num_rows);

  // setup entries and initialize
  for (int i = 0; i < num_rows; i++) {
    db_vec[i] = *(db_vec_block)
	      + i*num_cols + 0; // row-major blocking
    for (int d = 0; d < num_cols; d++) {
      db_vec[i][d] = 0.0; // initialize
    }
  }

  // return the db_vec
  return db_vec;
 
}

void DriverMLMOD::set_db_vec_to_tensor_full(double **z,
                                            at::Tensor output_m,
                                            int num_rows, int num_cols, int *rowII, 
                                            int offset_tensor) {

  auto output_m_a = output_m.accessor<float,2>(); 

  if (rowII != NULL) { 
    for (int i = 0; i < num_rows; i++) {
      int iii = rowII[i];
      for (int d = 0; d < num_cols; d++) {
	int I1 = iii*num_cols + d + offset_tensor; 
	z[iii][d] = output_m_a[I1][0]; 
      }
    } // i
  } else { // no rowII restrictions
    for (int i = 0; i < num_rows; i++) {
      int iii = i;
      for (int d = 0; d < num_cols; d++) {
	int I1 = iii*num_cols + d + offset_tensor; 
	z[iii][d] = output_m_a[I1][0]; 
      }
    } // i
  }

}

/*
  Add where we assume tensor is same overall size as the z, 
  just reshaped with
  tensor.shape=[num_rows*num_cols,1] 
  z.shape=[num_rows,num_cols]

  z[iii][d] += output_m_a[I1][0],
  where   
  iii = rowII[i]; 
  I1 = iii*num_cols + d + offset_tensor

*/
void DriverMLMOD::add_tensor_full_to_db_vec(double **z,
                                            at::Tensor output_m,
                                            int num_rows, int num_cols, int *rowII, 
                                            int offset_tensor) {

  auto output_m_a = output_m.accessor<float,2>(); 

  if (rowII != NULL) { 
    for (int i = 0; i < num_rows; i++) {
      int iii = rowII[i];
      for (int d = 0; d < num_cols; d++) {
	int I1 = iii*num_cols + d + offset_tensor; 
	z[iii][d] += output_m_a[I1][0]; 
      }
    } // i
  } else { // no rowII restrictions
    for (int i = 0; i < num_rows; i++) {
      int iii = i;
      for (int d = 0; d < num_cols; d++) {
	int I1 = iii*num_cols + d + offset_tensor; 
	z[iii][d] += output_m_a[I1][0]; 
      }
    } // i
  }

}

/*
  Add where we assume tensor is just segment of the z, 
  just reshaped with
  tensor.shape=[rowII.size*num_cols,1] 
  z.shape=[rowII.size,num_cols]
  rowII.size=num_rows

  z[iii][d] += output_m_a[J][0],
  where   
  iii = rowII[i]; 
  J = i*num_cols + d + offset_tensor

*/
void DriverMLMOD::add_tensor_seg_to_db_vec(double **z,
                                           at::Tensor output_m,
                                           int num_rows, int num_cols, int *rowII, 
                                           int offset_tensor) {

  auto output_m_a = output_m.accessor<float,2>(); 
  int J = 0;

  if (rowII != NULL) { 
    for (int i = 0; i < num_rows; i++) {
      int iii = rowII[i];
      for (int d = 0; d < num_cols; d++) {
	z[iii][d] += output_m_a[J][0]; J++;
      }
    } // i
  } else { // no rowII restrictions
    for (int i = 0; i < num_rows; i++) {
      int iii = i;
      for (int d = 0; d < num_cols; d++) {
	z[iii][d] += output_m_a[J][0]; J++;
      }
    } // i
  }

}

void DriverMLMOD::set_to_zero_db_vec(double **z,
                                     int num_rows, int num_cols, int *rowII) {

  if (rowII != NULL) { 
    for (int i = 0; i < num_rows; i++) {
      int iii = rowII[i];
      for (int d = 0; d < num_cols; d++) {
	z[iii][d] = 0.0; 
      }
    } // i
  } else { // no rowII restrictions
    for (int i = 0; i < num_rows; i++) {
      int iii = i;
      for (int d = 0; d < num_cols; d++) {
	z[iii][d] = 0.0; 
      }
    } // i
  }

}

int DriverMLMOD::get_tensor_input_size(int num_II,int *II, int num_dim,
                                       int mask_input, int mask_list_n, int *mask_list,
                                       int groupbit, int *mask_atom) {
    int count = 0;

    // collect data for size of inputs
    if (mask_input & mask_list[IN_X]) { // X 
      count += num_II*num_dim;
    }

    if (mask_input & mask_list[IN_V]) { // V
      count += num_II*num_dim;
    }

    if (mask_input & mask_list[IN_F]) { // F
      count += num_II*num_dim;
    }

    if (mask_input & mask_list[IN_Type]) { // Type
      count += num_II;
    }

    if (mask_input & mask_list[IN_Time]) { // Time
      count += 1;
    }

    return count; 
}

void DriverMLMOD::build_tensor_from_masks(at::TensorAccessor<float,2,at::DefaultPtrTraits,long> input_m_a,
                                          int num_indices,int *indices,
                                          int mask_input, int mask_list_n, int *mask_list,
                                          int groupbit, int *mask_atom,
                                          double **x,double **v,double **f, int *type, double time) {

   // if indices == NULL, then all indices used
   // if mask_atom == NULL, then all atoms used
   // if mask_list == NULL, then all input types are used  

  // (@optimize later using neigh list thresholds)
  int J = 0; int num_dim = 3; int flag; int ii;

  // X
  if ((mask_input == NULL) || (mask_input & mask_list[IN_X])) { 
    for (int i = 0; i < num_indices; i++) {
      if (indices != NULL) {
        ii = indices[i];      
      } else {
        ii = i;
      }
      if ((mask_atom == NULL) || (mask_atom[ii] & groupbit)) {
        flag = 1;
      } else {
        flag = 0;
      }
      if (flag == 1) {
        for (int d = 0; d < num_dim; d++) {
          input_m_a[J][0] = x[ii][d];
          J += 1;
        }
      }
    }
  }

  // V
  if ((mask_input == NULL) || (mask_input & mask_list[IN_V])) { 
    for (int i = 0; i < num_indices; i++) {
      if (indices != NULL) {
        ii = indices[i];      
      } else {
        ii = i;
      }
      if ((mask_atom == NULL) || (mask_atom[ii] & groupbit)) {
        flag = 1;
      } else {
        flag = 0;
      }
      if (flag == 1) {
	for (int d = 0; d < num_dim; d++) {
	  input_m_a[J][0] = v[ii][d];
	  J += 1;
	}
      }
    }
  }

  // F
  if ((mask_input == NULL) || (mask_input & mask_list[IN_F])) { 
    for (int i = 0; i < num_indices; i++) {
      if (indices != NULL) {
        ii = indices[i];      
      } else {
        ii = i;
      }
      if ((mask_atom == NULL) || (mask_atom[ii] & groupbit)) {
        flag = 1;
      } else {
        flag = 0;
      }
      if (flag == 1) {
	for (int d = 0; d < num_dim; d++) {
	  input_m_a[J][0] = f[ii][d];
	  J += 1;
	}
      }
    }
  }

  // Type
  if ((mask_input == NULL) || (mask_input & mask_list[IN_Type])) { 
    for (int i = 0; i < num_indices; i++) {
      if (indices != NULL) {
        ii = indices[i];      
      } else {
        ii = i;
      }
      if ((mask_atom == NULL) || (mask_atom[ii] & groupbit)) {
        flag = 1;
      } else {
        flag = 0;
      }
      if (flag == 1) {
        input_m_a[J][0] = (double)type[ii];  // warning cast, not ideal
        J += 1;
      }
    }
  }

  // time
  if ((mask_input == NULL) || (mask_input & mask_list[IN_Time])) { 
    input_m_a[J][0] = time; 
    J += 1;
  }

}

void DriverMLMOD::get_indices_for_group(int groupbit,int *mask_atom, int nlocal, int *II_size_ptr, int **II_ptr, int *num_II_ptr) {
  int num_II = *num_II_ptr; 
  int *II = *II_ptr;
  int II_size = *II_size_ptr;
  int count, J;
   
  count = 0;
  for (int i = 0; i < nlocal; i++) {
    if (groupbit & mask_atom[i]) {
      count++;
    }
  }
  num_II = count; *num_II_ptr = num_II;
  
  // allocate memory if II_size is too small
  if ((II == NULL) || (II_size < num_II)) {
    if ((II != NULL) && (II_size > 0)) { free(II); }
    II_size = num_II;
    II = (int *)malloc(II_size*sizeof(int));
    *II_ptr = II; *II_size_ptr = II_size; 
  }

  J = 0;
  for (int i = 0; i < nlocal; i++) {
    if (groupbit & mask_atom[i]) {
      II[J] = i; J++;
    }
  }
  
}

// parser
void DriverMLMOD::parse_xml_params(char *filename) {

  XMLDocument doc;
  doc.LoadFile(filename);

  if (flag_verbose >= 1) {
    printf("parsing filename = %s\n",filename);
  }

  const char *model_type_str;
  XMLElement* model_data_element 
    = doc.FirstChildElement("MLMOD")->FirstChildElement("model_data");
  model_data_element->QueryStringAttribute("type",&model_type_str);
  if (flag_verbose >= 1) {
    printf("model_data: type = %s \n",model_type_str);
  }

  this->model_type_str = model_type_str;

  // parse based on the model_type
  if (this->model_type_str.compare(MODEL_TYPE_STR_dX_MF) == 0) {
    parse_xml_params_dX_MF(model_data_element);
  } else if (this->model_type_str.compare(MODEL_TYPE_STR_dX_MF_ML1) == 0) {
    parse_xml_params_dX_MF_ML1(model_data_element);
  } else if (this->model_type_str.compare(MODEL_TYPE_STR_dX_MF_Q1_ML1_Pair) == 0) {
    parse_xml_params_dX_MF_Q1_ML1_Pair(model_data_element);
  } else if (this->model_type_str.compare(MODEL_TYPE_STR_dX_MF_Q1_ML1_N2N) == 0) {
    parse_xml_params_dX_MF_Q1_ML1_N2N(model_data_element);
  } else if (this->model_type_str.compare(MODEL_TYPE_STR_F_ML1) == 0) {
    parse_xml_params_F_ML1(model_data_element);
  } else if (this->model_type_str.compare(MODEL_TYPE_STR_F_X_ML1) == 0) {
    parse_xml_params_F_X_ML1(model_data_element);
  } else if (this->model_type_str.compare(MODEL_TYPE_STR_F_Pair_ML1) == 0) {
    parse_xml_params_F_Pair_ML1(model_data_element);
  } else if (this->model_type_str.compare(MODEL_TYPE_STR_QoI_ML1) == 0) {
    parse_xml_params_QoI_ML1(model_data_element);
  } else if (this->model_type_str.compare(MODEL_TYPE_STR_Dyn_ML1) == 0) {
    parse_xml_params_Dyn_ML1(model_data_element);
  } else {
    printf("Not recognonized model_data type = %s \n",this->model_type_str.c_str());
  }

}

// parser case specific data
void DriverMLMOD::parse_xml_params_dX_MF(XMLElement *model_data_element) {

  const char *base_name;
  const char *base_dir;
  const char *mask_fix_str;
  const char *mask_input_str;

  this->model_type = MODEL_TYPE_dX_MF;

  ModelData_dX_MF_Type *m_data 
    = (ModelData_dX_MF_Type *)malloc(sizeof(ModelData_dX_MF_Type));

  m_data->mask_input_str = new string();
  m_data->mask_fix_str = new string();

  this->model_data = (void *) m_data;

  XMLElement* eta_element = model_data_element->FirstChildElement("eta");
  eta_element->QueryDoubleAttribute("value",&m_data->eta);
  if (flag_verbose >= 2) {
    printf("model_data: eta = %g \n",m_data->eta);
  }

  XMLElement* a_element = model_data_element->FirstChildElement("a");
  a_element->QueryDoubleAttribute("value",&m_data->a);
  if (flag_verbose >= 2) {
    printf("model_data: a = %g \n",m_data->a);
  }

  XMLElement* epsilon_element = model_data_element->FirstChildElement("epsilon");
  epsilon_element->QueryDoubleAttribute("value",&m_data->epsilon);
  if (flag_verbose >= 2) {
    printf("model_data: epsilon = %g \n",m_data->epsilon);
  }

  XMLElement* mask_fix_element 
     = model_data_element->FirstChildElement("mask_fix");
  if (mask_fix_element != NULL) {
    mask_fix_element->QueryStringAttribute("value",&mask_fix_str);
    *m_data->mask_fix_str = mask_fix_str;
  } else {
    *m_data->mask_fix_str = "FINAL_INTEGRATE";  /* default value */
  }

  if (flag_verbose >= 2) {
    printf("model_data: mask_fix_str = %s \n",m_data->mask_fix_str->c_str());
  } 

  parse_mask_fix_str(m_data->mask_fix_str->c_str(),&m_data->mask_fix);

  XMLElement* mask_input_str_element 
     = model_data_element->FirstChildElement("mask_input");
  if (mask_input_str_element != NULL) {
    mask_input_str_element->QueryStringAttribute("value",&mask_input_str);
    *m_data->mask_input_str = mask_input_str;
  } else {
    *m_data->mask_input_str = IN_X_str;  /* default value */
  }
  if (flag_verbose >= 2) {
    printf("model_data: mask_input = %s \n",m_data->mask_input_str->c_str());
  }
  // setup mask based on the input string "X V F Type Time"
  parse_mask_input_str(m_data->mask_input_str->c_str(),&m_data->mask_input,
                       &m_data->mask_list_n, &m_data->mask_list);
}


// parser case specific data
void DriverMLMOD::parse_xml_params_dX_MF_ML1(XMLElement *model_data_element) {
  
  const char *M_ii_filename;
  const char *M_ij_filename;
  
  const char *base_name;
  const char *base_dir;
  const char *mask_fix_str;
  const char *mask_input_str;

  this->model_type = MODEL_TYPE_dX_MF_ML1;

  ModelData_dX_MF_ML1_Type *m_data 
    = (ModelData_dX_MF_ML1_Type *)malloc(sizeof(ModelData_dX_MF_ML1_Type));

  m_data->M_ii_model = new torch::jit::Module();
  m_data->M_ij_model = new torch::jit::Module();

  m_data->M_ii_filename = new string();
  m_data->M_ij_filename = new string();
  m_data->mask_input_str = new string();
  m_data->mask_fix_str = new string();
  
  // m_data->M_ii_model = new torch::jit::Module();
  // m_data->M_ij_model = new torch::jit::Module();

  this->model_data = (void *) m_data;

  XMLElement* M_ii_filename_element 
     = model_data_element->FirstChildElement("M_ii_filename");
  M_ii_filename_element->QueryStringAttribute("value",&M_ii_filename);
  *m_data->M_ii_filename= M_ii_filename;
  
  if (flag_verbose >= 2) {
    printf("model_data: M_ii_filename = %s \n",m_data->M_ii_filename->c_str());
  }

  //printf("WARNING: Using hard-coded filename currently \n");
  // load the .pt PyTorch model 
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    *m_data->M_ii_model = torch::jit::load(m_data->M_ii_filename->c_str());
  } catch (const c10::Error& e) {
    std::cerr << "ERROR: error loading the torch model \n";
    exit(1);
    //return -1; 
  }

  XMLElement* M_ij_filename_element 
     = model_data_element->FirstChildElement("M_ij_filename");
  M_ij_filename_element->QueryStringAttribute("value",&M_ij_filename);
  *m_data->M_ij_filename = M_ij_filename;
  if (flag_verbose >= 2) {
    printf("model_data: M_ij_filename = %s \n",m_data->M_ij_filename->c_str());
  }

  //printf("WARNING: Using hard-coded filename currently \n");
  // load the .pt PyTorch model 
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    *m_data->M_ij_model = torch::jit::load(m_data->M_ij_filename->c_str());
  } catch (const c10::Error& e) {
    std::cerr << "ERROR: error loading the torch model \n";
    exit(1);
    //return -1; 
  }

}

// parser case specific data
void DriverMLMOD::parse_xml_params_dX_MF_Q1_ML1_Pair(XMLElement *model_data_element) {
  
  const char *M_ii_filename;
  const char *M_ij_filename;

  const char *base_name;
  const char *base_dir;
  const char *mask_fix_str;
  const char *mask_input_str;
  
  this->model_type = MODEL_TYPE_dX_MF_Q1_ML1_Pair;

  ModelData_dX_MF_Q1_ML1_Pair_Type *m_data 
    = (ModelData_dX_MF_Q1_ML1_Pair_Type *)malloc(sizeof(ModelData_dX_MF_Q1_ML1_Pair_Type));

  m_data->M_ii_model = new torch::jit::Module();
  m_data->M_ij_model = new torch::jit::Module();

  m_data->M_ii_filename = new string();
  m_data->M_ij_filename = new string();
  m_data->mask_input_str = new string();
  m_data->mask_fix_str = new string();
  
  // m_data->M_ii_model = new torch::jit::Module();
  // m_data->M_ij_model = new torch::jit::Module();

  this->model_data = (void *) m_data;

  m_data->flag_init = 0;
  m_data->prev_v_size = 0; m_data->store_x_size = 0; m_data->xi_size = 0;
  m_data->Q_xi_size = 0; m_data->II_size = 0;
  m_data->avg_diff_M_size[0] = 0; m_data->avg_diff_M_size[1] = 0;
  m_data->M_size[0] = 0; m_data->M_size[1] = 0;
  m_data->Q_size[0] = 0; m_data->Q_size[1] = 0; m_data->A_size = 0;

  XMLElement* M_ii_filename_element 
     = model_data_element->FirstChildElement("M_ii_filename");
  M_ii_filename_element->QueryStringAttribute("value",&M_ii_filename);
  *m_data->M_ii_filename = M_ii_filename;
  if (flag_verbose >= 2) {
    printf("model_data: M_ii_filename = %s \n",m_data->M_ii_filename->c_str());
  }

  //printf("WARNING: Using hard-coded filename currently \n");
  // load the .pt PyTorch model 
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    *m_data->M_ii_model = torch::jit::load(m_data->M_ii_filename->c_str());
  } catch (const c10::Error& e) {
    std::cerr << "ERROR: error loading the torch model \n";
    exit(-1);
    //return -1; 
  }

  XMLElement* M_ij_filename_element 
     = model_data_element->FirstChildElement("M_ij_filename");
  M_ij_filename_element->QueryStringAttribute("value",&M_ij_filename);
  *m_data->M_ij_filename = M_ij_filename;
  if (flag_verbose >= 2) {
    printf("model_data: M_ij_filename = %s \n",m_data->M_ij_filename->c_str());
  }

  //printf("WARNING: Using hard-coded filename currently \n");
  // load the .pt PyTorch model 
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    *m_data->M_ij_model = torch::jit::load(m_data->M_ij_filename->c_str());
  } catch (const c10::Error& e) {
    std::cerr << "ERROR: error loading the torch model \n";
    exit(-1);
    //return -1; 
  }

  XMLElement* delta_element = model_data_element->FirstChildElement("delta");
  delta_element->QueryDoubleAttribute("value",&m_data->delta);
  if (flag_verbose >= 2) {
    printf("model_data: delta = %g \n",m_data->delta);
  }

  XMLElement* KBT_element = model_data_element->FirstChildElement("KBT");
  KBT_element->QueryDoubleAttribute("value",&m_data->KBT);
  if (flag_verbose >= 2) {
    printf("model_data: KBT = %g \n",m_data->KBT);
  }

  XMLElement* flag_stochastic_element 
    = model_data_element->FirstChildElement("flag_stochastic");
  flag_stochastic_element->QueryIntAttribute("value",&m_data->flag_stochastic);
  if (flag_verbose >= 2) {
    printf("model_data: flag_stochastic = %d \n",m_data->flag_stochastic);
  }

  if (m_data->KBT == 0) {
    if (flag_verbose >= 1) {
      cout << "WARNING: KBT = 0, so setting flag_stochastic = 0.\n";
    }
    m_data->flag_stochastic = 0; 
  }

  XMLElement* flag_thm_drift_element 
    = model_data_element->FirstChildElement("flag_thm_drift");
  flag_thm_drift_element->QueryIntAttribute("value",&m_data->flag_thm_drift);
  if (flag_verbose >= 2) {
    printf("model_data: flag_thm_drift = %d \n",m_data->flag_thm_drift);
  }

}

// parser case specific data
void DriverMLMOD::parse_xml_params_dX_MF_Q1_ML1_N2N(XMLElement *model_data_element) {
  
  const char *M_filename;
  
  const char *base_name;
  const char *base_dir;
  const char *mask_fix_str;
  const char *mask_input_str;

  this->model_type = MODEL_TYPE_dX_MF_Q1_ML1_N2N;

  ModelData_dX_MF_Q1_ML1_N2N_Type *m_data 
    = (ModelData_dX_MF_Q1_ML1_N2N_Type *)malloc(sizeof(ModelData_dX_MF_Q1_ML1_N2N_Type));

  m_data->M_model = new torch::jit::Module();
  m_data->M_filename = new string();
  m_data->mask_input_str = new string();
  m_data->mask_fix_str = new string();
  
  // m_data->M_model = new torch::jit::Module();

  this->model_data = (void *) m_data;

  m_data->flag_init = 0;
  m_data->prev_v_size = 0; m_data->store_x_size = 0; m_data->xi_size = 0;
  m_data->Q_xi_size = 0; m_data->II_size = 0;
  m_data->avg_diff_M_size[0] = 0; m_data->avg_diff_M_size[1] = 0;
  m_data->M_size[0] = 0; m_data->M_size[1] = 0;
  m_data->Q_size[0] = 0; m_data->Q_size[1] = 0; m_data->A_size = 0;

  XMLElement* M_filename_element 
     = model_data_element->FirstChildElement("M_filename");
  M_filename_element->QueryStringAttribute("value",&M_filename);
  *m_data->M_filename = M_filename;
  if (flag_verbose >= 2) {
    printf("model_data: M_filename = %s \n",m_data->M_filename->c_str());
  }
  //printf("WARNING: Using hard-coded filename currently \n");
  // load the .pt PyTorch model 
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    *m_data->M_model = torch::jit::load(m_data->M_filename->c_str());
  } catch (const c10::Error& e) {
    std::cerr << "ERROR: error loading the torch model \n";
    exit(-1);
    //return -1; 
  }

  XMLElement* delta_element = model_data_element->FirstChildElement("delta");
  delta_element->QueryDoubleAttribute("value",&m_data->delta);
  if (flag_verbose >= 2) {
    printf("model_data: delta = %g \n",m_data->delta);
  }
  XMLElement* KBT_element = model_data_element->FirstChildElement("KBT");
  KBT_element->QueryDoubleAttribute("value",&m_data->KBT);
  if (flag_verbose >= 2) {
    printf("model_data: KBT = %g \n",m_data->KBT);
  }
  XMLElement* flag_stochastic_element 
    = model_data_element->FirstChildElement("flag_stochastic");
  flag_stochastic_element->QueryIntAttribute("value",&m_data->flag_stochastic);
  if (flag_verbose >= 2) {
    printf("model_data: flag_stochastic = %d \n",m_data->flag_stochastic);
  }
  if (m_data->KBT == 0) {
    if (flag_verbose >= 1) {
      cout << "WARNING: KBT = 0, so setting flag_stochastic = 0.\n";
     }   
    m_data->flag_stochastic = 0; 
  }

  XMLElement* flag_thm_drift_element 
    = model_data_element->FirstChildElement("flag_thm_drift");
  flag_thm_drift_element->QueryIntAttribute("value",&m_data->flag_thm_drift);
  if (flag_verbose >= 2) {
    printf("model_data: flag_thm_drift = %d \n",m_data->flag_thm_drift);
  }
}

// parser case specific data
void DriverMLMOD::parse_xml_params_F_ML1(XMLElement *model_data_element) {
  
  const char *F_filename;
  const char *base_name;
  const char *base_dir;
  const char *mask_fix_str;
  const char *mask_input_str;
  
  this->model_type = MODEL_TYPE_F_ML1;

  ModelData_F_ML1_Type *m_data 
    = (ModelData_F_ML1_Type *)malloc(sizeof(ModelData_F_ML1_Type));

  m_data->F_model = new torch::jit::Module();

  m_data->base_name = new string();
  m_data->base_dir = new string();
  m_data->F_filename = new string();
  m_data->mask_input_str = new string();
  m_data->mask_fix_str = new string();
  
  m_data->flag_init = 0;
  m_data->num_II = 0;
  m_data->II_size = 0;
  m_data->II = NULL;

  this->model_data = (void *) m_data;

  XMLElement* base_name_element 
     = model_data_element->FirstChildElement("base_name");
  base_name_element->QueryStringAttribute("value",&base_name);
  *m_data->base_name = base_name;
  if (flag_verbose >= 2) {
    printf("model_data: base_name = %s \n",m_data->base_name->c_str());
  }
  XMLElement* base_dir_element 
     = model_data_element->FirstChildElement("base_dir");
  base_dir_element->QueryStringAttribute("value",&base_dir);
  *m_data->base_dir = base_dir;
  if (flag_verbose >= 2) {
    printf("model_data: base_dir = %s \n",m_data->base_dir->c_str());
  }
  // create a new directory for output for mlmod
  int rv = mkdir(m_data->base_dir->c_str(),0755);
    
  if ((rv == -1) && (errno != EEXIST)) {  
    cout << "Failed making directory path = " << m_data->base_dir->c_str() << endl;
  }

  XMLElement* F_filename_element 
     = model_data_element->FirstChildElement("F_filename");
  F_filename_element->QueryStringAttribute("value",&F_filename);
  *m_data->F_filename = F_filename;
  if (flag_verbose >= 2) {
    printf("model_data: F_filename = %s \n",m_data->F_filename->c_str());
  }
  //printf("WARNING: Using hard-coded filename currently \n");
  // load the .pt PyTorch model 
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    *m_data->F_model = torch::jit::load(m_data->F_filename->c_str());
  } catch (const c10::Error& e) {
    std::cerr << "ERROR: error loading the torch model \n";
    exit(-1);
    //return -1; 
  }


  XMLElement* mask_fix_element 
     = model_data_element->FirstChildElement("mask_fix");
  if (mask_fix_element != NULL) {
    mask_fix_element->QueryStringAttribute("value",&mask_fix_str);
    *m_data->mask_fix_str = mask_fix_str;
    if (flag_verbose >= 2) {
      printf("model_data: mask_fix_str = %s \n",m_data->mask_fix_str->c_str());
    } 
   } else {
    std::cerr << "ERROR: missing mask_fix in MLMOD parameter file. \n";
    exit(-1);  
  }

  parse_mask_fix_str(m_data->mask_fix_str->c_str(),&m_data->mask_fix);

  XMLElement* mask_input_str_element 
     = model_data_element->FirstChildElement("mask_input");
  mask_input_str_element->QueryStringAttribute("value",&mask_input_str);
  *m_data->mask_input_str = mask_input_str;
  if (flag_verbose >= 2) {
    printf("model_data: mask_input = %s \n",m_data->mask_input_str->c_str());
  }
  // setup mask based on the input string "X V F Type"
  parse_mask_input_str(m_data->mask_input_str->c_str(),&m_data->mask_input,
                       &m_data->mask_list_n, &m_data->mask_list);
  
}

// parser case specific data
void DriverMLMOD::parse_xml_params_F_X_ML1(XMLElement *model_data_element) {
  
  const char *F_filename;
  const char *base_name;
  const char *base_dir;
  const char *mask_fix_str;
  const char *mask_input_str;
  
  this->model_type = MODEL_TYPE_F_X_ML1;

  ModelData_F_X_ML1_Type *m_data 
    = (ModelData_F_X_ML1_Type *)malloc(sizeof(ModelData_F_X_ML1_Type));

  m_data->F_model = new torch::jit::Module();

  m_data->base_name = new string();
  m_data->base_dir = new string();
  m_data->F_filename = new string();
  m_data->mask_input_str = new string();
  m_data->mask_fix_str = new string();

  m_data->flag_init = 0;
  m_data->num_II = 0;
  m_data->II_size = 0;
  m_data->II = NULL;

  this->model_data = (void *) m_data;

  XMLElement* base_name_element 
     = model_data_element->FirstChildElement("base_name");
  base_name_element->QueryStringAttribute("value",&base_name);
  *m_data->base_name = base_name;
  if (flag_verbose >= 2) {
    printf("model_data: base_name = %s \n",m_data->base_name->c_str());
  }
  XMLElement* base_dir_element 
     = model_data_element->FirstChildElement("base_dir");
  base_dir_element->QueryStringAttribute("value",&base_dir);
  *m_data->base_dir = base_dir;
  if (flag_verbose >= 2) {
    printf("model_data: base_dir = %s \n",m_data->base_dir->c_str());
  }
  // create a new directory for output for mlmod
  int rv = mkdir(m_data->base_dir->c_str(),0755);
    
  if ((rv == -1) && (errno != EEXIST)) {  
    if (flag_verbose >= 1) {
      cout << "Failed making directory path = " << m_data->base_dir->c_str() << endl;
    } 
  }

  XMLElement* F_filename_element 
     = model_data_element->FirstChildElement("F_filename");
  F_filename_element->QueryStringAttribute("value",&F_filename);
  *m_data->F_filename = F_filename;
  if (flag_verbose >= 2) {
    printf("model_data: F_filename = %s \n",m_data->F_filename->c_str());
  }
  //printf("WARNING: Using hard-coded filename currently \n");
  // load the .pt PyTorch model 
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    *m_data->F_model = torch::jit::load(m_data->F_filename->c_str());
  } catch (const c10::Error& e) {
    std::cerr << "ERROR: error loading the torch model \n";
    exit(-1);
    //return -1; 
  }


  XMLElement* mask_fix_element 
     = model_data_element->FirstChildElement("mask_fix");
  if (mask_fix_element != NULL) {
    mask_fix_element->QueryStringAttribute("value",&mask_fix_str);
    *m_data->mask_fix_str = mask_fix_str;
    if (flag_verbose >= 2) {
      printf("model_data: mask_fix_str = %s \n",m_data->mask_fix_str->c_str());
    } 
  } else {
    std::cerr << "ERROR: missing mask_fix in MLMOD parameter file. \n";
    exit(-1);  
  }

  parse_mask_fix_str(m_data->mask_fix_str->c_str(),&m_data->mask_fix);

  XMLElement* mask_input_str_element 
     = model_data_element->FirstChildElement("mask_input");
  mask_input_str_element->QueryStringAttribute("value",&mask_input_str);
  *m_data->mask_input_str = mask_input_str;
  if (flag_verbose >= 2) {
    printf("model_data: mask_input = %s \n",m_data->mask_input_str->c_str());
  }
  // setup mask based on the input string "X V F Type"
  parse_mask_input_str(m_data->mask_input_str->c_str(),&m_data->mask_input,
                       &m_data->mask_list_n, &m_data->mask_list);
  
}

// parser case specific data
void DriverMLMOD::parse_xml_params_F_Pair_ML1(XMLElement *model_data_element) {
  
  const char *F_filename;
  const char *base_name;
  const char *base_dir;
  const char *mask_fix_str;
  const char *mask_input_str;
  
  this->model_type = MODEL_TYPE_F_Pair_ML1;

  ModelData_F_Pair_ML1_Type *m_data 
    = (ModelData_F_Pair_ML1_Type *)malloc(sizeof(ModelData_F_Pair_ML1_Type));

  m_data->F_model = new torch::jit::Module();

  m_data->base_name = new string();
  m_data->base_dir = new string();
  m_data->F_filename = new string();
  m_data->mask_input_str = new string();
  m_data->mask_fix_str = new string();
  
  m_data->flag_init = 0;
  m_data->num_II = 0;
  m_data->II_size = 0;
  m_data->II = NULL;

  this->model_data = (void *) m_data;

  XMLElement* base_name_element 
     = model_data_element->FirstChildElement("base_name");
  base_name_element->QueryStringAttribute("value",&base_name);
  *m_data->base_name = base_name;
  if (flag_verbose >= 2) {
    printf("model_data: base_name = %s \n",m_data->base_name->c_str());
  }
  XMLElement* base_dir_element 
     = model_data_element->FirstChildElement("base_dir");
  base_dir_element->QueryStringAttribute("value",&base_dir);
  *m_data->base_dir = base_dir;
  if (flag_verbose >= 2) {
    printf("model_data: base_dir = %s \n",m_data->base_dir->c_str());
  }
  // create a new directory for output for mlmod
  int rv = mkdir(m_data->base_dir->c_str(),0755);
    
  if ((rv == -1) && (errno != EEXIST)) {  
    if (flag_verbose >= 1) {
      cout << "Failed making directory path = " << m_data->base_dir->c_str() << endl;
    } 
  }

  XMLElement* F_filename_element 
     = model_data_element->FirstChildElement("F_filename");
  F_filename_element->QueryStringAttribute("value",&F_filename);
  *m_data->F_filename = F_filename;
  if (flag_verbose >= 2) {
    printf("model_data: F_filename = %s \n",m_data->F_filename->c_str());
  }
  //printf("WARNING: Using hard-coded filename currently \n");
  // load the .pt PyTorch model 
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    *m_data->F_model = torch::jit::load(m_data->F_filename->c_str());
  } catch (const c10::Error& e) {
    std::cerr << "ERROR: error loading the torch model \n";
    exit(-1);
    //return -1; 
  }


  XMLElement* mask_fix_element 
     = model_data_element->FirstChildElement("mask_fix");
  if (mask_fix_element != NULL) {
    mask_fix_element->QueryStringAttribute("value",&mask_fix_str);
    *m_data->mask_fix_str = mask_fix_str;
    if (flag_verbose >= 2) {
      printf("model_data: mask_fix_str = %s \n",m_data->mask_fix_str->c_str());
    } 
   } else {
    std::cerr << "ERROR: missing mask_fix in MLMOD parameter file. \n";
    exit(-1);  
  }

  parse_mask_fix_str(m_data->mask_fix_str->c_str(),&m_data->mask_fix);

  XMLElement* mask_input_str_element 
     = model_data_element->FirstChildElement("mask_input");
  mask_input_str_element->QueryStringAttribute("value",&mask_input_str);
  *m_data->mask_input_str = mask_input_str;
  if (flag_verbose >= 2) {
    printf("model_data: mask_input = %s \n",m_data->mask_input_str->c_str());
  }
  // setup mask based on the input string "X V F Type"
  parse_mask_input_str(m_data->mask_input_str->c_str(),&m_data->mask_input,
                       &m_data->mask_list_n, &m_data->mask_list);
  
}


// parser case specific data
void DriverMLMOD::parse_xml_params_QoI_ML1(XMLElement *model_data_element) {
  
  const char *base_name;
  const char *base_dir;

  const char *mask_input_str;
  const char *mask_fix_str;
  
  const char *qoi_filename;

  this->model_type = MODEL_TYPE_QoI_ML1;

  ModelData_QoI_ML1_Type *m_data 
    = (ModelData_QoI_ML1_Type *)malloc(sizeof(ModelData_QoI_ML1_Type));

  m_data->QoI_model = new torch::jit::Module();

  m_data->qoi_filename = new string();
  m_data->base_name = new string();
  m_data->base_dir = new string();
  m_data->mask_input_str = new string();
  m_data->mask_fix_str = new string();
  
  m_data->flag_init = 0;
  m_data->num_II = 0;
  m_data->II_size = 0;
  m_data->II = NULL;

  this->model_data = (void *) m_data;

  XMLElement* base_name_element 
     = model_data_element->FirstChildElement("base_name");
  base_name_element->QueryStringAttribute("value",&base_name);
  *m_data->base_name = base_name;
  if (flag_verbose >= 2) {
    printf("model_data: base_name = %s \n",m_data->base_name->c_str());
  }
  XMLElement* base_dir_element 
     = model_data_element->FirstChildElement("base_dir");
  if (base_dir_element != NULL) {
    base_dir_element->QueryStringAttribute("value",&base_dir);
    *m_data->base_dir = base_dir;
    if (flag_verbose >= 2) {
      printf("model_data: base_dir = %s \n",m_data->base_dir->c_str());
    } 
  } else {
    *m_data->base_dir = "./mlmod"; // default dir
  }

  // create a new directory for output for mlmod
  int rv = mkdir(m_data->base_dir->c_str(),0755);
    
  if ((rv == -1) && (errno != EEXIST)) {  
    if (flag_verbose >= 1) {
      cout << "Failed making directory path = " << m_data->base_dir->c_str() << endl;
    } 
  }

  XMLElement* qoi_filename_element 
     = model_data_element->FirstChildElement("qoi_filename");
  qoi_filename_element->QueryStringAttribute("value",&qoi_filename);
  *m_data->qoi_filename = qoi_filename;
  if (flag_verbose >= 2) {
    printf("model_data: qoi_filename = %s \n",m_data->qoi_filename->c_str());
  }
  //printf("WARNING: Using hard-coded filename currently \n");
  // load the .pt PyTorch model 
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    *m_data->QoI_model = torch::jit::load(m_data->qoi_filename->c_str());
  } catch (const c10::Error& e) {
    std::cerr << "ERROR: error loading the torch model \n";
    exit(-1);
    //return -1; 
  }

  XMLElement* mask_fix_element 
     = model_data_element->FirstChildElement("mask_fix");
  if (mask_fix_element != NULL) {
    mask_fix_element->QueryStringAttribute("value",&mask_fix_str);
    *m_data->mask_fix_str = mask_fix_str;
    if (flag_verbose >= 2) {
      printf("model_data: mask_fix_str = %s \n",m_data->mask_fix_str->c_str());
    } 
  } else {
    std::cerr << "ERROR: missing mask_fix in MLMOD parameter file. \n";
    exit(-1);  
  }

  parse_mask_fix_str(m_data->mask_fix_str->c_str(),&m_data->mask_fix);
  //this->fixMLMOD->mask = m_data->mask_fix; // warning: double check if need further triggers in lammps
  //setmask(); // double-check this sets correct fields and triggers in lammps

  XMLElement* mask_input_str_element 
     = model_data_element->FirstChildElement("mask_input");
  mask_input_str_element->QueryStringAttribute("value",&mask_input_str);
  *m_data->mask_input_str = mask_input_str;
  if (flag_verbose >= 2) {
    printf("model_data: mask_input = %s \n",m_data->mask_input_str->c_str());
  }
  // setup mask based on the input string "X V F Type"
  parse_mask_input_str(m_data->mask_input_str->c_str(),&m_data->mask_input,
                       &m_data->mask_list_n, &m_data->mask_list);
  
  XMLElement* skip_step_element 
     = model_data_element->FirstChildElement("skip_step");
  if (skip_step_element != NULL) {
    skip_step_element->QueryIntAttribute("value",&m_data->skip_step);
  } else {
    m_data->skip_step = 1;
  }
  fixMLMOD->nevery = m_data->skip_step; /* syncronize with lammps skip step */
  if (flag_verbose >= 2) {
    printf("model_data: skip_step = %d \n",m_data->skip_step);
  }
}

// parser case specific data
void DriverMLMOD::parse_xml_params_Dyn_ML1(XMLElement *model_data_element) {
  
  const char *base_name;
  const char *base_dir;

  const char *mask_input_str;
  const char *mask_fix_str;
  
  const char *dyn1_filename;
  const char *dyn2_filename;
  
  this->model_type = MODEL_TYPE_Dyn_ML1;

  ModelData_Dyn_ML1_Type *m_data 
    = (ModelData_Dyn_ML1_Type *)malloc(sizeof(ModelData_Dyn_ML1_Type));

  m_data->base_name = new string();
  m_data->base_dir = new string();

  m_data->Dyn1_model = new torch::jit::Module();
  m_data->Dyn2_model = new torch::jit::Module();

  m_data->dyn1_filename = new string();
  m_data->dyn2_filename = new string();
  m_data->mask_input_str = new string();
  m_data->mask_fix_str = new string();
  
  m_data->flag_init = 0;
  m_data->num_II = 0;
  m_data->II_size = 0;
  m_data->II = NULL;

  // m_data->Dyn1_model = new torch::jit::Module();
  // m_data->Dyn2_model = new torch::jit::Module();

  this->model_data = (void *) m_data;

  XMLElement* base_name_element 
     = model_data_element->FirstChildElement("base_name");
  base_name_element->QueryStringAttribute("value",&base_name);
  *m_data->base_name = base_name;
  if (flag_verbose >= 2) {
    printf("model_data: base_name = %s \n",m_data->base_name->c_str());
  }
  XMLElement* base_dir_element 
     = model_data_element->FirstChildElement("base_dir");
  if (base_dir_element != NULL) {
    base_dir_element->QueryStringAttribute("value",&base_dir);
    *m_data->base_dir = base_dir;
    if (flag_verbose >= 2) {
      printf("model_data: base_dir = %s \n",m_data->base_dir->c_str());
    } 
  } else {
    *m_data->base_dir = "./mlmod"; // default dir
  }

  // create a new directory for output for mlmod
  int rv = mkdir(m_data->base_dir->c_str(),0755);
    
  if ((rv == -1) && (errno != EEXIST)) {  
    cout << "Failed making directory path = " << m_data->base_dir->c_str() << endl;
  }


  XMLElement* dyn1_filename_element 
     = model_data_element->FirstChildElement("dyn1_filename");
  dyn1_filename_element->QueryStringAttribute("value",&dyn1_filename);
  *m_data->dyn1_filename = dyn1_filename;
  if (flag_verbose >= 2) {
    printf("model_data: dyn1_filename = %s \n",m_data->dyn1_filename->c_str());
  }
  //printf("WARNING: Using hard-coded filename currently \n");
  // load the .pt PyTorch model 
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    *m_data->Dyn1_model = torch::jit::load(m_data->dyn1_filename->c_str());
  } catch (const c10::Error& e) {
    std::cerr << "ERROR: error loading the torch model \n";
    exit(-1);
    //return -1; 
  }

  XMLElement* dyn2_filename_element 
     = model_data_element->FirstChildElement("dyn2_filename");
  dyn2_filename_element->QueryStringAttribute("value",&dyn2_filename);
  *m_data->dyn2_filename = dyn2_filename;
  if (flag_verbose >= 2) {
    printf("model_data: dyn2_filename = %s \n",m_data->dyn2_filename->c_str());
  }
  //printf("WARNING: Using hard-coded filename currently \n");
  // load the .pt PyTorch model 
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    *m_data->Dyn2_model = torch::jit::load(m_data->dyn2_filename->c_str());
  } catch (const c10::Error& e) {
    std::cerr << "ERROR: error loading the torch model \n";
    exit(-1);
    //return -1; 
  }


  XMLElement* mask_fix_element 
     = model_data_element->FirstChildElement("mask_fix");
  if (mask_fix_element != NULL) {
    mask_fix_element->QueryStringAttribute("value",&mask_fix_str);
    *m_data->mask_fix_str = mask_fix_str;
    if (flag_verbose >= 2) {
      printf("model_data: mask_fix_str = %s \n",m_data->mask_fix_str->c_str());
    } 
  } else {
    std::cerr << "ERROR: missing mask_fix in MLMOD parameter file. \n";
    exit(-1);  
  }

  parse_mask_fix_str(m_data->mask_fix_str->c_str(),&m_data->mask_fix);
  //this->fixMLMOD->mask = m_data->mask_fix; // warning: double check if need further triggers in lammps
  //setmask(); // double-check this sets correct fields and triggers in lammps

  XMLElement* mask_input_str_element 
     = model_data_element->FirstChildElement("mask_input");
  mask_input_str_element->QueryStringAttribute("value",&mask_input_str);
  *m_data->mask_input_str = mask_input_str;
  if (flag_verbose >= 2) {
    printf("model_data: mask_input = %s \n",m_data->mask_input_str->c_str());
  }
  // setup mask based on the input string "X V F Type"
  parse_mask_input_str(m_data->mask_input_str->c_str(),&m_data->mask_input,
                       &m_data->mask_list_n, &m_data->mask_list);

}

void DriverMLMOD::parse_mask_fix_str(const char *mask_str,int *mask_ptr) {

  // parse string to determine mask
  *mask_ptr = 0;
  char* cp_str = strdup(mask_str); // make a copy
  char *token = strtok(cp_str," "); // tokenize 
  while (token != NULL) {

    if (strcmp(token,"INITIAL_INTEGRATE") == 0) {
      (*mask_ptr) |= INITIAL_INTEGRATE;
    }

    if (strcmp(token,"FINAL_INTEGRATE") == 0) {
      (*mask_ptr) |= FINAL_INTEGRATE;
    }

    if (strcmp(token,"POST_FORCE") == 0) {
      (*mask_ptr) |= POST_FORCE;
    }

    token = strtok(NULL," ");
  }

  free(cp_str);

}

void DriverMLMOD::parse_mask_input_str(const char *mask_input_str,int *mask_input_ptr,
                                       int *mask_list_n_ptr,int **mask_list_ptr) {


  *mask_list_n_ptr = IN_num_types;
  (*mask_list_ptr) = new int[*mask_list_n_ptr];

  int *mask_list = *mask_list_ptr;

  // setup mask filters
  mask_list[0] = 1;
  for (int i = 1; i < *mask_list_n_ptr; i++) {
    mask_list[i] = 2*mask_list[i -1];
  }

  // parse string to determine mask
  *mask_input_ptr = 0;
  char* cp_str = strdup(mask_input_str); // make a copy
  char *token = strtok(cp_str," "); // tokenize 
  while (token != NULL) {

    if (strcmp(token,IN_X_str.c_str()) == 0) {
      (*mask_input_ptr) |= mask_list[IN_X];
    }

    if (strcmp(token,IN_V_str.c_str()) == 0) {
      (*mask_input_ptr) |= mask_list[IN_V];
    }

    if (strcmp(token,IN_F_str.c_str()) == 0) {
      (*mask_input_ptr) |= mask_list[IN_F];
    }

    if (strcmp(token,IN_Type_str.c_str()) == 0) {
      (*mask_input_ptr) |= mask_list[IN_Type];
    }

    if (strcmp(token,IN_Time_str.c_str()) == 0) {
      (*mask_input_ptr) |= mask_list[IN_Time];
    }

    token = strtok(NULL," ");
  }

  free(cp_str);

}

void DriverMLMOD::print_tensor_2d(const char *name, at::Tensor t) {

  auto t_a = t.accessor<float,2>(); 
  int n1 = t.size(0); 
  int n2 = t.size(1);

  cout << name << " = [";	
  for (int i = 0; i < n1; i++) {

    cout << "[";	  
    for (int j = 0; j < n2; j++) {
      cout << t_a[i][j];
      if (j < n2 - 1) {
        cout << ",";
      }   
    }
    cout << "]"; 
    if (i < n1 - 1) {
      cout << "," << endl;
    }

  }
  cout << "]" << endl;

}

void DriverMLMOD::write_array_txt_1D(const char *filename,
                                     int n,double *v) {

  ofstream fid(filename);
  // fid.precision(6)
  if (fid.is_open()) {
    for (int i = 0; i < n; i++) {
      fid << v[i] << endl;
    }
    fid.close();
  } else { 
    cout << "WARNING: Error creating filename = " << filename << endl;
  }

}

void DriverMLMOD::write_array_txt_2D(const char *filename,
                                     int m,int n,double **A) {

  ofstream fid(filename);
  // fid.precision(6)
  if (fid.is_open()) {
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        fid << A[i][j];
        if (j < n - 1) {
          fid << " ";
        } else {
          fid << endl;
        }
      }
    }
    fid.close();
  } else {
    cout << "WARNING: Error creating filename = " << filename << endl;
  }

}


void DriverMLMOD::test1() {

  // run simple test of Torch
  cout << "-----------------------------------" << endl;
  cout << "Constructor case (fixMLMOD,LAMMPS,narg,arg)" << endl;
  cout << "MLMOD: Testing Torch routines." << endl;
  torch::Tensor tensor = torch::rand({2, 3});
  cout << "Tensor = " << tensor << endl;               

}

void DriverMLMOD::test2() { 

  // column-major format for matrices for Lapack (fortran)
  double A[] = {1,2,3,  
		2,5,6,
		3,6,7}; 

  int nn = 3; char uplo = 'L'; int info = 0; int lda = nn;
  // run simple test of lapack
  cout << "-----------------------------------" << endl;
  cout << "Constructor case (fixMLMOD,LAMMPS,narg,arg)" << endl;
  cout << "MLMOD: Testing Lapack routine." << endl;

  // compute the cholesky factor (done in-place, no return, only upper triangular part)
  //dpotrf_(&uplo,&nn,A,&lda,&info);
  //LAPACKE_dpotrf2(matrix_layout,&uplo,&nn,A,&lda);
  info = LAPACKE_dpotrf2(LAPACK_COL_MAJOR,uplo,nn,A,lda);

  // test if any erros occured 
  assert(info == 0);

  // only the lower triangular part is used, 
  // so zero out the rest (note, column-major format) 
  for (int i = 0; i < nn; i++) {
    for (int j = i + 1; j < nn; j++) {
      A[j*nn + i] = 0.0;
    }
  }

  // show the matrix factor L of A = L*L^T
  cout << "L = [";
  for (int i = 0; i < nn; i++) {
    for (int j = 0; j < nn; j++) {    
      cout << A[j*nn + i];
      if (j < nn - 1){ 
	cout << " ";
      } 
      if ((i < nn - 1) && (j == nn - 1)) {
	cout << "\n     ";
      }
      if ((i == nn - 1) && (j == nn - 1)) {
	cout << "]\n";
      }
    }
  }

  cout << "\n";

}

