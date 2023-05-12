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
  "USER-MLMOD Package: \n"
  "@article{mlmod_atzberger,\n"
  "  author  = {Atzberger, P. J.},\n"
  "  journal = {arxiv},\n"
  "  title   = {MLMOD Package: Machine Learning Methods for " 
  "Data-Driven Modeling in LAMMPS},\n"
  "  year    = {2021},\n"
  "  note    = {http://atzberger.org},\n"
  "  doi     = {10.48550/arXiv.2107.14362},\n"
  "  url     = {https://arxiv.org/abs/2107.14362},\n"
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
  int flag = 0;

  /* ================= init ================= */
  /* add to citation collection */
  if (lmp->citeme) lmp->citeme->add(cite_mlmod_str);

  this->fixMLMOD = fixMLMOD;
  this->lammps = lmp;

  this->mlmod_seed = 1;
  this->random = new RanMars(lammps,this->mlmod_seed);  // WARNING: Be careful for MPI, since each seed the same (use comm->me) 

  // init constants
  MODEL_TYPE_NULL = 0;
  MODEL_TYPE_STR_NULL = "NULL";

  MODEL_TYPE_dX_MF = 1;
  MODEL_TYPE_STR_dX_MF = "dX_MF";

  MODEL_TYPE_dX_MF_ML1 = 2;
  MODEL_TYPE_STR_dX_MF_ML1 = "dX_MF_ML1";

  MODEL_TYPE_dX_MF_Q1_ML1_Pair = 3;
  MODEL_TYPE_STR_dX_MF_Q1_ML1_Pair = "dX_MF_Q1_ML1_Pair";

  MODEL_TYPE_dX_MF_Q1_ML1_N2N = 4;
  MODEL_TYPE_STR_dX_MF_Q1_ML1_N2N = "dX_MF_Q1_ML1_N2N";
  
  MODEL_TYPE_F_ML1 = 5;
  MODEL_TYPE_STR_F_ML1 = "F_ML1";
  
  MODEL_TYPE_QoI_ML1 = 6;
  MODEL_TYPE_STR_QoI_ML1 = "QoI_ML1";
  
  MODEL_TYPE_Dyn_ML1 = 7;
  MODEL_TYPE_STR_Dyn_ML1 = "Dyn_ML1";

  this->model_type = MODEL_TYPE_NULL; 
  this->model_type_str = MODEL_TYPE_STR_NULL;

  flag = 0;  
  if (flag == 1) {    
    // run simple test of Torch
    cout << "-----------------------------------" << endl;
    cout << "Constructor case (fixMLMOD,LAMMPS,narg,arg)" << endl;
    cout << "MLMOD: Testing Torch routines." << endl;
    torch::Tensor tensor = torch::rand({2, 3});
    cout << "Tensor = " << tensor << endl;               
  }

  flag = 0;  
  if (flag == 1) {    

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

    // test is any erros occured 
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

  // parse the XML fiele
  parse_xml_params(arg[3]);

  /* == Write some data to XML info file */
  writeInfo();

  cout << "-----------------------------------" << endl;

}

/* destructor */
DriverMLMOD::~DriverMLMOD() {

  if (model_type == MODEL_TYPE_dX_MF_Q1_ML1_Pair) {

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

}

void DriverMLMOD::setup(int vflag) { /* lammps setup */
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
    printf("MLMOD: setmask(), mask = %d\n",mask);
  } else if (model_type == MODEL_TYPE_dX_MF_ML1) {
    mask = FINAL_INTEGRATE;
    printf("MLMOD: setmask(), mask = %d\n",mask); 
  } else if (model_type == MODEL_TYPE_dX_MF_Q1_ML1_Pair) {
    mask = INITIAL_INTEGRATE | FINAL_INTEGRATE;
    printf("MLMOD: setmask(), mask = %d\n",mask); 
  } else if (model_type == MODEL_TYPE_dX_MF_Q1_ML1_N2N) {
    mask = INITIAL_INTEGRATE | FINAL_INTEGRATE;
    printf("MLMOD: setmask(), mask = %d\n",mask); 
  } else if (model_type == MODEL_TYPE_F_ML1) {
    //mask = INITIAL_INTEGRATE | FINAL_INTEGRATE;
    // @@@ make sure already initialized
    ModelData_F_ML1_Type *m_data 
      = (ModelData_F_ML1_Type *) this->model_data;
    mask = m_data->mask_fix;
    printf("MLMOD: setmask(), mask = %d\n",mask); 
  } else if (model_type == MODEL_TYPE_QoI_ML1) {
    //mask = INITIAL_INTEGRATE | FINAL_INTEGRATE;
    // @@@ make sure already initialized
    ModelData_QoI_ML1_Type *m_data 
      = (ModelData_QoI_ML1_Type *) this->model_data;
    mask = m_data->mask_fix;
    printf("MLMOD: setmask(), mask = %d\n",mask); 
  } else if (model_type == MODEL_TYPE_Dyn_ML1) {
    //mask = INITIAL_INTEGRATE | FINAL_INTEGRATE;
    // @@@ make sure already initialized
    ModelData_Dyn_ML1_Type *m_data 
      = (ModelData_Dyn_ML1_Type *) this->model_data;
    mask = m_data->mask_fix;
    printf("MLMOD: setmask(), mask = %d\n",mask); 
  } else {
    // default case 
    mask = FINAL_INTEGRATE;
    printf("MLMOD: setmask(), default, mask = %d\n",mask); 
  }
 
  return mask;

}

/* ---------------------------------------------------------------------- */
void DriverMLMOD::pre_exchange()
{

}

/* ---------------------------------------------------------------------- */
void DriverMLMOD::end_of_step()
{

}

/* ---------------------------------------------------------------------- */

void DriverMLMOD::init()
{

}


void DriverMLMOD::initial_integrate(int vflag)
{

  if (this->model_type == MODEL_TYPE_dX_MF_Q1_ML1_Pair) { 
    initial_integrate_dX_MF_Q1_ML1_Pair();
  } else if (this->model_type == MODEL_TYPE_dX_MF_Q1_ML1_N2N) { 
    initial_integrate_dX_MF_Q1_ML1_N2N();
  } else if (this->model_type == MODEL_TYPE_F_ML1) { 
    initial_integrate_F_ML1();
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
    final_integrate_F_ML1();
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
  int *mask = atom->mask;
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

      if ((mask[i] & groupbit) && (mask[j] & groupbit)) {

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
    if (mask[i] & groupbit) { 
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
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;
  int num_dim = 3;

  double dt = update->dt;

  int groupbit = this->fixMLMOD->groupbit;

  double M_ii[3][3];
  double M_ij[3][3];

  // create tensors
  at::Tensor input1_m = torch::zeros({1,3});
  auto input1_m_a = input1_m.accessor<float,2>(); 
  std::vector<torch::jit::IValue> input1_m_i{input1_m};
  //auto input1_m_i = std::vector<torch::jit::IValue>{torch::jit::IValue::from(input1_m)};
  // std::vector<torch::jit::IValue>{torch::jit::IValue::from(my_array.data())};

  at::Tensor input2_m = torch::zeros({1,6});
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
    if (mask[i] & groupbit) {
      for (int d = 0; d < num_dim; d++) {
        v[i][d] = 0.0;   
      }
    }
  }

  // loop over the points and apply the tensor
  
  // compute the mobility contributions
  for (int i = 0; i < nlocal; i++) {
    for (int j = i; j < nlocal; j++) {

      if ((mask[i] & groupbit) && (mask[j] & groupbit)) {

        if (i == j) { // self-mobility response

          // set the xi tensor 
          for (int d = 0; d < num_dim; d++) {
            input1_m_a[0][d] = x[i][d];
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
              M_ii[d][k] = output_m_a[0][I1]; 
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
              M_ij[d][k] = output_m_a[0][I1]; 
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
    if (mask[i] & groupbit) { 
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
  int *mask = atom->mask;
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
    // (warning we assume stays the same in simulations)
    int count = 0;
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        count++;
      }
    }
    m_data->num_II = count; // set number of points controlled by this fix

    if (m_data->II_size == 0) { // alloc array to collect indices
      m_data->II = (int *)malloc(sizeof(int)*m_data->num_II);
      m_data->II_size = m_data->num_II;
    }

    // collect the indices
    count = 0;
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        m_data->II[count] = i;
        count++;
      }
    }

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
  at::Tensor input1_m = torch::zeros({1,num_dim});
  auto input1_m_a = input1_m.accessor<float,2>(); 
  std::vector<torch::jit::IValue> input1_m_i{input1_m};

  at::Tensor input2_m = torch::zeros({1,2*num_dim});
  auto input2_m_a = input2_m.accessor<float,2>(); 
  std::vector<torch::jit::IValue> input2_m_i{input2_m};

  at::Tensor output_m; // leave un-initialized

  // initialize the velocity to zero
  for (int i = 0; i < m_data->num_II; i++) {
    int iii = m_data->II[i];
    for (int d = 0; d < num_dim; d++) {
      v[iii][d] = 0.0;   
    }
  }

  // construct the mobility tensor M (@optimize later using neigh list thresholds)
  for (int i = 0; i < m_data->num_II; i++) {
    int iii = m_data->II[i];
    for (int j = 0; j < m_data->num_II; j++) {
      int jjj = m_data->II[j];

      if (i == j) { // self-mobility response

        // set the xi tensor 
        for (int d = 0; d < num_dim; d++) {
          input1_m_a[0][d] = x[iii][d];
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
            m_data->M[i*num_dim + d][j*num_dim + k] = output_m_a[0][I1]; 
          }
        }

      } else { // i != j

        // set the xi and xj tensors 
        for (int d = 0; d < num_dim; d++) {
          input2_m_a[0][d] = x[iii][d];
          input2_m_a[0][num_dim + d] = x[jjj][d];
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
            m_data->M[i*num_dim + d][j*num_dim + k] = output_m_a[0][I1]; 
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
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;
  int num_dim = 3;

  double dt = update->dt;

  int groupbit = this->fixMLMOD->groupbit;

  // create tensors
  at::Tensor input1_m = torch::zeros({1,num_dim});
  auto input1_m_a = input1_m.accessor<float,2>(); 
  std::vector<torch::jit::IValue> input1_m_i{input1_m};

  at::Tensor input2_m = torch::zeros({1,2*num_dim});
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
          input1_m_a[0][d] = x[iii][d];
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
            m_data->M[i*num_dim + d][j*num_dim + k] = output_m_a[0][I1]; 
          }
        }

      } else { // i != j

        // set the xi and xj tensors 
        for (int d = 0; d < num_dim; d++) {
          input2_m_a[0][d] = x[iii][d];
          input2_m_a[0][num_dim + d] = x[jjj][d];
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
            m_data->M[i*num_dim + d][j*num_dim + k] = output_m_a[0][I1]; 
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
  int *mask = atom->mask;
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
    // (warning we assume stays the same in simulations)
    int count = 0;
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        count++;
      }
    }
    m_data->num_II = count; // set number of points controlled by this fix

    if (m_data->II_size == 0) { // alloc array to collect indices
      m_data->II = (int *)malloc(sizeof(int)*m_data->num_II);
      m_data->II_size = m_data->num_II;
    }

    // collect the indices
    count = 0;
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        m_data->II[count] = i;
        count++;
      }
    }

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
  at::Tensor input2_m = torch::zeros({1,m_data->num_II*num_dim});
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
      input2_m_a[0][i*num_dim + d] = x[iii][d];
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
          m_data->M[i*num_dim + d][j*num_dim + k] = output_m_a[0][I1]; 
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
  int *mask = atom->mask;
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
  at::Tensor input_m = torch::zeros({1,m_data->num_II*num_dim});
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
      input_m_a[0][i*num_dim + d] = x[iii][d];
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
          m_data->M[i*num_dim + d][j*num_dim + k] = output_m_a[0][I1]; 
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

    if (mask[i] & groupbit) {
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
/* ---------------------------------------------------------------------- */
void DriverMLMOD::initial_integrate_F_ML1()
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
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;
  int num_dim = 3;

  double dt = update->dt;

  int groupbit = this->fixMLMOD->groupbit;

  // use .pt model from PyTorch for ML learned model for M(X)
  // get model data
  ModelData_F_ML1_Type *m_data 
    = (ModelData_F_ML1_Type *) this->model_data;

  // initialize
  if (m_data->flag_init == 0) {

    // determine index list controlled by this fix 
    // (warning we assume stays the same in simulations)
    int count = 0;
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        count++;
      }
    }
    m_data->num_II = count; // set number of points controlled by this fix

    if (m_data->II_size == 0) { // alloc array to collect indices
      m_data->II = (int *)malloc(sizeof(int)*m_data->num_II);
      m_data->II_size = m_data->num_II;
    }

    // collect the indices
    count = 0;
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        m_data->II[count] = i;
        count++;
      }
    }

   // collect data for size of inputs
   count = 0;
   if (m_data->mask_input & m_data->IN_X) { 
     count += m_data->num_II*num_dim;
   }

   if (m_data->mask_input & m_data->IN_V) { 
     count += m_data->num_II*num_dim;
   }

   if (m_data->mask_input & m_data->IN_F) { 
     count += m_data->num_II*num_dim;
   }

   if (m_data->mask_input & m_data->IN_Type) {
     count += m_data->num_II;
   }

   m_data->input_size = count; 

   m_data->flag_init = 1;

  } // end flag_init

  // create tensors
  at::Tensor input2_m = torch::zeros({1,m_data->input_size});
  //auto input2_m_a = input2_m.accessor<float,2>(); 
  at::TensorAccessor<float,2,at::DefaultPtrTraits,long> input2_m_a = input2_m.accessor<float,2>(); 
  std::vector<torch::jit::IValue> input2_m_i{input2_m};

  at::Tensor output_m; // leave un-initialized

  // construct the input for force QoI() 
  constr_input_masked(input2_m_a,m_data->num_II,m_data->II,m_data->mask_input,
                      m_data->mask_list_n,m_data->mask_list,
                      x,v,f,type);

  // evaluate the model F(X) and obtain tensor output
  output_m = m_data->F_model->forward(input2_m_i).toTensor();
  // accessor for the 2D tensor 
  // 1 x m_data->num_II*num_dim*m_data->num_II*num_dim
  auto output_m_a = output_m.accessor<float,2>(); 

  // reshape into the needed force F(X) -> f[i][d] tensor components
  for (int i = 0; i < m_data->num_II; i++) {
    int iii = m_data->II[i];
    for (int d = 0; d < num_dim; d++) {
      int I1 = i*num_dim + d; 
      f[iii][d] += output_m_a[0][I1]; 
    }
  } // i

}

/* ---------------------------------------------------------------------- */
void DriverMLMOD::final_integrate_F_ML1()
{
  // @@@ WARNING!!! POST_FORCE might be more appropriate than the final (integration...)

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
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;
  int num_dim = 3;

  double dt = update->dt;

  int groupbit = this->fixMLMOD->groupbit;

  // use .pt model from PyTorch for ML learned model for M(X)
  // get model data
  ModelData_F_ML1_Type *m_data 
    = (ModelData_F_ML1_Type *) this->model_data;

  // create tensors
  at::Tensor input_m = torch::zeros({1,m_data->input_size});
  //auto input_m_a = input_m.accessor<float,2>(); 
  at::TensorAccessor<float,2,at::DefaultPtrTraits,long> input_m_a = input_m.accessor<float,2>(); 
  std::vector<torch::jit::IValue> input_m_i{input_m};

  at::Tensor output_m; // leave un-initialized

  // construct the input for force F() 
  constr_input_masked(input_m_a,m_data->num_II,m_data->II,m_data->mask_input,
                      m_data->mask_list_n,m_data->mask_list,
                      x,v,f,type);

  // evaluate the model F(X) and obtain tensor output
  output_m = m_data->F_model->forward(input_m_i).toTensor();
  // accessor for the 2D tensor 
  // 1 x m_data->num_II*num_dim*m_data->num_II*num_dim
  auto output_m_a = output_m.accessor<float,2>(); 

  // reshape into the needed f[iii][d] tensor components
  for (int i = 0; i < m_data->num_II; i++) {
    int iii = m_data->II[i];
    for (int d = 0; d < num_dim; d++) {
      int I1 = i*num_dim + d; 
      f[iii][d] = output_m_a[0][I1]; 
    }
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
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;
  int num_dim = 3;

  double dt = update->dt;

  int groupbit = this->fixMLMOD->groupbit;

  // use .pt model from PyTorch for ML learned model for M(X)
  // get model data
  ModelData_QoI_ML1_Type *m_data 
    = (ModelData_QoI_ML1_Type *) this->model_data;

  // initialize
  if (m_data->flag_init == 0) {

    // determine index list controlled by this fix 
    // (warning we assume stays the same in simulations)
    int count = 0;
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        count++;
      }
    }
    m_data->num_II = count; // set number of points controlled by this fix

    if (m_data->II_size == 0) { // alloc array to collect indices
      m_data->II = (int *)malloc(sizeof(int)*m_data->num_II);
      m_data->II_size = m_data->num_II;
    }

    // collect the indices
    count = 0;
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        m_data->II[count] = i;
        count++;
      }
    }

   // collect data for size of inputs
   count = 0;
   if (m_data->mask_input & m_data->IN_X) { 
     count += m_data->num_II*num_dim;
   }

   if (m_data->mask_input & m_data->IN_V) { 
     count += m_data->num_II*num_dim;
   }

   if (m_data->mask_input & m_data->IN_F) { 
     count += m_data->num_II*num_dim;
   }

   if (m_data->mask_input & m_data->IN_Type) {
     count += m_data->num_II;
   }

   m_data->input_size = count; 

   // set flag initialized
   m_data->flag_init = 1;

  } // end flag_init

  if (lammps->update->ntimestep % m_data->skip_step == 0) { 

    // create tensors (@optimize should do this just once)
    at::Tensor input2_m = torch::zeros({1,m_data->input_size});
    // auto input2_m_a = input2_m.accessor<float,2>(); 
    at::TensorAccessor<float,2,at::DefaultPtrTraits,long> input2_m_a = input2_m.accessor<float,2>(); 
    std::vector<torch::jit::IValue> input2_m_i{input2_m};

    at::Tensor output_m; // leave un-initialized

    // construct the input for force QoI() 
    constr_input_masked(input2_m_a,m_data->num_II,m_data->II,m_data->mask_input,
                        m_data->mask_list_n,m_data->mask_list,
                        x,v,f,type);

    // evaluate the model F(X) and obtain tensor output
    output_m = m_data->QoI_model->forward(input2_m_i).toTensor();
    // accessor for the 2D tensor 
    // 1 x m_data->num_II*num_dim*m_data->num_II*num_dim
    auto output_m_a = output_m.accessor<float,2>();

    // write the output 
    const char *base_name = m_data->base_name->c_str();
    const char* base_dir = m_data->base_dir->c_str();
    char *filename = new char[strlen(base_name) + strlen(base_dir) + 17];
    sprintf(filename,"%s/%s_0_%.8ld.xml",
            base_dir,base_name,lammps->update->ntimestep);
    ofstream fid(filename);
    // fid.precision(6)  
    if (fid.is_open()) {
      fid << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
      fid << "<base_name value=\"" << base_name << "\"/>\n";
      fid << "<data>\n";
      int NN = output_m.size(1);
      for (int i = 0; i < NN; i++) {
        int I1 = i; 
        fid << output_m_a[0][I1] << " "; 
      }
      fid << "</data>\n";
      fid.close();
    }

    delete[] filename;

    // stores data as local vector (might need to set other flags too to access) 
    //int local_flag;                // 0/1 if local data is stored
    //int size_local_rows;           // rows in local vector or array
    //int size_local_cols;           // 0 = vector, N = columns in local array
    //int local_freq;                // frequency local data is available at

    // @@@ output the QoI to the specified file on skips
    
    // @@@ or store QoI as "compute array associated with the fix"

    // reshape into the needed force QoI(X) -> tensor components
    /*
    for (int i = 0; i < m_data->num_II; i++) {
      int iii = m_data->II[i];
      for (int d = 0; d < num_dim; d++) {
        int I1 = i*num_dim + d; 
        f[iii][d] += output_m_a[0][I1]; 
      }
    } // i
    */
  }

}

/* ---------------------------------------------------------------------- */
void DriverMLMOD::final_integrate_QoI_ML1()
{
  // @@@ WARNING!!! POST_FORCE might be more appropriate than the final (integration...)

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
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;
  int num_dim = 3;

  double dt = update->dt;

  int groupbit = this->fixMLMOD->groupbit;

  // use .pt model from PyTorch for ML learned model for M(X)
  // get model data
  ModelData_QoI_ML1_Type *m_data 
    = (ModelData_QoI_ML1_Type *) this->model_data;

  if (lammps->update->ntimestep % m_data->skip_step == 0) { 

    // create tensors (@optimize should do this just once)
    at::Tensor input_m = torch::zeros({1,m_data->input_size});
    //auto input_m_a = input_m.accessor<float,2>(); 
    at::TensorAccessor<float,2,at::DefaultPtrTraits,long> input_m_a = input_m.accessor<float,2>(); 
    std::vector<torch::jit::IValue> input_m_i{input_m};

    at::Tensor output_m; // leave un-initialized

    // construct the input for force QoI() 
    constr_input_masked(input_m_a,m_data->num_II,m_data->II,m_data->mask_input,
                        m_data->mask_list_n,m_data->mask_list,
                        x,v,f,type);

    // evaluate the model QoI() and obtain tensor output
    output_m = m_data->QoI_model->forward(input_m_i).toTensor();
    // accessor for the 2D tensor 
    // 1 x m_data->num_II*num_dim*m_data->num_II*num_dim
    auto output_m_a = output_m.accessor<float,2>(); 

    // reshape into the needed f[iii][d] tensor components
    /*
    for (int i = 0; i < m_data->num_II; i++) {
      int iii = m_data->II[i];
      for (int d = 0; d < num_dim; d++) {
        int I1 = i*num_dim + d; 
        f[iii][d] = output_m_a[0][I1]; 
      }
    }
    */

    // write the output
    const char *base_name = m_data->base_name->c_str();
    const char* base_dir = m_data->base_dir->c_str();
    char *filename = new char[strlen(base_name) + strlen(base_dir) + 17];
    sprintf(filename,"%s/%s_0_%.8ld.xml",
            base_dir,base_name,lammps->update->ntimestep);
    ofstream fid(filename);
    // fid.precision(6)  
    if (fid.is_open()) {
      fid << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
      fid << "<base_name value=\"" << base_name << "\"/>\n";
      fid << "<data>\n";
      int NN = output_m.size(1);
      for (int i = 0; i < NN; i++) {
        int I1 = i; 
        fid << output_m_a[0][I1] << " ";     
      }
      fid << "\n";
      fid << "</data>\n";
      fid.close();
    }

    delete[] filename;

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
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;
  int num_dim = 3;

  double dt = update->dt;

  int groupbit = this->fixMLMOD->groupbit;

  // use .pt model from PyTorch for ML learned model for M(X)
  // get model data
  ModelData_Dyn_ML1_Type *m_data 
    = (ModelData_Dyn_ML1_Type *) this->model_data;

  // initialize
  if (m_data->flag_init == 0) {

    // determine index list controlled by this fix 
    // (warning we assume stays the same in simulations)
    int count = 0;
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        count++;
      }
    }
    m_data->num_II = count; // set number of points controlled by this fix

    if (m_data->II_size == 0) { // alloc array to collect indices
      m_data->II = (int *)malloc(sizeof(int)*m_data->num_II);
      m_data->II_size = m_data->num_II;
    }

    // collect the indices
    count = 0;
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        m_data->II[count] = i;
        count++;
      }
    }
   
   // collect data for size of inputs
   count = 0;
   if (m_data->mask_input & m_data->IN_X) { 
     count += m_data->num_II*num_dim;
   }

   if (m_data->mask_input & m_data->IN_V) { 
     count += m_data->num_II*num_dim;
   }

   if (m_data->mask_input & m_data->IN_F) { 
     count += m_data->num_II*num_dim;
   }

   if (m_data->mask_input & m_data->IN_Type) {
     count += m_data->num_II;
   }

   m_data->input_size = count; 

   m_data->flag_init = 1;

  } // end flag_init

  // create tensors
  at::Tensor input2_m = torch::zeros({1,m_data->input_size});
  //auto input2_m_a = input2_m.accessor<float,2>(); 
  at::TensorAccessor<float,2,at::DefaultPtrTraits,long> input2_m_a = input2_m.accessor<float,2>(); 
  std::vector<torch::jit::IValue> input2_m_i{input2_m};

  at::Tensor output_m; // leave un-initialized

  // construct the input for update (X,V) = Gamma(X,V,...) 
  // (@optimize later using neigh list thresholds)
  constr_input_masked(input2_m_a,m_data->num_II,m_data->II,m_data->mask_input,
                      m_data->mask_list_n,m_data->mask_list,
                      x,v,f,type);
  
  // evaluate the model (X,V) = Gamma(X,V) and obtain tensor output
  output_m = m_data->Dyn1_model->forward(input2_m_i).toTensor();
  // accessor for the 2D tensor 
  // 1 x m_data->num_II*num_dim*m_data->num_II*num_dim
  auto output_m_a = output_m.accessor<float,2>(); 

  // reshape into the needed x[iii][d],v[iii][d] tensor components
  int I0 = 0; int II0 = 0;
  for (int i = 0; i < m_data->num_II; i++) {
    int iii = m_data->II[i];
    for (int d = 0; d < num_dim; d++) {
      int I1 = I0 + i*num_dim + d;
      x[iii][d] = output_m_a[0][I1];
      II0 += 1; 
    }
  }
  I0 = II0;
  for (int i = 0; i < m_data->num_II; i++) {
    int iii = m_data->II[i];
    for (int d = 0; d < num_dim; d++) {
      int I1 = I0 + i*num_dim + d;
      v[iii][d] = output_m_a[0][I1];      
      II0 += 1; 
    }
  }
  
}

/* ---------------------------------------------------------------------- */
void DriverMLMOD::final_integrate_Dyn_ML1()
{
  // @@@ WARNING!!! POST_FORCE might be more appropriate than the final (integration...)

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
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;
  int num_dim = 3;

  double dt = update->dt;

  int groupbit = this->fixMLMOD->groupbit;

  // use .pt model from PyTorch for ML learned model for 
  // update (X,V) = Gamma(X,V) get model data
  ModelData_Dyn_ML1_Type *m_data 
    = (ModelData_Dyn_ML1_Type *) this->model_data;

  // create tensors
  at::Tensor input_m = torch::zeros({1,m_data->input_size});
  //auto input_m_a = input_m.accessor<float,2>(); 
  at::TensorAccessor<float,2,at::DefaultPtrTraits,long> input_m_a = input_m.accessor<float,2>(); 
  std::vector<torch::jit::IValue> input_m_i{input_m};

  at::Tensor output_m; // leave un-initialized

  // construct the input for update (X,V) = Gamma(X,V,...) 
  constr_input_masked(input_m_a,m_data->num_II,m_data->II,m_data->mask_input,
                      m_data->mask_list_n,m_data->mask_list,
                      x,v,f,type);

  // evaluate the model (X,V) = Gamma(X,V) and obtain tensor output
  output_m = m_data->Dyn2_model->forward(input_m_i).toTensor();
  // accessor for the 2D tensor 
  // 1 x m_data->num_II*num_dim*m_data->num_II*num_dim
  auto output_m_a = output_m.accessor<float,2>(); 

  // reshape into the needed x[iii][d],v[iii][d] tensor components
  int I0 = 0; int II0 = 0;
  for (int i = 0; i < m_data->num_II; i++) {
    int iii = m_data->II[i];
    for (int d = 0; d < num_dim; d++) {
      int I1 = I0 + i*num_dim + d;
      x[iii][d] = output_m_a[0][I1];
      II0 += 1; 
    }
  }
  I0 = II0;
  for (int i = 0; i < m_data->num_II; i++) {
    int iii = m_data->II[i];
    for (int d = 0; d < num_dim; d++) {
      int I1 = I0 + i*num_dim + d;
      v[iii][d] = output_m_a[0][I1];      
      II0 += 1; 
    }
  }
  
}

void DriverMLMOD::reset_dt()
{

}

void DriverMLMOD::post_force(int vflag) {

}


/*****************************************************************************************/
/* Supporting codes */
/*****************************************************************************************/
void DriverMLMOD::init_attributes() {

}

void DriverMLMOD::writeInfo() {
}

// pass along the initialization
void DriverMLMOD::init_from_fix() {

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

void DriverMLMOD::constr_input_masked(at::TensorAccessor<float,2,at::DefaultPtrTraits,long> input_m_a,
                                      int num_indices,int *indices,
                                      int mask_input, int mask_list_n, int *mask_list,
                                      double **x,double **v,double **f, int *type) {

  // (@optimize later using neigh list thresholds)
  int I0 = 0; int II0 = 0; int num_dim = 3;
  if (mask_input & mask_list[0]) { 
    I0 = II0;
    for (int i = 0; i < num_indices; i++) {
      int iii = indices[i];
      for (int d = 0; d < num_dim; d++) {
        input_m_a[0][I0 + i*num_dim + d] = x[iii][d];
        II0 += 1;
      }
    }
  }

  if (mask_input & mask_list[1]) { 
    I0 = II0;
    for (int i = 0; i < num_indices; i++) {
      int iii = indices[i];
      for (int d = 0; d < num_dim; d++) {
        input_m_a[0][I0 + i*num_dim + d] = v[iii][d];
        II0 += 1;
      }
    }
  }

  if (mask_input & mask_list[2]) { 
    I0 = II0;
    for (int i = 0; i < num_indices; i++) {
      int iii = indices[i];
      for (int d = 0; d < num_dim; d++) {
        input_m_a[0][I0 + i*num_dim + d] = f[iii][d];
        II0 += 1;
      }
    }
  }

  if (mask_input & mask_list[3]) {
    I0 = II0;
    for (int i = 0; i < num_indices; i++) {
      int iii = indices[i];
      input_m_a[0][I0 + i] = (double)type[iii];  // warning cast, not ideal
      II0 += 1;
    }
  }

}

// parser
void DriverMLMOD::parse_xml_params(char *filename) {

  XMLDocument doc;
  doc.LoadFile(filename);

  printf("parsing filename = %s\n",filename);

  const char *model_type_str;
  XMLElement* model_data_element 
    = doc.FirstChildElement("MLMOD")->FirstChildElement("model_data");
  model_data_element->QueryStringAttribute("type",&model_type_str);
  printf("model_data: type = %s \n",model_type_str);

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

  this->model_type = MODEL_TYPE_dX_MF;

  ModelData_dX_MF_Type *m_data 
    = (ModelData_dX_MF_Type *)malloc(sizeof(ModelData_dX_MF_Type));
  this->model_data = (void *) m_data;


  XMLElement* eta_element = model_data_element->FirstChildElement("eta");
  eta_element->QueryDoubleAttribute("value",&m_data->eta);
  printf("model_data: eta = %g \n",m_data->eta);

  XMLElement* a_element = model_data_element->FirstChildElement("a");
  a_element->QueryDoubleAttribute("value",&m_data->a);
  printf("model_data: a = %g \n",m_data->a);

  XMLElement* epsilon_element = model_data_element->FirstChildElement("epsilon");
  epsilon_element->QueryDoubleAttribute("value",&m_data->epsilon);
  printf("model_data: epsilon = %g \n",m_data->epsilon);

}


// parser case specific data
void DriverMLMOD::parse_xml_params_dX_MF_ML1(XMLElement *model_data_element) {
  
  const char *M_ii_filename;
  const char *M_ij_filename;
  
  this->model_type = MODEL_TYPE_dX_MF_ML1;

  ModelData_dX_MF_ML1_Type *m_data 
    = (ModelData_dX_MF_ML1_Type *)malloc(sizeof(ModelData_dX_MF_ML1_Type));
  m_data->M_ii_filename = new string();
  m_data->M_ij_filename = new string();
  m_data->M_ii_model = new torch::jit::script::Module();
  m_data->M_ij_model = new torch::jit::script::Module();

  this->model_data = (void *) m_data;

  XMLElement* M_ii_filename_element 
     = model_data_element->FirstChildElement("M_ii_filename");
  M_ii_filename_element->QueryStringAttribute("value",&M_ii_filename);
  *m_data->M_ii_filename = M_ii_filename;
  printf("model_data: M_ii_filename = %s \n",m_data->M_ii_filename->c_str());

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
  printf("model_data: M_ij_filename = %s \n",m_data->M_ij_filename->c_str());

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

}

// parser case specific data
void DriverMLMOD::parse_xml_params_dX_MF_Q1_ML1_Pair(XMLElement *model_data_element) {
  
  const char *M_ii_filename;
  const char *M_ij_filename;
  
  this->model_type = MODEL_TYPE_dX_MF_Q1_ML1_Pair;

  ModelData_dX_MF_Q1_ML1_Pair_Type *m_data 
    = (ModelData_dX_MF_Q1_ML1_Pair_Type *)malloc(sizeof(ModelData_dX_MF_Q1_ML1_Pair_Type));
  m_data->M_ii_filename = new string();
  m_data->M_ij_filename = new string();
  m_data->M_ii_model = new torch::jit::script::Module();
  m_data->M_ij_model = new torch::jit::script::Module();

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
  printf("model_data: M_ii_filename = %s \n",m_data->M_ii_filename->c_str());

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
  printf("model_data: M_ij_filename = %s \n",m_data->M_ij_filename->c_str());

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
  printf("model_data: delta = %g \n",m_data->delta);

  XMLElement* KBT_element = model_data_element->FirstChildElement("KBT");
  KBT_element->QueryDoubleAttribute("value",&m_data->KBT);
  printf("model_data: KBT = %g \n",m_data->KBT);

  XMLElement* flag_stochastic_element 
    = model_data_element->FirstChildElement("flag_stochastic");
  flag_stochastic_element->QueryIntAttribute("value",&m_data->flag_stochastic);
  printf("model_data: flag_stochastic = %d \n",m_data->flag_stochastic);

  if (m_data->KBT == 0) {
    cout << "WARNING: KBT = 0, so setting flag_stochastic = 0.\n";
    m_data->flag_stochastic = 0; 
  }

  XMLElement* flag_thm_drift_element 
    = model_data_element->FirstChildElement("flag_thm_drift");
  flag_thm_drift_element->QueryIntAttribute("value",&m_data->flag_thm_drift);
  printf("model_data: flag_thm_drift = %d \n",m_data->flag_thm_drift);

}

// parser case specific data
void DriverMLMOD::parse_xml_params_dX_MF_Q1_ML1_N2N(XMLElement *model_data_element) {
  
  const char *M_filename;
  
  this->model_type = MODEL_TYPE_dX_MF_Q1_ML1_N2N;

  ModelData_dX_MF_Q1_ML1_N2N_Type *m_data 
    = (ModelData_dX_MF_Q1_ML1_N2N_Type *)malloc(sizeof(ModelData_dX_MF_Q1_ML1_N2N_Type));

  m_data->M_filename = new string();
  m_data->M_model = new torch::jit::script::Module();

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
  printf("model_data: M_filename = %s \n",m_data->M_filename->c_str());

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
  printf("model_data: delta = %g \n",m_data->delta);

  XMLElement* KBT_element = model_data_element->FirstChildElement("KBT");
  KBT_element->QueryDoubleAttribute("value",&m_data->KBT);
  printf("model_data: KBT = %g \n",m_data->KBT);

  XMLElement* flag_stochastic_element 
    = model_data_element->FirstChildElement("flag_stochastic");
  flag_stochastic_element->QueryIntAttribute("value",&m_data->flag_stochastic);
  printf("model_data: flag_stochastic = %d \n",m_data->flag_stochastic);

  if (m_data->KBT == 0) {
    cout << "WARNING: KBT = 0, so setting flag_stochastic = 0.\n";
    m_data->flag_stochastic = 0; 
  }

  XMLElement* flag_thm_drift_element 
    = model_data_element->FirstChildElement("flag_thm_drift");
  flag_thm_drift_element->QueryIntAttribute("value",&m_data->flag_thm_drift);
  printf("model_data: flag_thm_drift = %d \n",m_data->flag_thm_drift);

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

  m_data->flag_init = 0;
  m_data->num_II = 0;
  m_data->II_size = 0;
  m_data->II = NULL;

  m_data->base_name = new string();
  m_data->base_dir = new string();

  m_data->F_filename = new string();
  m_data->F_model = new torch::jit::script::Module();

  m_data->mask_fix_str = new string();
  m_data->mask_input_str = new string();

  this->model_data = (void *) m_data;

  XMLElement* base_name_element 
     = model_data_element->FirstChildElement("base_name");
  base_name_element->QueryStringAttribute("value",&base_name);
  *m_data->base_name = base_name;
  printf("model_data: base_name = %s \n",m_data->base_name->c_str());

  XMLElement* base_dir_element 
     = model_data_element->FirstChildElement("base_dir");
  base_dir_element->QueryStringAttribute("value",&base_dir);
  *m_data->base_dir = base_dir;
  printf("model_data: base_dir = %s \n",m_data->base_dir->c_str());

  // create a new directory for output for mlmod
  int rv = mkdir(m_data->base_dir->c_str(),0755);
    
  if ((rv == -1) && (errno != EEXIST)) {  
    cout << "Failed making directory path = " << m_data->base_dir->c_str() << endl;
  }

  XMLElement* F_filename_element 
     = model_data_element->FirstChildElement("F_filename");
  F_filename_element->QueryStringAttribute("value",&F_filename);
  *m_data->F_filename = F_filename;
  printf("model_data: F_filename = %s \n",m_data->F_filename->c_str());

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
    printf("model_data: mask_fix_str = %s \n",m_data->mask_fix_str->c_str());
  } else {
    std::cerr << "ERROR: missing mask_fix in MLMOD parameter file. \n";
    exit(-1);  
  }

  parse_mask_fix_str(m_data->mask_fix_str->c_str(),&m_data->mask_fix);

  XMLElement* mask_input_str_element 
     = model_data_element->FirstChildElement("mask_input");
  mask_input_str_element->QueryStringAttribute("value",&mask_input_str);
  *m_data->mask_input_str = mask_input_str;
  printf("model_data: mask_input = %s \n",m_data->mask_input_str->c_str());

  // setup mask based on the input string "X V F Type"
  parse_mask_input_str(m_data->mask_input_str->c_str(),&m_data->mask_input,
                       &m_data->mask_list_n, &m_data->mask_list);
  m_data->IN_X = m_data->mask_list[0];m_data->IN_V = m_data->mask_list[1];
  m_data->IN_F = m_data->mask_list[2];m_data->IN_Type = m_data->mask_list[3];
  
}

// parser case specific data
void DriverMLMOD::parse_xml_params_QoI_ML1(XMLElement *model_data_element) {
  
  const char *base_name;
  const char *base_dir;

  const char *mask_input_str;
  const char *mask_fix_str;
  
  const char *QoI_filename;

  this->model_type = MODEL_TYPE_QoI_ML1;

  ModelData_QoI_ML1_Type *m_data 
    = (ModelData_QoI_ML1_Type *)malloc(sizeof(ModelData_QoI_ML1_Type));

  m_data->flag_init = 0;
  m_data->num_II = 0;
  m_data->II_size = 0;
  m_data->II = NULL;

  m_data->base_name = new string();
  m_data->base_dir = new string();

  m_data->QoI_filename = new string();
  m_data->QoI_model = new torch::jit::script::Module();

  m_data->mask_fix_str = new string();
  m_data->mask_input_str = new string();

  this->model_data = (void *) m_data;

  XMLElement* base_name_element 
     = model_data_element->FirstChildElement("base_name");
  base_name_element->QueryStringAttribute("value",&base_name);
  *m_data->base_name = base_name;
  printf("model_data: base_name = %s \n",m_data->base_name->c_str());

  XMLElement* base_dir_element 
     = model_data_element->FirstChildElement("base_dir");
  if (base_dir_element != NULL) {
    base_dir_element->QueryStringAttribute("value",&base_dir);
    *m_data->base_dir = base_dir;
    printf("model_data: base_dir = %s \n",m_data->base_dir->c_str());
  } else {
    *m_data->base_dir = "./mlmod"; // default dir
  }

  // create a new directory for output for mlmod
  int rv = mkdir(m_data->base_dir->c_str(),0755);
    
  if ((rv == -1) && (errno != EEXIST)) {  
    cout << "Failed making directory path = " << m_data->base_dir->c_str() << endl;
  }

  XMLElement* QoI_filename_element 
     = model_data_element->FirstChildElement("QoI_filename");
  QoI_filename_element->QueryStringAttribute("value",&QoI_filename);
  *m_data->QoI_filename = QoI_filename;
  printf("model_data: QoI_filename = %s \n",m_data->QoI_filename->c_str());

  //printf("WARNING: Using hard-coded filename currently \n");
  // load the .pt PyTorch model 
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    *m_data->QoI_model = torch::jit::load(m_data->QoI_filename->c_str());
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
    printf("model_data: mask_fix_str = %s \n",m_data->mask_fix_str->c_str());
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
  printf("model_data: mask_input = %s \n",m_data->mask_input_str->c_str());

  // setup mask based on the input string "X V F Type"
  parse_mask_input_str(m_data->mask_input_str->c_str(),&m_data->mask_input,
                       &m_data->mask_list_n, &m_data->mask_list);
  m_data->IN_X = m_data->mask_list[0];m_data->IN_V = m_data->mask_list[1];
  m_data->IN_F = m_data->mask_list[2];m_data->IN_Type = m_data->mask_list[3];
  
  XMLElement* skip_step_element 
     = model_data_element->FirstChildElement("skip_step");
  if (skip_step_element != NULL) {
    skip_step_element->QueryIntAttribute("value",&m_data->skip_step);
  } else {
    m_data->skip_step = 1;
  }
  printf("model_data: skip_step = %d \n",m_data->skip_step);

}

// parser case specific data
void DriverMLMOD::parse_xml_params_Dyn_ML1(XMLElement *model_data_element) {
  
  const char *base_name;
  const char *base_dir;

  const char *mask_input_str;
  const char *mask_fix_str;
  
  const char *Dyn1_filename;
  const char *Dyn2_filename;
  
  this->model_type = MODEL_TYPE_Dyn_ML1;

  ModelData_Dyn_ML1_Type *m_data 
    = (ModelData_Dyn_ML1_Type *)malloc(sizeof(ModelData_Dyn_ML1_Type));

  m_data->flag_init = 0;
  m_data->num_II = 0;
  m_data->II_size = 0;
  m_data->II = NULL;

  m_data->base_name = new string();
  m_data->base_dir = new string();

  m_data->mask_fix_str = new string();
  m_data->mask_input_str = new string();

  m_data->Dyn1_filename = new string();
  m_data->Dyn2_filename = new string();
  m_data->Dyn1_model = new torch::jit::script::Module();
  m_data->Dyn2_model = new torch::jit::script::Module();

  this->model_data = (void *) m_data;

  XMLElement* base_name_element 
     = model_data_element->FirstChildElement("base_name");
  base_name_element->QueryStringAttribute("value",&base_name);
  *m_data->base_name = base_name;
  printf("model_data: base_name = %s \n",m_data->base_name->c_str());

  XMLElement* base_dir_element 
     = model_data_element->FirstChildElement("base_dir");
  if (base_dir_element != NULL) {
    base_dir_element->QueryStringAttribute("value",&base_dir);
    *m_data->base_dir = base_dir;
    printf("model_data: base_dir = %s \n",m_data->base_dir->c_str());
  } else {
    *m_data->base_dir = "./mlmod"; // default dir
  }

  // create a new directory for output for mlmod
  int rv = mkdir(m_data->base_dir->c_str(),0755);
    
  if ((rv == -1) && (errno != EEXIST)) {  
    cout << "Failed making directory path = " << m_data->base_dir->c_str() << endl;
  }


  XMLElement* Dyn1_filename_element 
     = model_data_element->FirstChildElement("dyn1_filename");
  Dyn1_filename_element->QueryStringAttribute("value",&Dyn1_filename);
  *m_data->Dyn1_filename = Dyn1_filename;
  printf("model_data: Dyn1_filename = %s \n",m_data->Dyn1_filename->c_str());

  //printf("WARNING: Using hard-coded filename currently \n");
  // load the .pt PyTorch model 
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    *m_data->Dyn1_model = torch::jit::load(m_data->Dyn1_filename->c_str());
  } catch (const c10::Error& e) {
    std::cerr << "ERROR: error loading the torch model \n";
    exit(-1);
    //return -1; 
  }

  XMLElement* Dyn2_filename_element 
     = model_data_element->FirstChildElement("dyn2_filename");
  Dyn2_filename_element->QueryStringAttribute("value",&Dyn2_filename);
  *m_data->Dyn2_filename = Dyn2_filename;
  printf("model_data: Dyn2_filename = %s \n",m_data->Dyn2_filename->c_str());

  //printf("WARNING: Using hard-coded filename currently \n");
  // load the .pt PyTorch model 
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    *m_data->Dyn2_model = torch::jit::load(m_data->Dyn2_filename->c_str());
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
    printf("model_data: mask_fix_str = %s \n",m_data->mask_fix_str->c_str());
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
  printf("model_data: mask_input = %s \n",m_data->mask_input_str->c_str());

  // setup mask based on the input string "X V F Type"
  parse_mask_input_str(m_data->mask_input_str->c_str(),&m_data->mask_input,
                       &m_data->mask_list_n, &m_data->mask_list);
  m_data->IN_X = m_data->mask_list[0];m_data->IN_V = m_data->mask_list[1];
  m_data->IN_F = m_data->mask_list[2];m_data->IN_Type = m_data->mask_list[3];
  

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

    token = strtok(NULL," ");
  }

  free(cp_str);

}

void DriverMLMOD::parse_mask_input_str(const char *mask_input_str,int *mask_input_ptr,
                                       int *mask_list_n_ptr,int **mask_list_ptr) {


  *mask_list_n_ptr = 4;
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

    if (strcmp(token,"X") == 0) {
      (*mask_input_ptr) |= mask_list[0];
    }

    if (strcmp(token,"V") == 0) {
      (*mask_input_ptr) |= mask_list[1];
    }

    if (strcmp(token,"F") == 0) {
      (*mask_input_ptr) |= mask_list[2];
    }

    if (strcmp(token,"Type") == 0) {
      (*mask_input_ptr) |= mask_list[3];
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


