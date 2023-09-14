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

#include <cstdlib>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <sstream>

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

/* MLMOD_includes */
#include "fix_mlmod.h"
#include "include/mlmod.h"

//using namespace MLMOD;
using namespace LAMMPS_NS;
using namespace FixConst;
using namespace std;

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
FixMLMOD::FixMLMOD() : Fix(NULL, 0, NULL) {
  /* WARNING: May need to modify LAMMPS codes so that we have
              Fix(NULL, 0, NULL) acts like empty constructor */

  //mlmod = new MLMOD();
   
}

/* =========================== Class definitions =========================== */
FixMLMOD::FixMLMOD(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg)
{

  const char *error_str_func = "FixMLMOD()";
  
  time_integrate = 1; /* set to 1 for fix performing integration, 0 if fix does not */
  
  /* setup the wrapper for MLMOD library */
  mlmod = new MLMOD(this,lmp,narg,arg);
  
}

/* destructor */
FixMLMOD::~FixMLMOD() {

  /* write some final information to disk */
  //writeFinalInfo();

}

void FixMLMOD::setup(int vflag) { /* lammps setup */

  mlmod->setup(vflag);
  
}

/* ---------------------------------------------------------------------- */
int FixMLMOD::setmask()
{
  /*
    mask |= INITIAL_INTEGRATE;
    mask |= FINAL_INTEGRATE;
   */

  // pass back value
  //MLMOD_integrator_mask = mlmod->setmask();

  //return MLMOD_integrator_mask; /* syncronize the MLMOD mask with that returned to LAMMPS */

  return mlmod->setmask();
}

/* ---------------------------------------------------------------------- */
void FixMLMOD::pre_exchange()
{

  /* pass to integrator to handle */
  mlmod->pre_exchange();

}

/* ---------------------------------------------------------------------- */
void FixMLMOD::end_of_step()
{

  /* pass to integrator to handle */
  mlmod->end_of_step();

}

/* ---------------------------------------------------------------------- */

void FixMLMOD::init()
{

  const char *error_str_func = "init()";

  /* == Initialize the MLMOD integrators. */
  /* update->integrate->step; (time step from LAMMPS) */

  /* == Check the integration style is Verlet, if not report error. */
  if (strcmp(update->integrate_style, "verlet") != 0) {
    stringstream message;
    message << "MLMOD requires for now use of the verlet integrate_style." <<  endl;
    //MLMOD_Package::packageError(error_str_code, error_str_func, message);
    //cout << message;
    printf("%s",message.str().c_str());
  }

  /* == Initialize data structures for MLMOD. */
  /* integrator trigger fix_init() for any associated initialization */
  mlmod->init_from_fix();
}

/* ----------------------------------------------------------------------
   allow for both per-type and per-atom mass
------------------------------------------------------------------------- */
//void FixMLMOD::integrate_initialize() {

//  MLMOD_IntegratorData->integrate_initialize();

//}


void FixMLMOD::initial_integrate(int vflag)
{

  mlmod->initial_integrate(vflag);

}

/* ---------------------------------------------------------------------- */

void FixMLMOD::final_integrate()
{

  mlmod->final_integrate();

}

/* ---------------------------------------------------------------------- */

double FixMLMOD::compute_array(int i, int j)
{

  return mlmod->compute_array(i,j);

}

/* ---------------------------------------------------------------------- */

void FixMLMOD::reset_dt()
{
  mlmod->reset_dt();
}

void FixMLMOD::post_force(int vflag) {
  mlmod->post_force(vflag);
}




