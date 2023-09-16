/**
 * @file 
 *
 * @brief MLMOD routines driving the machine learning models and simulations.
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

/* MLMOD_includes */
#include "include/mlmod.h"
#include "driver_mlmod.h"

#include "fix_mlmod.h"
#include "lammps.h"

// PyTorch C/C++ interface
//#include <torch/torch.h>

#include <cstdlib>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <sstream>

using namespace LAMMPS_NS;
using namespace USER_MLMOD;
using namespace FixConst;
using namespace std;

/* ------------------------------------------------------------------------- */
MLMOD::MLMOD() { driver_mlmod = new DriverMLMOD(); }

/* ------------------------------------------------------------------------- */
MLMOD::MLMOD(LAMMPS_NS::FixMLMOD *fixMLMOD, LAMMPS *lmp, int narg, char **arg) { driver_mlmod = new DriverMLMOD(fixMLMOD,lmp,narg,arg); }

/* ------------------------------------------------------------------------- */
MLMOD::~MLMOD() { delete driver_mlmod; }

/* ------------------------------------------------------------------------- */
void MLMOD::setup(int vflag) { driver_mlmod->setup(vflag); }

/* ---------------------------------------------------------------------- */
int MLMOD::setmask(){ return driver_mlmod->setmask(); }

/* ---------------------------------------------------------------------- */
void MLMOD::pre_exchange() { driver_mlmod->pre_exchange(); }

/* ---------------------------------------------------------------------- */
void MLMOD::end_of_step() { driver_mlmod->end_of_step(); }

/* ---------------------------------------------------------------------- */
void MLMOD::init() { driver_mlmod->init_from_fix(); } // @@@ double-check

/* ------------------------------------------------------------------------- */
void MLMOD::initial_integrate(int vflag) { driver_mlmod->initial_integrate(vflag); }

/* ---------------------------------------------------------------------- */
void MLMOD::final_integrate() { driver_mlmod->final_integrate(); }

/* ---------------------------------------------------------------------- */
double MLMOD::compute_array(int i, int j) { return driver_mlmod->compute_array(i,j); }

/* ---------------------------------------------------------------------- */
void MLMOD::reset_dt() { driver_mlmod->reset_dt(); }

/* ---------------------------------------------------------------------- */
void MLMOD::post_force(int vflag) { driver_mlmod->post_force(vflag); 
}

/* ---------------------------------------------------------------------- */
void MLMOD::init_from_fix() { driver_mlmod->init_from_fix(); }

/* ---------------------------------------------------------------------- */


