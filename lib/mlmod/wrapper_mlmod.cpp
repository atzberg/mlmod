/* -----------------------------------------------------------------------------

  MLMOD: Machine Learning (ML) Modeling (MOD) and Simulation Package 

  Paul J. Atzberger
  http://atzberger.org/
  
  Please cite the follow paper when referencing this package
  
  "MLMOD Package: Machine Learning Methods for Data-Driven Modeling in LAMMPS",
  Atzberger, P. J., aarXiv.2107.14362, 2021.
  
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

/* MLMOD_includes */
#include "wrapper/wrapper_mlmod.h"
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
WrapperMLMOD::WrapperMLMOD() { driver_mlmod = new DriverMLMOD(); }

/* ------------------------------------------------------------------------- */
WrapperMLMOD::WrapperMLMOD(LAMMPS_NS::FixMLMOD *fixMLMOD, LAMMPS *lmp, int narg, char **arg) { driver_mlmod = new DriverMLMOD(fixMLMOD,lmp,narg,arg); }

/* ------------------------------------------------------------------------- */
WrapperMLMOD::~WrapperMLMOD() { delete driver_mlmod; }

void WrapperMLMOD::setup(int vflag) { driver_mlmod->setup(vflag); }

/* ---------------------------------------------------------------------- */
int WrapperMLMOD::setmask(){ return driver_mlmod->setmask(); }

/* ---------------------------------------------------------------------- */
void WrapperMLMOD::pre_exchange() { driver_mlmod->pre_exchange(); }

/* ---------------------------------------------------------------------- */
void WrapperMLMOD::end_of_step() { driver_mlmod->end_of_step(); }

/* ---------------------------------------------------------------------- */
void WrapperMLMOD::init() { driver_mlmod->init_from_fix(); } // @@@ double-check

void WrapperMLMOD::initial_integrate(int vflag) { driver_mlmod->initial_integrate(vflag); }

/* ---------------------------------------------------------------------- */
void WrapperMLMOD::final_integrate() { driver_mlmod->final_integrate(); }

/* ---------------------------------------------------------------------- */
void WrapperMLMOD::reset_dt() { driver_mlmod->reset_dt(); }

/* ---------------------------------------------------------------------- */
void WrapperMLMOD::post_force(int vflag) { driver_mlmod->post_force(vflag); 
}
/* ---------------------------------------------------------------------- */
void WrapperMLMOD::init_from_fix() { driver_mlmod->init_from_fix(); }



