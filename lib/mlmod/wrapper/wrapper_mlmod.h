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

#ifndef LMP_WRAPPER_MLMOD_H
#define LMP_WRAPPER_MLMOD_H

//#include "mpi.h"
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cstddef>

using namespace std;

// forward declaration

namespace USER_MLMOD {

class DriverMLMOD;

}

namespace LAMMPS_NS {

class FixMLMOD;
class LAMMPS;

struct WrapperMLMOD {

 public:

  WrapperMLMOD(class FixMLMOD *, class LAMMPS *, int, char **);

  WrapperMLMOD();
  ~WrapperMLMOD();

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
  void init_from_fix();

  /* =========================== Variables =========================== */
  LAMMPS *lmp; /* lammps data */
  USER_MLMOD::DriverMLMOD *driver_mlmod;

};

}

#endif


