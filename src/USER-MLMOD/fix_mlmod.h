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

#ifdef FIX_CLASS

FixStyle(mlmod,FixMLMOD)

#else

#ifndef LMP_FIX_MLMOD_H
#define LMP_FIX_MLMOD_H

#ifndef FFT_FFTW /* determine if FFTW package specified */
  #warning "No FFTW package specified for MLMOD codes. Some methods may be disabled."
#else
  #define USE_PACKAGE_FFTW3
#endif

#include "fix.h"
#include "lammps.h"

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cstddef>

// forward declaration
//namespace MLMOD {
//struct WrapperMLMOD;
//}

using namespace std;

namespace LAMMPS_NS {

//class WrapperMLMOD;
class WrapperMLMOD;

class FixMLMOD : public Fix {

 public:

  const char *error_str_code;

  /* ================= function prototypes ================= */
  FixMLMOD(class LAMMPS *, int, char **);

  FixMLMOD(); /* used to construct empty object,
                primarily for testing purposes */
  ~FixMLMOD();

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

  /* =========================== Function Calls =========================== */

  /* =========================== Variables =========================== */
  LAMMPS                                             *lammps; /* lammps data */
  WrapperMLMOD                                       *wrapper_mlmod;

};

}

#endif
#endif
