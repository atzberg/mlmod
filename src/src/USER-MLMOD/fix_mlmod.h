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
  
  /** @brief Lammps interface to mlmod library using the lammps "fix"
      mechanism.
      
      @param LAMMPS*: reference to the running lammps instance
      @param narg: number of command arguments
      @param args: list of strings for the arguments 
  **/
  FixMLMOD(class LAMMPS *, int narg, char **args);

  /** @brief Lammps interface to mlmod library using the lammps "fix"
      mechanism.  Empty constructor used mostly for testing.
  **/
  FixMLMOD(); /* used to construct empty object,
                primarily for testing purposes */

  ~FixMLMOD();

  int          setmask();
  virtual void init();
  void         setup(int vflag);

  /** @brief Lammps calls at the start of each timestep. **/
  virtual void initial_integrate(int);

  /** @brief Lammps calls at the end of each timestep. **/
  virtual void final_integrate();

  /** @brief Lammps calls when timestep size changes. **/
  void         reset_dt();

  /** @brief Lammps calls after forces are calculated. **/
  void         post_force(int vflag);
  
  // additional methods/hooks
  void pre_exchange();
  void end_of_step();

  /* =========================== Function Calls =========================== */

  /* =========================== Variables =========================== */

  /** @brief Reference to the lammps instance. **/
  LAMMPS                                             *lammps; /* lammps data */

  /** @brief Reference to the mlmod library wrapper. **/
  WrapperMLMOD                                       *wrapper_mlmod;

};

}

#endif
#endif
