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
//struct MLMOD;
//}

using namespace std;

namespace LAMMPS_NS {

//class MLMOD;
class MLMOD;

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

  /** @brief Lammps calls to set the mask for when to call into the fix. **/
  int          setmask();

  /** @brief Lammps calls when initializing the fix. **/
  virtual void init();

  /** @brief Lammps calls to setup the fix. **/
  void         setup(int vflag);

  /** @brief Lammps calls at the start of each timestep. **/
  void initial_integrate(int);

  /** @brief Lammps calls at the end of each timestep. **/
  void final_integrate();

  /** @brief Lammps calls when retrieving array data. **/
  double compute_array(int i, int j);

  /** @brief Lammps calls when timestep size changes. **/
  void         reset_dt();

  /** @brief Lammps calls after forces are calculated. **/
  void         post_force(int vflag);
  
  // additional methods/hooks

  /** @brief Lammps calls before exchanging data between processors.. **/
  void pre_exchange();

  /** @brief Lammps calls at the very end of a time-step.. **/
  void end_of_step();

  /* =========================== Function Calls =========================== */

  /* =========================== Variables =========================== */

  /** @brief Reference to the lammps instance. **/
  LAMMPS *lammps; /* lammps data */

  /** @brief Reference to the mlmod library wrapper. **/
  MLMOD *mlmod;

};

}

#endif
#endif
