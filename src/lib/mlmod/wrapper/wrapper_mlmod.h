/**
 * @file 
 *
 * @brief MLMOD routines driving the machine learning models and simulations.
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

/** @brief Forward declaration of the mlmod driver class. **/
class DriverMLMOD;

}

namespace LAMMPS_NS {

/** @brief Forward declaration of the mlmod lammps class. **/
class FixMLMOD;

/** @brief Forward declaration of the lammps class. **/
class LAMMPS;

/** @brief Prototype for the interface wrapping the mlmod package. **/
struct WrapperMLMOD {

 public:
    
  /** @brief Main constructor for mlmod package wrapper. 
  
      @param FixMLMOD*: reference to the mlmod-lammps "fix" interface  
      @param LAMMPS*: reference to the running lammps instance
      @param narg: number of command arguments
      @param args: list of strings for the arguments 
  **/
  WrapperMLMOD(class FixMLMOD *, class LAMMPS *, int narg, char **args);

  /** @brief Constructor for empy object which is used primarily for testing.
   **/
  WrapperMLMOD();
  ~WrapperMLMOD();

  int          setmask();
  virtual void init();
  void         setup(int vflag);

  /** @brief Called by lammps at the start of each timestep. **/
  virtual void initial_integrate(int);

  /** @brief Called by lammps at the end of each timestep. **/
  virtual void final_integrate();

  /** @brief Called by lammps when timestep size changes. **/
  void         reset_dt();

  /** @brief Called by lammps after forces are calculated. **/
  void         post_force(int vflag);
  
  // additional methods/hooks
  void pre_exchange();
  void end_of_step();
  void init_from_fix();

  /* =========================== Variables =========================== */

  /** @brief Reference to lammps instance. **/
  LAMMPS *lmp; /* lammps data */

  /** @brief Reference to mlmod driver instance. **/
  USER_MLMOD::DriverMLMOD *driver_mlmod;

};

}

#endif


