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

#ifndef LMP_MLMOD_H
#define LMP_MLMOD_H

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
struct MLMOD {

 public:
    
  /** @brief Main constructor for mlmod package wrapper. 
  
      @param FixMLMOD*: reference to the mlmod-lammps "fix" interface  
      @param LAMMPS*: reference to the running lammps instance
      @param narg: number of command arguments
      @param args: list of strings for the arguments 
  **/
  MLMOD(class FixMLMOD *, class LAMMPS *, int narg, char **args);

  /** @brief Constructor for empy object which is used primarily for testing.
   **/
  MLMOD();
  ~MLMOD();

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
  
  /** @brief Called by lammps after forces are calculated. **/
  double       compute_array(int i, int j);

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


