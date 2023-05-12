**Quick Start**

The preferred approach is to use the Python interface to the package
and the pre-compiled binaries.  These are available using "pip install"
from the ./pip directory.  As an alternative docker images are also 
available.  For details, latest releases, and examples see the 
current mlmod github repository and http://atzberger.org/ 

**Compiling from Source Code**

The mlmod package is implemented in C/C++ and has been organized as a library
that is interfaced with LAMMPS.  To compile from the source codes requires
downloading or installing a few dependent packages.  To ensure compatibility,
we also list the versions that were used during development and testing.  These
are:

LAMMPS mesoscale and molecular dynamics software \
https://lammps.org/ \
(version, Mar 28, 2023)

tinyxml2 package for parsing XML \
https://github.com/leethomason/tinyxml2 \
(version 9.0.0)

libtorch (PyTorch C++ API) \
https://pytorch.org/cppdocs/ \
(version 2.0.0)

liblapacke (Numerical linear algebra) \
https://netlib.org/lapack/ \
(version 3.10.0)

libfftw3 (FFT package) \
https://www.fftw.org/ \
(version 3.3.8)

For LAMMPS you may also need the lastest versions of libpng and libjpeg and other 
packages (depending on the options you choose for compilation), see LAMMPS
documentation for details.

Scripts and Makefiles have been put together to help with compilation within
LAMMPS.  This may require further reading of the LAMMPS compilation steps and
editing the makefiles to adjust the source and library installation locations.

Compilation of mlmod for LAMMPS involves the following steps:

* Download LAMMPS from https://lammps.org/ 

* Copy the mlmod ./lib directory to the LAMMPS ./lib directory.  Copy
the mlmod ./src/USER-MLMOD directory to the LAMMPS ./src directory.
Copy the mlmod ./src/USER-MLMOD/MAKE to the LAMMPS ./src/MAKE/MINE directory.

* Build the mlmod shared library libmlmod.so by going to the LAMMPS ./lib/mlmod
directory and running ./build.sh.  You may need to adjust the paths 
in the Makefiles for the dependencies for the locations of the 
installed libraries and include files.

* Build the mpi stubs by going into LAMMPS ./src/STUBS and running ``make``. 
 
* Build MLMOD-LAMMPS by going to the LAMMPS ./src directory and using 
```
make yes-user-mlmod yes-molecule 
make mlmod_serial
```
* You may need to edit the Makfile in ./src/MAKE/Makefile.mlmod_serial 
to adjust to the paths of the locations for the installed packages.

* The LAMMPS binary is given by lmp_mlmod_serial.  This can be used to run LAMMPS 
scripts, "lmp_mlmod_serial -in script.lammps."  As a quick test, 
run "./lmp_mlmod_serial -h".  You may need to adjust the LD_LIBRARY_PATH 
to include the location of the shared libraries for use at runtime. 

* See the examples folder and mlmod website for further details on 
setting up models and running simulations.

**Additional Information**

For more details, see the LAMMPS documentation pages https://lammps.org/ 
and the MLMOD examples and documentation pages.

For more information see 
http://atzberger.org/

