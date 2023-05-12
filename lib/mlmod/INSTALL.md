The MLMOD library depends on the following libraries

tinyxml2: XML parser library \
torch: PyTorch library \
lapacke: Numerical linear algebra library \

You may need to adjust the paths for the locations of these libraries. 

The package builds into a library which is linked with LAMMPS with the USER-MLMOD enabled 
(see LAMMPS document for details on building and user package modules:
https://lammps.org/). 

An example build is given in the script
./build.sh

For more details, see the installation instructions in 
the mlmod ./src directory. 
