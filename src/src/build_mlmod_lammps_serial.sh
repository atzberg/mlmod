#!/bin/sh -x

#make no-all
make yes-molecule yes-vtk yes-user-mlmod  

# make MPI stubs
cwd=$(pwd)
cd ./STUBS
make clean; make 
cd $cwd

# setup the mlmod make
cp ../lib/mlmod/MAKE/Makefile.lammps.mlmod_serial ../lib/mlmod/Makefile.lammps

# build the mlmod shared library
#cwd=$(pwd)
#cd ../lib/mlmod
#./m_build_mlmod_serial.sh
#cd $cwd

# PJA: Remember to copy ../lib/vtk/Makefile.lammps correctly for given build 
# This distinguishes the 6.3 vs 7.1, etc...
cp ../lib/vtk/Makefile.lammps.mlmod_serial ../lib/vtk/Makefile.lammps

# touch, so the SVN and date information updated
touch fix_mlmod.cpp

# make the library
make mlmod_serial  mode=shared

