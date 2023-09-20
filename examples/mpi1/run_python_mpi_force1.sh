#!/bin/bash

# Paul J. Atzberger

# may need to adjust path of python within simple_mpi.py
# also, may need to manually install the lammps_mpi.so and lmp
# with LD_LIBRARY_PATH adjusted in $pkg_dir/lammps 
# the $pkg_dir obtained from python -m site.

out_file_base=./output/mpi_force1/mpirun_python_output
rm -rf $out_file_base

nproc=4
mpirun -n $nproc --oversubscribe --output-filename $out_file_base python mpi_force1.py in.force1

