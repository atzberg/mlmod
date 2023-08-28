#!/bin/bash 

if [[ "$1" = "-l" ]]
then 
  lammps_dir=$2
else
  echo " " 
  echo "Requires specifying the base lammps directory."
  echo "This should be the parent directory to ./src, ./lib."
  echo "Use the flag:"
  echo "  -l (lammps-dir)"
  exit 1
fi

echo " "
echo "For a fresh build, attempts to clean out all previous files related to mlmod "
echo "from the lammps distribution located at:"
echo lammps_dir=$lammps_dir

# remove mlmod package settings in lammps
cwd=$PWD
cd $lammps_dir/src
make no-user-mlmod
cd $cwd

# remove files 
rm -rf $lammps_dir/src/USER-MLMOD
rm -rf $lammps_dir/lib/mlmod
rm $lammps_dir/src/MAKE/MINE/Makefile*.mlmod*
rm $lammps_dir/lib/vtk/Makefile.*mlmod*
rm $lammps_dir/src/build*mlmod*.sh
rm $lammps_dir/src/*mlmod*.*

echo "You may still need to manually remove files for a fresh build. "
echo "" 
echo "See the documentation pages for more information."
 
