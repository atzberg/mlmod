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
echo "Copying files for mlmod into the lammps distribution located at:"
echo lammps_dir=$lammps_dir

cp -r ./src/USER-MLMOD $lammps_dir/src
cp -r ./src/USER-MLMOD/MAKE/* $lammps_dir/src/MAKE/MINE
cp -r ./lib/mlmod $lammps_dir/lib
cp ./lib/vtk/* $lammps_dir/lib/vtk
cp ./src/build_mlmod_lammps_serial.sh $lammps_dir/src

echo ""
echo "To try to quick build, you can use the script in directory \$lammps_dir/src:"
echo "build_mlmod_lammps_serial.sh "
echo ""
echo "You may still need to install dependent packages or adjust paths/settings in "
echo "the build files for your platform."
echo "" 
echo "See the documentation pages for more information."
 
