#!/bin/bash -x 

# run full pipe-line, including generating and updating the ml models. 

out_dir="$PWD/output/run_sh_python_debug/test_001"
model_dir="$PWD/mlmod_model1"

debug_dir="$PWD/debug"

debug_file=python_debug1.gdb
python_file=run_sim_qoi1.py

# clean-up for new run
rm -rf $out_dir/*
rm -rf $out_dir

# setup dir
mkdir -p $out_dir
mkdir -p $out_dir/vtk

# generate the model again
rm -f $model_dir/*F_*.*
./gen_and_setup_qoi1.sh

# copy model data
cp -r $model_dir $out_dir
cp $debug_dir/$debug_file $out_dir
cp $python_file $out_dir

# copy this script
#cp run.sh $out_dir/archive__run_sh.sh
cp $0 $out_dir/archive__run_script.sh

#echo ""
#echo "Setting up symbolic links"
#echo $PWD

# NOTE: May need to adjust these to point to the compiled lammps binaries
# PJA: maybe change to relative paths, makes more portable

# NOTE: adjust this to give symbolic link pointing 
# to the compiled binary.

#ln -s (path-to-lammps/lmp) lmp_mlmod_lammps

lammps_base=$PWD/../../..

#ln -s $lammps_base/src/lmp_atz_mlmod_serial_debug $out_dir/lmp_mlmod_lammps
#cp lmp $out_dir/lmp

#ln -s $PWD/../../../lmp_serial_debug_atz2 $out_dir/lmp_mlmod_lammps
#ln -s $PWD/lmp_mlmod_lammps $out_dir/lmp_mlmod_lammps

#ln -s $PWD/liblammps.so $out_dir/liblammps.so
#ln -s $PWD/liblammps_serial_debug_atz2.so $out_dir/liblammps_serial_debug_atz2.so

echo ""
echo "Run the simulation:"
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./
LD_LIBRARY_PATH=/home/atzberg/anaconda3/envs/mlmod-lammps/lib/python3.9/site-packages/mlmod_lammps/lammps
LD_LIBRARY_PATH=$lammps_dir/lib/mlmod/libtorch_cxx11/lib

#LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$lammps_base/src
#LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$lammps_base/lib/mlmod/libtorch/lib
#LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$lammps_base/lib/mlmod
#LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$lammps_base/lib/mlmod
#LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$lammps_base/lib/selm

# ensure can be used by any child processes
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH

echo LD_LIBRARY_PATH=$LD_LIBRARY_PATH 

echo ""
echo "Changing to output directory to run simulations."

# run the simulation
# ./lmp_mlmod_lammps -in Model.LAMMPS_script

# -tui: gives a text (terminal) ui interface
#gdb -q -tui python -x $debug_file


cd $out_dir

echo ""
echo "List files:"
ls -lah 

# run 
python run_sim_qoi1.py


