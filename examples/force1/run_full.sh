#!/bin/bash -x 

# run full pipe-line, including generating and updating the ml models. 

# generate the model and copy to template folder
./gen_and_setup_force1.sh

# run 
python run_sim_force1.py


