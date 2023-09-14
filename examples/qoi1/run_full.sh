#!/bin/bash -x 

# run full pipe-line, including generating and updating the ml models. 
model_dir=./mlmod_model1

# generate the model again
./gen_and_setup_qoi1.sh

# copy models into the template folder
./cp_qoi1.sh

# run 
python run_sim_qoi1.py

