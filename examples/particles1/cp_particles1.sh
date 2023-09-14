#!/bin/bash -x

dir1=./output/gen_mlmod_oseen1/gen_001
dir2=./output/gen_mlmod_rpy1/gen_001
dir3=./mlmod_model1

mkdir -p $dir3
cp $dir1/M*.* $dir3
cp $dir2/M*.* $dir3

