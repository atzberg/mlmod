#!/bin/bash

echo "Building mlmod library"

make -f Makefile.serial clean
make -f Makefile.serial lib_shared


