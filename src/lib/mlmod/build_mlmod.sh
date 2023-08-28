#!/bin/bash

# check number of inputs
if [ "$#" -eq 0 ]; then
  echo ""
  echo "Need to specify one of the platforms:"
  ls -lah ./MAKE/Makefile.lib.* | sed -n -e 's/^.* //p' | sed -n -e 's/\.\/MAKE\/Makefile\.lib\.//p' 
  echo ""
  exit 1
fi

echo "Building mlmod library for platform: $1"

make -f ./MAKE/Makefile.lib.$1 clean
make -f ./MAKE/Makefile.lib.$1 lib_shared

