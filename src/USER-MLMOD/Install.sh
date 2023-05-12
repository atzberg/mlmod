# Install/unInstall package files in LAMMPS

if (test $1 = 1) then

#  echo "  "
#  echo "WARNING: (PJA) We use symbolic links instead of copy of source codes."
#  echo "         This was done to help avoid issues when debugging and changing files. "
#  echo "         If one is not careful the copies are changed and edits not retained."
#  echo "         To avoid this annoying issue, symbolic links are used."
#  echo "         For the final release a copy using cp -p command should be used instead."
#

  echo "================================================================================"
  echo "USER-MLMOD Package: Machine Learning (ML) for Data-Driven Modeling (MOD)        "
  echo "--------------------------------------------------------------------------------"
  #echo " "

  #echo "Symbolic link copying files for USER-MLMOD package."  
  #basePath=$(pwd -P)
  #echo "Base Path = $basePath"
  #ln -sf $basePath/*.h ../
  #ln -sf $basePath/*.cpp ../

  echo "Copying files for the package into the source directory."
  #cp -p $PWD/*.h ../
  #cp -p $PWD/*.cpp ../
  find $PWD -name '*.h' -exec cp "{}" ../ >& mlmod_cp_h.log \;
  find $PWD -name '*.cpp' -exec cp "{}" ../ >& mlmod_cp_cpp.log \;

  echo " "
  echo "Note USER-MLMOD current version uses serial head node."
  echo " "
  echo "For more information and examples see "
  echo "http://atzberger.org"
  echo " "
  echo "================================================================================"

elif (test $1 = 0) then

  rm ../fix_mlmod.cpp
  rm ../fix_mlmod.h

  echo "  "
#  echo "WARNING: (PJA) List of files to remove is not yet implemented."

fi


# test is MOLECULE package installed already
if (test $1 = 1) then
  if (test ! -e ../angle_harmonic.cpp) then
    echo "Must install MOLECULE package with USER-MLMOD, for example by command 'make yes-molecule'."
    exit 1
  fi

fi

# setup library settings
if (test $1 = 1) then

  if (test -e ../Makefile.package) then
    sed -i -e 's/[^ \t]*mlmod[^ \t]* //' ../Makefile.package
    sed -i -e 's|^PKG_INC =[ \t]*|&-I../../lib/mlmod |' ../Makefile.package
    sed -i -e 's|^PKG_PATH =[ \t]*|&-L../../lib/mlmod |' ../Makefile.package

#   sed -i -e 's|^PKG_INC =[ \t]*|&-I../../lib/mlmod |' ../Makefile.package
#   sed -i -e 's|^PKG_PATH =[ \t]*|&-L../../lib/mlmod$(LIBSOBJDIR) |' ../Makefile.package
    sed -i -e 's|^PKG_LIB =[ \t]*|&-lmlmod |' ../Makefile.package
    sed -i -e 's|^PKG_SYSINC =[ \t]*|&$(user-mlmod_SYSINC) |' ../Makefile.package
    sed -i -e 's|^PKG_SYSLIB =[ \t]*|&$(user-mlmod_SYSLIB) |' ../Makefile.package
    sed -i -e 's|^PKG_SYSPATH =[ \t]*|&$(user-mlmod_SYSPATH) |' ../Makefile.package
  fi

  if (test -e ../Makefile.package.settings) then
    sed -i -e '/^include.*mlmod.*$/d' ../Makefile.package.settings
    # multiline form needed for BSD sed on Macs
    sed -i -e '4 i \
include ..\/..\/lib\/mlmod\/Makefile.lammps
' ../Makefile.package.settings
  fi

elif (test $1 = 0) then

  if (test -e ../Makefile.package) then
    sed -i -e 's/[^ \t]*mlmod[^ \t]* //' ../Makefile.package
  fi

  if (test -e ../Makefile.package.settings) then
    sed -i -e '/^include.*mlmod.*$/d' ../Makefile.package.settings
  fi

fi

