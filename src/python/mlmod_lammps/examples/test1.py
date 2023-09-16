# quick test of package
# see http://atzberger.org/ for examples and more information

from selm_lammps.lammps import PyLammps

L = PyLammps();

L.command("info all");

