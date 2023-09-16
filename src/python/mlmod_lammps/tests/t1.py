from mlmod_lammps.lammps import IPyLammps;

def test():
  print("Create Lammps session:");
  L = IPyLammps();
  print("Show Lammps information:");
  L = L.command("info all");

