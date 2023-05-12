## Installation of Python Package from Binaries 

To install pre-compiled package for Python, download package

- Linux Debian 9+/Ubuntu: [mlmod_lammps-0.0.4.dev0-py3-none-manylinux_2_24_x86_64.whl](http://web.math.ucsb.edu/~atzberg/mlmod/distr/mlmod_lammps-0.0.4.dev0-py3-none-manylinux_2_24_x86_64.whl) <br>
- Linux Debian 9+/Ubuntu (flexible install): [mlmod_lammps-0.0.4.dev0-py3-none-any.whl](http://web.math.ucsb.edu/~atzberg/mlmod/distr/mlmod_lammps-0.0.4.dev0-py3-none-any.whl)

Install using

```pip install -U (substitute-the-filename-here).whl```

To test the package installed run 

```python -c "from mlmod_lammps.tests import t1; t1.test()"```

See the examples and documentation for information on usage of the package. 

