<p align="left">
<img src="https://github.com/atzberg/mlmod/blob/main/images/docs/mlmod_software.png" width="90%"> 
</p>

[Documentation](https://web.atzberger.org/mlmod/docs/index.html) |
[Examples](./examples) |
[Paper](./paper/paper.pdf)
                                                                                                
### MLMOD: Machine Learning Methods for Data-Driven Modeling in LAMMPS
Now available with Jupyter notebooks and Python scripts for readily setting up
models and simulations.

**MLMOD** is a Python/C++ package for utilizing machine learning methods and
data-driven modeling for simulations in LAMMPS.  The package provides methods
for time-step integrators for dynamics and interactions using general ML model
classes, including Neural Networks, Kernel Regression, and others.  Models can
be trained and exported from PyTorch or from other machine learning frameworks.
Note, this is an early prototype alpha release with future implementations and
features to come.  Please provide feedback or information on bugs using the
forms below. 

LAMMPS is an optimized molecular dynamics package in C/C++ providing many
interaction potentials and analysis tools for modeling and simulation.
Interaction methods include particle-mesh electrostatics, common coarse-grained
potentials, many-body interactions, and others.

**Quick Start**

__To try automatically__ to install the package for Python, you can use the script
[quick_install.py](./quick_install.py).
```
python ./quick_install.py
```

__To install directly__ the pre-compiled packages for Python, download one of the
following

- Linux Debian 9+/Ubuntu (previous):
  [mlmod_lammps-1.0.3-py3-none-manylinux_2_24_x86_64.whl](https://web.atzberger.org/mlmod/distr/mlmod_lammps-1.0.3-py3-none-manylinux_2_24_x86_64.whl)
- Linux Debian 9+/Ubuntu (latest):
  [mlmod_lammps-1.0.4-py3-none-manylinux_2_34_x86_64.whl](https://web.atzberger.org/mlmod/distr/mlmod_lammps-1.0.4-py3-none-manylinux_2_34_x86_64.whl)  
- Linux Debian 9+/Ubuntu (flexible install) (previous):
  [mlmod_lammps-1.0.3-py3-none-any.whl](https://web.atzberger.org/mlmod/distr/mlmod_lammps-1.0.3-py3-none-any.whl)
- Linux Debian 9+/Ubuntu (flexible install) (latest):
  [mlmod_lammps-1.0.4-py3-none-any.whl](https://web.atzberger.org/mlmod/distr/mlmod_lammps-1.0.4-py3-none-any.whl)

Install using 

```
pip install -U (substitute-filename-or-url-here).whl
```

To test the package installed run 

```
python -c "from mlmod_lammps.tests import t1; t1.test()"
```

__For example models and simulations__, see the notebooks and scripts in the
[examples
folder](/examples).  

__Documentation pages__ can be found
[here](https://web.atzberger.org/mlmod/docs/index.html).

Pre-compiled binaries are currently for (Debian 9+/Ubuntu and Centos 7+, Python 3.6+).

__If you installed previously__ this package, please be sure to update to the
latest version using 
```
pip install -U (substitute-filename-or-url-here).whl
```

__Video__ giving a brief overview of MLMOD can be found
[here](https://youtu.be/BZulqaZT5o0). 


**Other ways to install the package**
For running prototype models and simulations on a desktop, such as Windows and
MacOS, we recommend using Docker container.  For example, install [Docker
Desktop](https://docs.docker.com/desktop/), or docker for linux, and then load
a standard ubuntu container by using in the terminal ```docker run -it
ubuntu:24.04 /bin/bash```  You can then use ```apt update; apt install
python3-pip```, and can then pip install and run the simulation package as
above.  Note use command ```python3``` in place of ```python``` when calling
scripts.  Pre-installed packages in anaconda also in 
```
docker run -it atzberg/ubuntu_24_04_anaconda3 /bin/bash
```  
Use 
```
conda activate mlmod-lammps
``` 

May need to update packages to the latest version.   

For more information on other ways to install or compile the package, please
see the [documentation
pages](https://web.atzberger.org/mlmod/docs/index.html).

---
Please cite the paper below for this package:

**MLMOD Package: Machine Learning Methods for Data-Driven Modeling in
LAMMPS"**, P.J. Atzberger, Journal of Open Source Software, 8(89), 5620, (2023) [[paper
link]](https://doi.org/10.21105/joss.05620).
```
@article{mlmod_atzberger,
  author    = {Paul J. Atzberger},
  journal   = {Journal of Open Source Software}, 
  title     = {MLMOD: Machine Learning Methods for Data-Driven Modeling in LAMMPS},
  year      = {2023},  
  publisher = {The Open Journal},
  volume    = {8},
  number    = {89},
  pages     = {5620},
  note      = {http://atzberger.org},
  doi       = {10.21105/joss.05620},
  url       = {https://doi.org/10.21105/joss.05620}
}
```
----

**Tutorials / Talks:**

- __Video__ giving a brief overview of MLMOD can be found
  [here](https://youtu.be/BZulqaZT5o0).

- __Documentation__ on using the package can be found
  [here](https://web.atzberger.org/mlmod/docs/index.html).

- __Examples__ can be found
  [here](https://github.com/atzberg/mlmod/tree/master/examples).

__Mailing List for Future Updates and Releases__

- Please join the mailing list for future updates and releases
  [here](https://forms.gle/7seLKxrVrAN9U8Bg7).

__Bugs or Issues__

- If you encounter any bugs or issues please let us know by providing information
  [here](https://forms.gle/xBYLs7Gi1dwYvCR5A).

__Please submit usage and citation information__

- If you use this package or related methods, please let us know by submitting
  information [here](https://forms.gle/1jBXhasS9SGWgUY37).  This helps us with
  reporting and with further development of the package.  Thanks.

__Acknowledgements__
We gratefully acknowledge support from NSF Grant DMS-2306101,
NSF Grant DMS-1616353, and DOE Grant ASCR PHILMS DE-SC0019246.

__Additional Information__ <br>
https://web.atzberger.org

----
[Documentation](https://web.atzberger.org/mlmod/docs/index.html) |
[Examples](./examples) |
[Paper](./paper/paper.pdf) |
[Atzberger Homepage](https://web.atzberger.org)

