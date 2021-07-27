<p align="left">
<img src="https://github.com/atzberg/mlmod/blob/main/images/docs/mlmod_software.png" width="80%"> 
</p>

### Mlmod: Machine Learning Methods for Data-Driven Modeling in LAMMPS
Now available with Jupyter notebooks and Python scripts for readily setting up models and simulations.

**Mlmod** is a Python/C++ package for utilizing machine learning methods and data-driven modeling for simulations in LAMMPS.
The package includes methods for time-step integrators for dynamics and interactions using general ML model classes, 
including Neural Networks, Kernel Regression, and others.  Models can be trained and exported from PyTorch or from 
other machine learning frameworks.  

LAMMPS is an optimized molecular dynamics package in C/C++ providing many interaction potentials and analysis tools for modeling and simulation.  Interaction methods include particle-mesh electrostatics, common coarse-grained potentials, many-body interactions, and others.

**Quick Start**

To install pre-compiled package for Python use

```pip install mlmod-lammps```

To test the package installed run 

```python -c "from mlmod_lammps.tests import t1; t1.test()"```

Pre-compiled binaries for (Debian 9+/Ubuntu and Centos 7+, Python 3.6+).

__If you installed previously__ this package, please be sure to update to the latest version using 
```pip install --upgrade mlmod-lammps```

__For example models, notebooks,__ and scripts, see the [examples folder](https://github.com/atzberg/mlmod/tree/master/examples).  

**Other ways to install the package**
For running prototype models and simulations on a desktop, such as Windows and MacOS, we recommend using Docker container.  For example, install [Docker Desktop](https://docs.docker.com/desktop/), or docker for linux, and then load a standard ubuntu container by using in the terminal ```docker run -it ubuntu:20.04 /bin/bash```  You can then use ```apt update; apt install python3-pip```, and can then pip install and run the simulation package as above.  Note use command ```python3``` in place of ```python``` when calling scripts.  

For more information on other ways to install or compile the package, please see the documentation page @@@

**Python/Jupyter Notebooks for Modeling and Simulations** 

MLMOD now easily can be set up using Python or Jupyter Notebooks.  See the documentation page and tutorial video for details, @@@

**Downloads:** The source package and additional binaries are available at the webpage: @@@

---
Please cite the paper below when referencing this package:
```
@article{atz_mlmod_lammps_machine_learning_models,
title = {MLMOD Package: Machine Learning Methods for Data-Driven Models in LAMMPS},
author = {Atzberger, P. J.},
journal = {arXiv @@@},
year = {2021},
doi = {@@@},
URL = {http://atzberger.org/},
}  
```
----

__Mailing List for Future Updates and Releases__

Please join the mailing list for future updates and releases [here](@@@).

__Bugs or Issues__

If you encounter any bugs or issues please let us know by providing information [here](@@@).

__Please submit usage and citation information__

If you use this package or related methods, please let us know by submitting information [here](@@@).  
This helps us with reporting and with further development of the package.  Thanks.

__Acknowledgements__
We gratefully acknowledge support for this project from NSF Grant DMS-1616353 and DOE Grant ASCR PHILMS DE-SC0019246.

__Additional Information__ <br>
http://atzberger.org/
