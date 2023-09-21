<p align="left">
<img src="doc_img/force1.png" width="15%"> 
</p>

MPI simulation using a force that acts on particles from a pytorch model $F(X)$.

To run the simulation for the python mlmod package, use 

```mpirun -n 4 python mpi_force1.py```

To generate the PyTorch models, use 

```python gen_mlmod_force1.py```

This requires python package ```mpi4py``` and having compiled the package with
linking to your machine's ```mpi``` libraries.  See the documentation pages
for more information.  

