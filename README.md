<p align="left">
<img src="https://github.com/atzberg/mlmod/blob/main/images/docs/mlmod_software.png" width="90%">
</p>

[Documentation](https://web.math.ucsb.edu/~atzberg/mlmod/docs/index.html) |
[Examples](./examples) |
[Paper](./paper/paper.pdf) 

---

# MLMOD: Data-Driven Modeling and Simulation in LAMMPS (Machine Learning Methods)

**MLMOD** is a Python/C++ machine learning simulation package that integrates data-driven models directly into [LAMMPS](https://www.lammps.org/) molecular dynamics simulations. It enables learning data-driven models for particle dynamics, forces, mobility tensors, and other quantities of interest using trained ML approaches — including neural networks, Gaussian process regression, and other PyTorch-compatible architectures.

---

## 🌐️ Background

Traditional molecular dynamics (MD) simulations use hand-crafted analytical models for interparticle forces and dynamics. MLMOD extends LAMMPS to allow these components to be replaced or augmented by machine-learned models trained from data or derived from physical principles.

### ✴️ Key Capabilities

| Capability | Description |
|---|---|
| **ML Dynamics Integrators** | Replace the LAMMPs integrator with a learned map $A(X^n, V^n, F^n; \theta)$ |
| **ML Force Fields** | Compute interparticle forces via a learned model $F(X, V, F, \text{Type})$ |
| **ML Mobility Tensors** | Hydrodynamic coupling via $M(X)$ (Oseen, RPY, or custom) |
| **Quantities of Interest** | On-the-fly observables $A(X, V, F, I_T, t)$ without modifying dynamics |
|  **MPI Parallelism** | Large-scale simulations with MPI-parallel LAMMPS |

Models are defined in PyTorch, traced with `torch.jit.trace`, exported to `.pt` format, and loaded at runtime by the MLMOD C++ extension to LAMMPS. The LAMMPS engine handles spatial decomposition, neighbor lists, and I/O while MLMOD handles model evaluation.

### 🔺 Paper

> P.J. Atzberger, *MLMOD Package: Machine Learning Methods for Data-Driven Modeling in LAMMPS*, Journal of Open Source Software, 8(89), 5620, (2023). [doi:10.21105/joss.05620](https://doi.org/10.21105/joss.05620)

---

## 📦 Installation

### 🔸 Quick Install (Automatic)

The `quick_install.py` script attempts to detect your platform and install the appropriate pre-built binary:

```bash
python quick_install.py
```

### 🔸 Direct Install (Pre-built Wheel)

Download the appropriate wheel for your platform and install with `pip`:

| Platform | Wheel |
|----------|-------|
| 🐧 Linux Debian 9+ / Ubuntu (standard) | `mlmod_lammps-1.0.3-py3-none-manylinux_2_24_x86_64.whl` |
| 🐧 Linux Debian 9+ / Ubuntu (flexible) | `mlmod_lammps-1.0.3-py3-none-any.whl` |

Pre-built binaries are currently available for Debian 9+/Ubuntu and CentOS 7+, with Python 3.6+.

Download from: `https://web.math.ucsb.edu/~atzberg/mlmod/distr/`

```bash
pip install -U mlmod_lammps-1.0.3-py3-none-manylinux_2_24_x86_64.whl
```

Or install directly from the URL:

```bash
pip install -U https://web.math.ucsb.edu/~atzberg/mlmod/distr/mlmod_lammps-1.0.3-py3-none-manylinux_2_24_x86_64.whl
```

### Install via requirements.txt

```bash
pip install -r requirements.txt
```

**Dependencies:**
- `numpy >= 1.21.1`
- `mlmod-lammps` (pre-built wheel, see above)
- `torch >= 1.11.0` *(optional — needed for model generation, not for running simulations)*

### 🐳 Docker Install (Windows / macOS)

For desktop platforms, use a Docker container with a standard Ubuntu base:

```bash
docker run -it ubuntu:20.04 /bin/bash
apt update && apt install python3-pip
pip3 install mlmod_lammps-1.0.3-py3-none-any.whl
```

Or use the pre-installed Anaconda image:

```bash
docker run -it atzberg/ubuntu_20_04_anaconda1 /bin/bash
conda activate mlmod-lammps
```

### ✅ Verify Installation

```bash
python -c "from mlmod_lammps.tests import t1; t1.test()"
```

For full build-from-source instructions, see the [documentation pages](https://web.math.ucsb.edu/~atzberg/mlmod/docs/index.html).

---

## 🚀 Usage Overview

The general workflow in MLMOD is:

1. **Define and train** a PyTorch model for dynamics, forces, mobility, or QoI.
2. **Export** the model to `.pt` format using `torch.jit.trace`.
3. **Run** a LAMMPS simulation that loads the `.pt` model via the MLMOD plugin.

```python
from mlmod_lammps.lammps import lammps
import mlmod_lammps.util as m_util

L = lammps()
Lc = m_util.wrap_L(L, m_util.Lc_print)

# Standard LAMMPS setup commands
Lc("units nano")
Lc("atom_style angle")
Lc("region mybox prism -18 18 -18 18 -18 18 0 0 0")
Lc("boundary p p p")
Lc("create_box 1 mybox")
# ... add atoms, define fixes using the mlmod model ...
```

---

## ✴️ Examples

All examples are in the [`examples/`](./examples) folder. Each example has a `gen_mlmod_*.py` script to generate the PyTorch model and a `run_sim_*.py` (or `.ipynb`) script to run the LAMMPS simulation. A `run_full.sh` convenience script runs both steps.

---

### 🔸Example 1: ML-Driven Dynamics (`examples/dynamics1/`)

Replaces the standard LAMMPS integrator with a learned map:

$$X^{n+1}, V^{n+1} = A(X^n, V^n, F^n, \text{Type}; \theta)$$

**① Generate the PyTorch model:**

```bash
cd examples/dynamics1
python gen_mlmod_dynamics1.py
```

This defines a `torch.nn.Module` that accepts a concatenated state vector `[X, V, F, Type]` and returns updated positions and velocities. The model is traced and saved to `output/gen_mlmod_dynamics1/gen_001/dyn1_dynamics1.pt`.

**② Run the simulation:**

```bash
python run_sim_dynamics1.py
```

The simulation uses LAMMPS with the MLMOD fix to call the exported model at each timestep. Output is written to VTK files for visualization.

---

### 🔸 Example 2: ML Force Field (`examples/force1/`)

Uses a learned force model $F(X, V, F, \text{Type})$ to compute forces on particles:

$$F_i = F_\theta(X, V, F, \text{Type})$$

**① Generate the PyTorch model:**

```bash
cd examples/force1
python gen_mlmod_force1.py
```

The model is a `torch.nn.Module` that takes the concatenated system state and returns per-atom force vectors. Saved to `output/gen_mlmod_force1/gen_001/F_force1.pt`.

**② Run the simulation:**

```bash
python run_sim_force1.py
```

---

### 🔸Example 3: ML Mobility Tensor (`examples/particles1/`)

Models overdamped Brownian dynamics of particles coupled through a hydrodynamic mobility tensor $M(X)$. Particle positions evolve as:

$$dX = M(X) F \, dt + \text{(noise)}$$

Two mobility models are provided:

- 🔵 **Oseen tensor** (`gen_mlmod_oseen1.py`): $M_{ij}$ computed from pairwise distances.
- 🟢 **Rotne-Prager-Yamakawa (RPY) tensor** (`gen_mlmod_rpy1.py`): Regularized version valid at short range.

**① Generate the model:**

```bash
cd examples/particles1
python gen_mlmod_oseen1.py   # or gen_mlmod_rpy1.py
```

This generates both the diagonal ($M_{ii}$) and off-diagonal ($M_{ij}$) mobility components as separate `.pt` models.

**② Run the simulation:**

```bash
python run_sim_particles1.py
```

A Jupyter notebook version is also available:

```bash
jupyter notebook run_sim_particles1.ipynb
```

Switch between Oseen and RPY by editing the `model_case` variable at the top of the script:

```python
model_case = 'rpy1'   # or 'oseen1'
```

---

### 🔸Example 4: Quantities of Interest (`examples/qoi1/`)

Computes observable quantities $A(X, V, F, I_T, t)$ on-the-fly during a simulation using a ML model, without modifying the dynamics.

```bash
cd examples/qoi1
python gen_mlmod_qoi1.py
python run_sim_qoi1.py
```

---

### 🔸Example 5: MPI Parallel Simulation (`examples/mpi1/`)

Runs a force-field simulation using MPI parallelism across multiple processes.

**① Generate the model:**

```bash
cd examples/mpi1
python gen_mlmod_force1.py
```

**② Run with MPI:**

```bash
mpirun -n 4 python mpi_force1.py
```

> ⚠️ Requires `mpi4py` and an MPI-compiled build of MLMOD/LAMMPS. See the [documentation](https://web.math.ucsb.edu/~atzberg/mlmod/docs/index.html) for build instructions.

---

## 🌟 Visualization

Simulation output is saved as VTK files in the `output/` directory. These can be visualized with [ParaView](https://www.paraview.org/). Each example includes a `vis_pv1.py` script and `vis_pv1.sh` convenience launcher.

---

## 💥 Defining Custom Models

Any PyTorch model that can be traced with `torch.jit.trace` can be used. The general pattern is:

```python
import torch

class MyForceModel(torch.nn.Module):
    def forward(self, z):
        # z is a flat column vector: [X; V; F; Type] for all atoms
        # reshape, compute, and return force vector of shape (num_atoms*num_dim, 1)
        ...
        return forces

model = MyForceModel()
traced = torch.jit.trace(model, torch.zeros((input_size, 1)))
traced.save("my_model.pt")
```

The `mask_input` string (e.g., `"X V F Type"`) controls which state quantities are concatenated and passed to the model.

---

## 📁 Project Structure

```
mlmod/
├── examples/
│   ├── dynamics1/     # ML time-step integrator example
│   ├── force1/        # ML force field example
│   ├── particles1/    # ML mobility tensor (Oseen/RPY) example
│   ├── qoi1/          # Quantities of interest example
│   └── mpi1/          # MPI parallel simulation example
│   └── ...            # Other examples
├── src/               # C++ source for LAMMPS plugin
├── tests/             # Package tests
├── doc/               # Documentation source
├── paper/             # JOSS paper
├── requirements.txt
└── quick_install.py
```

---

## 💡 Citation

If you use MLMOD in your research, please cite:

```bibtex
@article{mlmod_atzberger,
  author    = {Paul J. Atzberger},
  journal   = {Journal of Open Source Software},
  title     = {MLMOD: Machine Learning Methods for Data-Driven Modeling in LAMMPS},
  year      = {2023},
  publisher = {The Open Journal},
  volume    = {8},
  number    = {89},
  pages     = {5620},
  doi       = {10.21105/joss.05620},
  url       = {https://doi.org/10.21105/joss.05620}
}
```

---

## 🔗 Additional Resources

| Resource | Link |
|----------|------|
| 🎬 Video overview | [YouTube](https://youtu.be/BZulqaZT5o0) |
| 📚 Documentation | [web.math.ucsb.edu/~atzberg/mlmod/docs](https://web.math.ucsb.edu/~atzberg/mlmod/docs/index.html) |
| 📬 Mailing list (updates & releases) | [Sign up](https://forms.gle/7seLKxrVrAN9U8Bg7) |
| 🐛 Bug reports | [Submit here](https://forms.gle/xBYLs7Gi1dwYvCR5A) |
| 📣 Usage / citation reporting | [Submit here](https://forms.gle/1jBXhasS9SGWgUY37) |

---

## 🔆 Acknowledgements

Support from NSF Grant DMS-2306101, NSF Grant DMS-1616353, and DOE Grant ASCR PHILMS DE-SC0019246 is gratefully acknowledged. More recent documentation and development used assistance from AI Anthropic Claude 3.6 Sonnet. The core algorithms and mathematical frameworks in this package were designed and manually implemented by the authors.

---

[Documentation](https://web.math.ucsb.edu/~atzberg/mlmod/docs/index.html) |
[Examples](./examples) |
[Paper](./paper/paper.pdf) |
[Atzberger Homepage](http://atzberger.org/)
