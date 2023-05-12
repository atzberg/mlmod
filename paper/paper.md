---
title: 'MLMOD: Machine Learning Methods for Data-Driven Modeling in LAMMPS'
tags:
  - Python
  - machine learning 
  - dynamics
authors:
  - name: Paul J. Atzberger
    orcid: 0000-0001-6806-8069
    affiliation: 1
affiliations:
 - name: Paul J. Atzberger, Professor, University California Santa Barbara
   index: 1
date: 4 May 2023
bibliography: paper.bib
---


# Summary

``MLMOD`` is a software package for incorporating machine learning approaches
and models into simulations of microscale mechanics and molecular dynamics in
LAMMPS. Recent machine learning approaches provide promising data-driven
approaches for learning representations for system behaviors from experimental
data and high fidelity simulations.  The package faciliates learning and using
data-driven models for (i) dynamics of the system at larger spatial-temporal
scales (ii) interactions between system components, (iii) features yielding
coarser degrees of freedom, and (iv) features for new quantities of interest
characterizing system behaviors.  ``MLMOD`` provides hooks in LAMMPS for (i)
modeling dynamics and time-step integration, (ii) modeling interactions, and
(iii) computing quantities of interest characterizing system states. The
package allows for use of machine learning methods with general model classes
including Neural Networks, Gaussian Process Regression, Kernel Models, and
other approaches.  Here we discuss our prototype C++/Python package, aims, and
example usage.  The package is integrated currently with the mesocale and
molecular dynamics simulation package LAMMPS and PyTorch.  The source code for
this initial version 1.0.0 of ``MLMOD`` has been archived to Zenodo with a DOI
in [@zenodo].  For related papers, examples, updates, and additional
information see <https://github.com/atzberg/mlmod> and 
<http://atzberger.org/>.

# Statement of Need

A practical challenge in using machine learning methods for simulations is the
efforts required to incorporate learned system features to augment existing
models and simulation methods.  Our package ``MLMOD`` aims to address this
aspect of data-driven modeling by providing a general interface for
incorporating ML models using standardized representations and by leveraging
existing simulation frameworks such as LAMMPS [@Plimpton:1995].  Our ``MLMOD``
package provides hooks which are triggered during key parts of simulation
calculations.  In this way standard machine learning frameworks can be used to
train ML models, such as PyTorch [@Paszke:2019] and TensorFlow [@Abadi:2015],
with the resulting models more amenable to being translated into practical
simulations.  The models obtained from learning can be accomodated in many
forms, including Deep Neural Networks (DNNs) [@Goodfellow:2016], Kernel
Regression Models (KRM) [@Scholkopf:2001], Gaussian Process Regression (GPR)
[@Rasmussen:2004], and others [@Hastie:2001].  

#  Data-Driven Modeling

Recent advances in machine learning, optimization, and available computational
resources are presenting new opportunities for data-driven modeling and
simulation in the natural sciences and engineering.  Empirical successes in
deep learning suggest promising non-linear techniques for learning
representations for system behaviors and other underlying features
[@Hinton:2006; @Goodfellow:2016].  Many previous deep learning methods have
been developed for problems motivated by image analysis and natural language
processing.  However, scientific computations and associated dynamical systems
present a unique set of challenges for developing and employing recent machine
learning approaches [@Atzberger:2018;@Brunton:2016;@Schmidt:2009].  

In scientific and engineering applications there are often important
constraints arising from physical principles required to obtain plausible
models and there is a need for results to be more interpretable.  In
large-scale scientific computations, bottom-up modeling efforts aim to start as
close as possible to first principles and perform computations to obtain
insights into larger-scale emergent behaviors.  Examples include the
rheological responses of soft materials and complex fluids from microstructure
interactions [@Atzberger:2013; @Bird:1987; @Lubensky:1997; @Kimura:2009],
molecular dynamics modeling of protein structures and functional domains from
atomic level interactions [@Mccammon:1988; @Karplus:2002; @Karplus:1983;
@Plimpton:1995], and prediction of weather and climate phenomena from detailed
physical models, sensor data, and other measurements [@Richardson:2007;
@Bauer:2015].  Obtaining observables and quantities of interest (QoI) from
simulations of such high fidelity detailed models can involve significant
computational resources [@Lusk:2011; @Sanbonmatsu:2007; @Washington:2009;
@Pan:2021; @Murr:2016; @Giessen:2020].  Data-driven learning methods present
opportunities to formulate more simplified models, provide model flexibility to
accomodate subtle effects, or make predictions which are less computationally
expensive.

Data-driven modeling can take many forms.  As a specific motivation for the
package and our initial implementations, we discuss a specific case
in detail, but our package also can be used more broadly.  In particular, we
consider  detailed molecular dynamics simulations of large spherical colloidal
particles within a bath of much smaller solvent particles.  A common problem is
to infer interaction laws between the colloidal particles given the surrounding
environment arising from the type of solution, charge, and other physical
conditions.  There is extensive theoretical literature on colloidal
interactions and approximate models [@Derjaguin:1941; @Doi:2013; @Jones:2002].
While analytic approaches have had success, there are many settings where
challenges remain which limit the accuracy [@Jones:2002; @Atzberger:2018b].
Computational modeling and simulation provides opportunities for capturing
phenomena in more physical detail and with better understanding of contributing
effects.

While simulations of colloids including the solvent and other environmental
factors can be used for making predictions, such computations can be expensive
given the many degrees of freedom and small time-scales of solvent-solvent
interactions.  Colloid coarse-grained models are sought which utilize
separation in scales, such as the contrast in size with the solvent and
dynamical time-scales.  In these circumstances, coarse-grained models aim to
capture the effective colloidal interactions and their dynamics.

![Data-driven modeling from detailed molecular simulations can be used
to train machine learning (ML) models for performing simulations at larger
spatial-temporal scales.  This can include models for the dynamics,
interactions, or for computing quantities of interest (QoI) characterizing the
system state.  The colloidal system for example could be modeled by dynamics at
a larger scale with a mobility $M$ obtained from training.
In the ``MLMOD`` package, the ML models can be represented by Deep Neural Networks,
Kernel Regression Models, or other model classes.](fig/data_driven_modeling2.png){ width=75% }

Relative to detailed molecular dynamics simulations, this motivates a
simplified model for the effective colloid dynamics 
$$\frac{d\mathbf{X}}{dt} = \mathbf{M}(\mathbf{X})\mathbf{F} 
+ k_B{T}\nabla_X \cdot \mathbf{M}(\mathbf{X}) + \mathbf{g}$$

$$< \mathbf{g}(s) \mathbf{g}(t)^T > = 2 k_B{T} \mathbf{M}(\mathbf{X}) \delta(t - s).$$
The $\mathbf{X} \in \mathbb{R}^{3n}$ refers
to the collective configuration of all $n$ colloids in these Smoluchowski
dynamics [@Smoluchowski:1906].  The $\mathbf{g}(t)$ gives the thermal fluctuations
for the temperature corresponding to $k_B{T}$.
Here, the main objectives in this model are
to determine (i) the *mobility tensor* $M = M(\mathbf{X})$ which
captures the effective dynamic coupling between the colloidal particles, and
(ii) the *interaction laws* $\mathbf{F}$ for configurations
$\mathbf{X}$.

Machine learning methods provide data-driven approaches for
learning representations and features for such modeling.  Optimization using
appropriate loss functions and training protocols can be used to identify
system features underlying interactions, symmetries, and other structures.  In
machine learning methods this is accomplished by using a class of
representations and by training with data to identify models from this class.
For making predictions in unobserved cases, this allows for interpolation, and
in some cases even extrapolation, especially when using explicit low
dimensional latent spaces or when imposing other inductive biases
[@Atzberger:2023; @Atzberger:2022].  For example, consider the colloidal
example in the simiplified case when we
assume the interactions can be approximated as pairwise.  The problem reduces
to a model $M = M(\mathbf{X}_1,\mathbf{X}_2)$ depending on six dimensions.
This can be further constrained to learn only symmetric positive semi-definite
tensors, for example by learning $L = L(\mathbf{X}_1,\mathbf{X}_2)$ to generate
$M = LL^T$.  

There are many ways we can obtain the model $M$.  For example, a common way to
estimate mobility in fluid mechanics is to apply active forces $\mathbf{F}$ and
compute the velocity response 
$< \mathbf{V} > = < {d\mathbf{X}}/{dt} > \approx 
\tau^{-1}< \Delta_{\tau} \mathbf{X}(t)> \approx \mathbf{M}\mathbf{F}$. 
The $\Delta_{\tau} \mathbf{X}(t) = \mathbf{X}(t + \tau) - \mathbf{X}(t)$ 
for $\tau$ chosen carefully.  For large
enough forces $\mathbf{F}$, the thermal fluctuations can be averaged away
readily by repeating this measurement and taking the mean.  In statistical
mechanics, another estimator is obtained when $\mathbf{F} = 0$ by using the
passive fluctuations of system. A moment-based estimator commonly used is
$M(\mathbf{X}) \approx ({2k_B{T}\tau)^{-1}} 
< \Delta_{\tau} \mathbf{X}(t)\Delta_{\tau} \mathbf{X}(t)^T >$ 
for $\tau$ chosen carefully.  While theoretically each of these estimators give
information on $M$, in practice there can be subtleties such as a good choice
for $\tau$, magnitude for $\mathbf{F}$, and role of fluctuations.  Even for
these more traditional estimators, it could still be useful for storage
efficiency and convenience to train an ML model to provide a compressed
representation and for interpolation for evaluating $M(\mathbf{X})$.  

Machine learning methods also could be used to train more directly from
simulation data for sampled colloid trajectories $\mathbf{X}(t)$
[@Atzberger:2023;Nielsen:2000].  The training would select an ML model
$M_\theta$ over some class of models $H$ parameterized by $\theta$, such as the
weights and biases of a Deep Neural Network.  For instance, this could be done
by Maximum Likelihood Estimation (MLE) or other losses from the trajectory data
$\mathbf{X}(t)$.  The MLE optimizes the objective 
$$M_{\theta} = \arg\min_{M_\theta \in {H}} 
-\log\rho_\theta(\mathbf{X}(t_1),\mathbf{X}(t_2),\ldots,\mathbf{X}(t_m)).$$ 
The $\rho_\theta$ denotes the likelihood probability density for the model 
with $M = M_\theta$ and observing the trajectory data $\{\mathbf{X}(t_i)\}$.
To obtain tractable and robust training algorithms, further approximations and
regularizations may be required to the MLE problem or alternatives used.  This
could include using variational inference approaches, further restrictions on
the model architectures, priors, or other information [@Atzberger:2020;
@Atzberger:2023; @Kingma:2014; @Blei:2017]. Combining such approximations with
further regularizations also could help facilitate learning, including of
possible symmetries and other features of trained models $M(\mathbf{X}) =
M_\theta$.  

The ``MLMOD`` package provides ways for transferring such learned models into
practical simulations within LAMMPS.  We discussed here one example of a basic
data-driven modeling approach for colloids.  The ``MLMOD`` package can be used
more generally and supports broad classes of models for incorporating machine
learning results into simulation components.  Components can include the
dynamics, interactions, or computing quantities of interest.  The initial
implementations we present supports the basic mobility modeling framework as a
proof-of-concept, with longer-term aims to support more general classes of
reduced dynamics and interactions in future releases.

# Structure of the Package Components 

The package is organized as a module
within LAMMPS that is called each time-step and has the potential to serve
multiple roles within simulations.  This includes (i) serving as a time-step
integrator updating the configuration of the system based on a specified
learned model, (ii) evaluating interactions between system components to
compute energy and forces, and (iii) computing quantities of interest (QoI) that
can be used as state information during simulations or in statistics.
The package is controlled by external XML files that specify the mode of
operation and source for pre-trained models and other information, see the
schematic in Figure 2.

![The MLMOD Package is structured modularly with subcomponents
for providing ML models in simulations for the dynamics, interactions, and
computing quantities of interest (QoI) characterizing the system state.  The
package makes use of standardized data formats such as XML for inputs and
export ML model formats from machine learning frameworks.](fig/mlmod_schematic3.png){ width=65% }

The ``MLMOD`` Package is
incorporated into a simulation by either using the LAMMPS scripting language or
the python interface.  This is done using the "fix" command in
LAMMPS [@Plimpton:1995], with this terminology historically motivated
by algorithms for "fixing" molecular bonds as rigid each time-step.  For our
package the command to set up the triggers for our algorithms is 
``fix m1 mlmod all filename.mlmod_params.``  This specifies the tag "m1" for this fix,
particle groups controlled by the package as "all", and the XML file of
parameters.  The XML file ``filename.mlmod_params`` specifies the ``MLMOD``
simulation mode and where to find the associated exported ML models.  An
example and more details are discussed below in the section on package usage.
The ``MLMOD`` Package can evaluate machine learning models using frameworks such as
C++ PyTorch API.  This allows both for the possibility of doing on-the-fly
learning and for using trained models to augment simulations.

A common approach would be to learn ML models by training on trajectory data
from detailed high fidelity simulations using a machine learning framework,
such as PyTorch [@Paszke:2019].  Once the model is trained, it can be
exported to a portable format such as Torch [@Collobert:2011].  The ``MLMOD``
package would import these pre-trained models from Torch files such as
``trained_model.pt``.  This allows for these models to then be invoked by ``MLMOD``
to provide elements for (i) performing time-step integration to model dynamics,
(ii) computing interactions between system components, and (iii) computing
quantities of interest (QoI) for further computations or as statistics.  This
provides a modular and general way for data-driven models obtained from
training with machine learning methods to be used to govern LAMMPS simulations.  

#  Example Usage of the Package
We give one basic example usage of the package in the case for modeling
colloids using a mobility tensor $M$.  To set up the triggers for the ``MLMOD``
package during LAMMPS simulations a typical command would look like 

``fix m1 c_group mlmod model.mlmod_params`` 

The ``m1`` gives the tag for the fix, ``c_group``
specifies the label for the group of particles controlled by this instance of
the ``MLMOD`` package. The ``mlmod`` specifies to use the ``MLMOD`` package with
XML parameter file ``model.mlmod_params``.  The XML parameter file controls the
package modes and the use of associated exported ML models.

Multiple instances of ``MLMOD`` package are permitted and can be used to
control different groups of particles by adjusting the ``c_group``.  The
package is designed with modularity so a *mode* is first defined in a parameter
file and then different sets of algorithms and parameters can be used within
the same simulation.  For the mobility example, an implementation is given by
the ``MLMOD`` simulation mode ``dX_MF_ML1``. For this modeling mode, a typical
parameter file would look like the following.
```
<?xml version="1.0" encoding="UTF-8"?> 
<MLMOD> 
  <model_data type="dX_MF_ML1"> 
    <M_ii_filename value="M_ii_torch.pt"/> 
    <M_ij_filename value="M_ij_torch.pt"/>
  </model_data> 
</MLMOD> 
```

This specifies for an assumed mobility tensor of pairwise interactions the
models for the self-mobility responses $M_{ii}(\mathbf{X})$ and the pairwise
mobility response $M_{ij}(\mathbf{X}) = M_{ji}(\mathbf{X})$, where $\mathbf{X}
= (\mathbf{X}_{1},\mathbf{X}_{2})$.  For example, a hydrodynamic model for
interactions when the two colloids of radius $a$ are not too close together is
to use the Oseen Tensors 
$M_{ii} = (6\pi\eta a)^{-1}{I}$ and 
$M_{ij} = (8\pi\eta r)^{-1}\left({I} + r^{-2}\mathbf{r}\mathbf{r}^T \right)$. 
The $\eta$ is the fluid viscosity, 
$\mathbf{r} = \mathbf{X} _{i}(t) -\mathbf{X} _{j}(t)$ 
with $r = \|\mathbf{r}\|$ give the particle separation.  The responses are 
$\mathbf{V} _{\ell} = M_{\ell m} \mathbf{F} _{m}$ 
with $\ell,m \in \{1,2\}$ and summation notation. For different environments
surrounding the colloids, these interactions would be learned from simulation
data. 

The ``dX_MF_ML1`` mode indicates this type of mobility model has interactions
from learned ML models.  The ML models are given by the files ``M_ii_torch.pt``
and ``M_ij_torch.pt``.  Related modes can also be implemented to extend models
to capture more complicated interactions or near-field effects.  For example,
to allow for localized many-body interactions with ML models giving
contributions to mobility $M(\mathbf{X})$.  In this way ``MLMOD`` can be used
for hybrid modeling combining ML models with more traditional modeling
approaches within a unified framework.

This gives one example, the ML interactions and integrators can be more general 
using any exported model from the machine learning framework.  Currently, the 
implementation uses PyTorch and
the export format based on torch script with ``.pt`` files.  This allows for a
variety of models to be used ranging from those based on Deep Neural Networks,
Kernel Regression Models, and others.  

# Conclusion 
The package ```MLMOD``` provides capabilities in LAMMPS for incorporating into
simulations data-driven models for dynamics and interactions obtained from
training with machine learning methods.  We describe here our initial
implementation.  For updates, examples, and additional information please see
<https://github.com/atzberg/mlmod> and <http://atzberger.org/>.

# Acknowledgements

Authors research supported by grants DOE Grant ASCR PHILMS DE-SC0019246, NSF
Grant DMS-1616353, and NSF Grant DMS-2306101.  Authors also acknowledge UCSB
Center for Scientific Computing NSF MRSEC (DMR1121053) and UCSB MRL NSF
CNS-1725797.  P.J.A. would also like to acknowledge a hardware grant from
Nvidia.  

# References




