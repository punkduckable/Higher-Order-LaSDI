# Higher-Order LaSDI

Higher-Order LaSDI provides tools for building reduced-order models from full order simulations using latent-space dynamics identification. The library supports physics that require multiple time derivatives (for example displacement and velocity) and it can handle time series with either uniform or non-uniform time grids. Latent dynamics models such as SINDy or damped-spring systems are provided and new models can be added easily.



## Getting Started

The easiest way to explore the code is with the Jupyter notebook in `examples/2d2p.ipynb`. It demonstrates training and evaluating a model on a Burgers' equation data set. Launch Jupyter and run the notebook step by step.

For a command line workflow use `src/Workflow.py` together with a YAML configuration file. Example configurations live in `examples/*.yml`. The following command trains on Burgers' equation

```bash
python src/Workflow.py --config examples/Burgers.yml
```

`Workflow.py` orchestrates data generation, training via the `GPLaSDI` class and evaluation of the learned model.



## Dependencies

The code has been tested with the following packages:

- Python (3.10)
- numpy (1.26.4)
- pytorch (2.5.1)
- scikit-learn (1.5.2)
- pyyaml (6.0.2)
- jupyter (1.0.0)
- scipy (1.14.1)
- matplotlib (3.9.2)
- seaborn (0.13.2)
- ffmpeg (1.4)
- mfem (4.7.0.1)
- cmake (3.28.1)
- mpi4py (4.0.3)

Note 1: To generate animations of the true FOM solution, learned FOM solution, and the error between them, you need to install ffmpeg.
To do this, run "conda install -c conda-forge ffmpeg" inside of your conda environment. 

Note 2: To run the non-linear elasticity example you will also need `PyMFEM` with parallel support. 
See the subsection below for details. 



## Installing `PyMFEM`

`PyMFEM` can be a little difficult to get running. 
However, most of the examples in this library were built using it. 
Some of the most common problems (in my experience) in building PyMFEM stem from package inconsistencies.
Below is a tutorial on how to install PyMFEM in a way that (hopefully) avoids most of these common issues.

First, create a new conda environment to install PyMFEM:

$$
\begin{aligned}
&\text{conda create --name=PyMFEM python=3.10} \\
&\test{conda activate PyLASDI} 
\end{aligned}
$$

Next, clone the PyMFEM repository (below is the SSH version) and checkout the version 4.7.0.1 release: 

$$
\begin{aligned}
&\text{git clone git@github.com:mfem/PyMFEM.git} \\
&\text{cd ./PyMFEM/} \\
&\text{git checkout v_4.7.0.1}
\end{aligned}
$$

Next, install the package's dependencies. This will probably install the wrong version of a few packages; we we will fix those issues next. 

$$\text{pip install -r requirements.txt}$$

PyMFEM (at least version 4.7.0.1) is not compatible with cmake 4.0 and beyond. Thus, we need to install an earlier version of cmake:

$$
\begin{aligned}
&\text{pip uninstall cmake} \\
&\text{pip install cmake==3.28.} 
\end{aligned}
$$

Next, you need to install MPI to get MFEM running in parallel. On a mac (with homebrew), you can do the following:

$$\text{brew install openmpi}$$

Once you have MPI installed, you need to add the mpi4py package to your environment to enable MPI in python code: 

$$\text{pip install mpi4py==4.0.3}$$

With all of this preparation work done, you should now be able to install PyMFEM using the following command:

$$\text{python setup.py install -v --user --with-parallel --with-gslib --CC=gcc --CXX=g++ --MPICC=mpicc --MPICXX=mpic++ --with-lapack}$$





## Repository Layout

- `src/Physics` – physics models and the base [`Physics`](src/Physics/Physics.py) class. Concrete solvers such as Burgers, Advection and NonlinearElasticity subclass this and implement `initial_condition` and `solve`.
- `src/LatentDynamics` – latent space dynamics models. [`LatentDynamics`](src/LatentDynamics/LatentDynamics.py) defines the interface (`calibrate` and `simulate`). Implementations include [`SINDy`](src/LatentDynamics/SINDy.py) and [`DampedSpring`](src/LatentDynamics/DampedSpring.py).
- `src/GPLaSDI.py` – the main training loop encapsulated by the `GPLaSDI` class. It couples the physics, model and latent dynamics and supports non-uniform time grids by switching finite-difference formulas based on the `Uniform_t_Grid` flag.
- `src/Model.py` – neural network autoencoders (`Autoencoder` and `Autoencoder_Pair`) used to encode full order states.
- `src/ParameterSpace.py` – utilities for defining training and testing parameter grids.
- `src/Workflow.py` – command line driver that loads configuration files, initializes all components and runs training.
- `examples/` – configuration files and the 2‑d‑2‑p notebook demonstrating typical usage.
- `src/Utilities` – finite difference and ODE solvers for both uniform and non-uniform grids.


## Non-uniform Time Grids

Physics objects expose a `Uniform_t_Grid` attribute which determines how derivatives are computed. When set to `False` higher-order schemes are replaced with non-uniform versions as shown in [`GPLaSDI`](src/GPLaSDI.py) where `Derivative1_Order2_NonUniform` is used when necessary.



## Extending the Code

New applications can be implemented by deriving from the appropriate base classes:

- **Physics** – subclass `Physics` and implement `initial_condition` and `solve` methods to interface with your full order solver. You need to add any new `Physics` subclasses to the `physics_dict` dictionary in `Initialize.py` before you can use start using them/referencing them in a configuration file.
- **LatentDynamics** – subclass `LatentDynamics` and implement `calibrate` and `simulate` methods to define your latent ODE model. You need to add any new `LatentDynamics` subclasses to the `ld_dict` dictionary in `Initialize.py` before you can start using them/referencing them in a configuration file.
- **Model** – extend one of the models in `Model.py` or add a new `torch.nn.Module` that provides `Encode`, `Decode` and `latent_initial_conditions` methods. You need to add any new `Model` subclasses to the `model_dict` and `model_load_dict` dictionaries in `Initialize.py` before you can start using them/referencing them in a configuration file.

Register your classes in `src/Initialize.py` so that configuration files can reference them. Example YAML files and the existing subclasses serve as templates for new problems.
