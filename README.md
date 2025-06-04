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

To run the non-linear elasticity example you will also need `PyMFEM` with parallel support.

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

- **Physics** – subclass `Physics` and implement `initial_condition` and `solve` to interface with your full order solver.
- **LatentDynamics** – subclass `LatentDynamics` and implement `calibrate` and `simulate` to define your latent ODE model.
- **Model** – extend one of the models in `Model.py` or add a new `torch.nn.Module` that provides `Encode`, `Decode` and `latent_initial_conditions` methods.

Register your classes in `src/Initialize.py` so that configuration files can reference them. Example YAML files and the existing subclasses serve as templates for new problems.
