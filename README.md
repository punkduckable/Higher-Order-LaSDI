# Higher-Order LaSDI

Higher-Order LaSDI provides tools for building reduced-order models (ROMs) from full-order simulations using latent-space dynamics identification with Gaussian Process-based greedy sampling. The library supports physics that require multiple time derivatives (e.g., displacement and velocity for second-order systems) and handles time series with either uniform or non-uniform time grids. Latent dynamics models such as SINDy and damped-spring systems are provided, and new EncoderDecoders can be added easily.

## Key Features

- **Gaussian Process-based Greedy Sampling**: Adaptive parameter space exploration using GP uncertainty quantification
- **Higher-Order Dynamics**: Native support for systems requiring multiple time derivatives
- **Flexible Time Grids**: Handles both uniform and non-uniform temporal discretizations with automatic finite difference scheme selection
- **Multiple Loss Functions**: Reconstruction, latent dynamics, rollout, IC rollout, chain rule, and consistency losses with configurable weights
- **Autoencoder Architectures**: Standard autoencoder and paired autoencoder for higher-order systems
- **Rich Activation Functions**: Support for 20+ activation functions including sine, ReLU, tanh, GELU, and more
- **Visualization Tools**: Automated plotting of latent trajectories, error heatmaps, and solution animations
- **Training stability & diagnostics**: Gradient clipping (config: `lasdi.gplasdi.gradient_clip`) to prevent exploding gradients, and per-parameter loss logging to `results/*_loss_by_param.pkl` for post-hoc analysis

## Getting Started

For a command-line workflow, use `src/Workflow.py` together with a YAML configuration file. Example configurations are provided in `examples/*.yml`. 

### Basic Usage

```bash
python src/Workflow.py --config examples/KleinGordon.yml
```

This command will:
1. Generate or load training data from the physics solver
2. Train the autoencoder and latent dynamics using GPLaSDI
3. Perform greedy sampling to adaptively select new training points
4. Evaluate the learned ROM on test data
5. Generate plots and error metrics

### Workflow Components

`Workflow.py` orchestrates the following pipeline:
- **Data Generation**: Calls physics solvers to generate training trajectories
- **Training**: Uses the `BayesianGLaSDI` class to train autoencoders and latent dynamics
- **Greedy Sampling**: Fits Gaussian Processes to latent dynamics coefficients and selects new training points based on GP uncertainty
- **Evaluation**: Computes rollout errors, relative errors, and standard deviations
- **Visualization**: Generates latent trajectory plots, error heatmaps, and solution animations

## Available Examples

The `examples/` directory contains configuration files for various physics problems:

| Example | Physics Type | Order | Grid Type |
|---------|-------------|-------|-----------|
| `Explicit.yml` | Custom explicit solver | 1st | Uniform |
| `Burgers.yml` | Burgers equation | 1st | Uniform |
| `Burgers2D.yml` | 2D Burgers equation | 1st | Uniform |
| `Thermal.yml` | Thermal diffusion (HDF5) | 1st | Uniform |
| `Advection.yml` | Advection equation (PyMFEM) | 1st | Uniform |
| `WaveEquation.yml` | Wave equation (PyMFEM) | 2nd | Uniform |
| `KleinGordon.yml` | Klein-Gordon equation (PyMFEM) | 2nd | Uniform |
| `Telegraphers.yml` | Telegraphers equation (PyMFEM) | 2nd | Uniform |
| `NonlinearElasticity.yml` | Nonlinear elasticity (PyMFEM) | 2nd | Uniform |

**Note**: PyMFEM examples require PyMFEM installation (see below).

## Repository Layout

### Core Components

- **`src/Workflow.py`** – Main command-line driver that loads configuration files, initializes components, and runs the training pipeline
- **`src/GPLaSDI.py`** – The `BayesianGLaSDI` class implements the main training loop with GP-based greedy sampling
- **`src/Initialize.py`** – Factory functions for initializing trainers, EncoderDecoders, physics solvers, and latent dynamics from config files
- **`src/EncoderDecoder`** – Neural network architectures:
  - `EncoderDecoder.py`: Base `EncoderDecoder` class.
  - `MLP.py`: Flexible MLP with customizable activations
  - `Autoencoder.py`: Standard autoencoder for first-order systems
  - `Autoencoder_Pair.py`: Paired autoencoder for higher-order systems (encodes multiple derivatives)
- **`src/ParameterSpace.py`** – Parameter space management, grid generation, and train/test split utilities
- **`src/SolveROMs.py`** – ROM simulation functions:
  - `average_rom()`: Simulate using GP mean predictions
  - `sample_roms()`: Simulate using samples from GP posteriors
  - Error computation and uncertainty quantification
- **`src/Sample.py`** – Greedy sampling logic and training space updates
- **`src/Enums.py`** – Enumerations for workflow states (`NextStep`, `Result`)

### Physics Solvers

- **`src/Physics/Physics.py`** – Base `Physics` class defining the interface
- **Built-in Python Solvers**:
  - `Burgers.py` – 1D Burgers equation
  - `Burgers2D.py` – 2D Burgers equation
  - `BurgersSecondOrder.py` – Second-order Burgers formulation
  - `Explicit.py` / `ExplicitSecondOrder.py` – Custom explicit solvers
  - `Thermal.py` – Thermal diffusion (loads from HDF5 files)
- **PyMFEM-based Solvers** (in `src/Physics/PyMFEM/`):
  - `advection.py` – Advection equation
  - `wave_equation.py` – Wave equation (2nd order)
  - `klein_gordon.py` – Klein-Gordon equation (2nd order)
  - `telegraphers.py` – Telegraphers equation (2nd order)
  - `nonlinear_elasticity.py` – Nonlinear elasticity (2nd order)

### Latent Dynamics

- **`src/LatentDynamics/LatentDynamics.py`** – Base `LatentDynamics` class
- **`src/LatentDynamics/SINDy.py`** – Sparse Identification of Nonlinear Dynamics
  - Uses polynomial library (currently order ≤ 1)
  - Supports learnable coefficients or least-squares fitting
- **`src/LatentDynamics/DampedSpring.py`** – Physics-informed damped spring dynamics

### Utilities

- **`src/Utilities/GaussianProcess.py`** – GP training and prediction:
  - `fit_gps()`: Fit GPs to latent dynamics coefficients
  - `eval_gp()`: Evaluate GP mean and standard deviation
  - `sample_coefs()`: Sample from GP posteriors
  - Automatic input/output scaling for numerical stability
  - Configurable kernels (Matern, RBF)
- **`src/Utilities/FiniteDifference.py`** – Derivative approximations:
  - Order 2 and Order 4 schemes for uniform grids
  - Order 2 non-uniform grid schemes
  - First and second derivatives
- **`src/Utilities/FirstOrderSolvers.py`** – ODE solvers for first-order systems:
  - RK1 (Euler), RK2, RK4 methods
- **`src/Utilities/SecondOrderSolvers.py`** – ODE solvers for second-order systems:
  - RK1, RK2, RK4 methods for second-order ODEs
- **`src/Utilities/Logging.py`** – Logging utilities and dictionary logging
- **`src/Utilities/Timing.py`** – `Timer` class for performance profiling
- **`src/Utilities/MoveOptimizer.py`** – Utilities for moving optimizers between devices (CPU/GPU)

### Visualization

- **`src/Plot.py`** – Plotting functions:
  - `Plot_Latent_Trajectories()`: Visualize latent space dynamics with GP uncertainty
  - `Plot_Heatmap2d()`: 2D parameter space heatmaps
  - `trainSpace_RelativeErrors_Heatmap()`: Error visualization for training parameters
- **`src/Animate.py`** – Animation generation:
  - `make_solution_movies()`: Create MP4 animations of solutions
  - `_scalar_anim()`: Animations for scalar fields on 2D/3D point clouds
  - `_vector_anim()`: Animations for vector fields
  - Requires `ffmpeg`

## Configuration Files

Configuration files are YAML-based and specify:

### LaSDI Settings (`lasdi`)
- Trainer type (currently `gplasdi`)
- Learning rate, number of GP samples
- Normalization settings
- Rollout parameters (initial probability, update frequency)
- Training iterations and greedy sampling iterations
- Loss weights: `recon`, `LD`, `rollout`, `IC_rollout`, `stab` (stability penalty), `chain_rule`, `consistency`
- Loss types: `MSE` or `MAE`
- Learnable coefficients flag

### Workflow Settings (`workflow`)
- Restart capability (load from checkpoint)
- Plotting options

### Parameter Space (`parameter_space`)
- Parameter definitions (uniform, list, or file-based)
- Test space configuration (grid, random, or file-based)

### EncoderDecoder Architecture (`EncoderDecoder`)
- EncoderDecoder type: `ae` (Autoencoder), `pair` (Autoencoder_Pair), or `3d CNN` (CNN_3D).
- Hidden layer widths
- Activation functions
- Latent dimension

### Latent Dynamics (`latent_dynamics`)
- Type: `sindy`, `spring`, or `switch sindy`.
- Stability regularization (stability penalty)

### Physics (`physics`)
- Physics type (must match a key in `physics_dict`)
- Physics-specific parameters (grid sizes, time steps, domain bounds, etc.)

## Dependencies

### Core Dependencies

Tested with the following versions:

- Python (3.10)
- numpy (1.26.4)
- torch (2.5.1)
- scikit-learn (1.5.2)
- pyyaml (6.0.2)
- scipy (1.14.1)
- matplotlib (3.9.2)
- seaborn (0.13.2)

These pacakages are listed in the "requirements.txt" file


### Installing with venv (recommended)

Create the virtual environment in a `venv/` subdirectory of `Higher-Order-LaSDI/`:

```bash
cd Higher-Order-LaSDI
python3 -m venv venv
source venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt 

# Alternatively, you can install the core dependencies with tested versions.
python -m pip install \
  numpy==1.26.4 \
  scipy==1.14.1 \
  scikit-learn==1.5.2 \
  pyyaml==6.0.2 \
  matplotlib==3.9.2 \
  seaborn==0.13.2 \

# PyTorch (pick the right wheel for your platform/CUDA)
python -m pip install torch==2.5.1
```

Add optional dependencies to the same venv:

```bash
# HDF5 data loading (Thermal example)
python -m pip install h5py==3.14.0

# PyMFEM prerequisites (see full PyMFEM build steps below)
python -m pip install cmake==3.28.1 mpi4py==4.0.3

# Jupyter for error diagnostics
python -m pip install jupyter==1.0.0
```



Deactivate with:

```bash
deactivate
```

### Optional Dependencies

**For animations:**
- ffmpeg (1.4)
- Install via your system package manager/module (e.g., `apt-get install ffmpeg`, `brew install ffmpeg`, or `module load ffmpeg`)

**For HDF5 data loading (Thermal example):**
- hdf5 (the system HDF5 library, not a python pacakage; 1.14.5)
- h5py (3.14.0)

**For PyMFEM examples:**
- mfem (4.7.0.1)
- cmake (3.28.1)
- mpi4py (4.0.3)
- See installation instructions below

**For Jupyter notebooks (error diagnostics):**
- jupyter (1.0.0)

## Installing PyMFEM

PyMFEM can be challenging to install. Below is a step-by-step guide to avoid common issues.

### 1. Create/Activate a venv (project-local)

PyMFEM should be installed into the same `Higher-Order-LaSDI/venv` created above.

```bash
cd Higher-Order-LaSDI
# If you haven't created the venv yet:
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### 2. Clone and Checkout PyMFEM

```bash
git clone git@github.com:mfem/PyMFEM.git
cd ./PyMFEM/
git checkout v4.7.0.1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Fix CMake Version

PyMFEM v4.7.0.1 requires cmake < 4.0:

```bash
pip uninstall cmake
pip install cmake==3.28
```

### 5. Install MPI

**On macOS with Homebrew:**
```bash
brew install openmpi
```

**On Linux:**
```bash
# Ubuntu/Debian
sudo apt-get install openmpi-bin libopenmpi-dev

# CentOS/RHEL
sudo yum install openmpi openmpi-devel
```

### 6. Install mpi4py

```bash
pip install mpi4py==4.0.3
```

### 7. Build PyMFEM

```bash
python setup.py install -v --with-parallel --with-gslib \
  --CC=gcc --CXX=g++ --MPICC=mpicc --MPICXX=mpic++ --with-lapack
```

**Note**: On some systems you may need to adjust compiler names (e.g., `gcc-11`, `g++-11`).

## Non-Uniform Time Grids

Physics objects expose a `Uniform_t_Grid` attribute which determines derivative computation:
- When `True`: Uses higher-order finite difference schemes (Order 4 when available)
- When `False`: Uses non-uniform grid schemes (currently Order 2)

The `BayesianGLaSDI` class automatically selects the appropriate finite difference method based on this flag. See `Derivative1_Order2_NonUniform` in `src/Utilities/FiniteDifference.py` for non-uniform grid implementation.

## Extending the Code

New applications can be implemented by deriving from the appropriate base classes:

### Adding a New Physics Solver

1. **Create a subclass** of `Physics` in `src/Physics/YourSolver.py`
2. **Implement required methods**:
   - `__init__(self, config, param_names)`: Initialize solver parameters
   - `initial_condition(self, param)`: Generate initial conditions for given parameters
   - `solve(self, param, t_grid)`: Solve and return solution trajectory
   - Optional: `threshold`: An optional callable function which accepts three arguments, the current time (t), a FOM frame, and the set of node positions. It should return a 'mask', an 1D array of 0's and 1's whose lenght matches the number of nodes. If present, this is passed to the animate functions. For each frame, only plot nodes whose corresponding mask value (at that time) is 1. 
3. **Register in `Initialize.py`**:
   ```python
   physics_dict = {
       ...
       'YourSolver': YourSolver,
   }
   ```
4. **Create config file** in `examples/YourSolver.yml`


### Adding a New Latent Dynamics Model

1. **Create a subclass** of `LatentDynamics` in `src/LatentDynamics/YourModel.py`
2. **Implement required methods**:
   - `__init__(self, n_z, Uniform_t_Grid)`: Initialize model
   - `calibrate(self, Latent_States, loss_type, t_Grid, input_coefs)`: Compute/update coefficients
   - `simulate(self, Coefs, IC, t_Grid, n_steps)`: Simulate forward in time
3. **Register in `Initialize.py`**:
   ```python
   ld_dict = {
       ...
       'yourmodel': YourModel,
   }
   ```


### Adding a New Sampler (Greedy Sampling Strategy)

Greedy sampling is implemented via `Sampler` classes in `src/Sample/`. A sampler selects the next
parameter point(s) to add to the training set after each training round.

1. **Create a subclass** of `Sampler` in `src/Sample/YourSampler.py`.
2. **Implement required methods**:
   - `__init__(self, config)`: Parse `config['type']` and your sampler-specific settings.
   - `Sample(self, trainer) -> NextStep`: Append the chosen point(s) to `trainer.param_space.train_space`
     and return `NextStep.RunSample`.
   - (Optional) override `Generate_Training_Data(self, trainer)` if you need custom data-generation behavior.
     Most samplers can reuse the base implementation.
3. **Register in `Initialize.py`**:
   ```python
   sampler_dict = {
       ...
       'YourSampler': YourSampler,
   }
   ```
4. **Configure in YAML**:
   ```yaml
   sampler:
     type: YourSampler
     YourSampler:
       # sampler-specific settings
   ```

### Adding a New EncoderDecoder Architecture

1. **Create a subclass** of `EncoderDecoder` which should be placed in a file in `src/EncoderDecoder`
2. **Implement required methods**:
   - `Encode(self, X(1), ... , X(n_IC))`: Encode full-order state to latent space
   - `Decode(self, Z(1), ... , Z(n_IC))`: Decode latent state to full-order space
   - `forward(self, X(1), ... , X(n_IC))`: Encode and then Decode the FOM states.
   - `export()`: Returns a dictionary that can be used to serialize the EncoderDecoder.
3. **Register in `Initialize.py`**:
   ```python
   encoder_decoder_dict = {
       ...
       'your_encoder_decoder': YourEncoderDecoder,
   }
   encoder_decoder_load_dict = {
       ...
       'your_encoder_decoder': load_YourEncoderDecoder,
   }
   ```
4. **Define how to train your architecture**: Import your new EncoderDecoder sub-class in `Trainer.py`. Either add the new class to one of the existing cases (using the pre-selected loss functions) in the `Trainer` class' `train` method, or define a new case to handle your new class.


## Testing and Development

The `Test/` directory contains Jupyter notebooks for testing and validating components:
- `FiniteDifference.ipynb` – Finite difference scheme validation
- `FirstOrderSolvers.ipynb` – First-order ODE solver tests
- `SecondOrderSolvers.ipynb` – Second-order ODE solver tests
- `DeriveFiniteDifference.ipynb` – Derivation of finite difference formulas

## Output and Results

Training produces several outputs:

### Saved Files
- **Checkpoint**: `checkpoint/checkpoint.pt` – EncoderDecoder weights and training state
- **Results**: `results/<physics_type>_loss_by_param.pkl` – Per-epoch loss curves (per training parameter + totals)
- **Figures**: `Figures/*.png` – Latent trajectory plots, error heatmaps
- **Animations**: `Figures/*.mp4` – Solution animations (if generated)

### Gradient clipping

To prevent exploding gradients during training, `BayesianGLaSDI` applies global gradient-norm clipping via `torch.nn.utils.clip_grad_norm_` (threshold: `lasdi.gplasdi.gradient_clip`, default: `15.0`). When clipping activates, a warning is logged.

### Per-parameter loss logging (`*_loss_by_param.pkl`)

During training, `src/GPLaSDI.py` writes a pickle file:

- Path: `results/<physics_type>_loss_by_param.pkl` (where `<physics_type>` is `config['physics']['type']`)
- Type: nested Python dictionaries

Structure:

- `loss_by_param[loss_name][param_tuple] -> {'epochs': [...], 'losses': [...]}`
- `param_tuple` is a tuple of training parameter values (in the same order as the parameter space)
- Each `loss_name` also contains a `'total'` entry, which is the loss summed across all training parameters for that epoch.

Common `loss_name` keys include:
- `recon`, `LD`, `stab`, `rollout_ROM`, `rollout_FOM`, `IC_rollout_ROM`, `IC_rollout_FOM`, and `total`
- For paired autoencoders, additional keys such as `recon_D`, `recon_V`, `consistency_Z`, `consistency_U`, `chain_rule_U`, `chain_rule_Z`, `rollout_*_D/V`, and `IC_rollout_*` are also logged.

Reading the file:

```python
import pickle

with open("results/Burgers_loss_by_param.pkl", "rb") as f:
    loss_by_param = pickle.load(f)

# Total reconstruction loss curve
epochs = loss_by_param["recon"]["total"]["epochs"]
losses = loss_by_param["recon"]["total"]["losses"]

# Per-parameter reconstruction curve (example param tuple)
# loss_by_param["recon"][(0.1, 0.2)] -> {'epochs': [...], 'losses': [...]}
```

For a ready-made parser/plotter, see the repo-root notebook `Analyze_Parameter_Losses.ipynb`, which loads `*_loss_by_param.pkl` and produces per-parameter loss plots.

### Logging
- Real-time logging to console and `output.txt`
- Loss tracking per parameter combination
- GP fitting statistics
- Timing information for performance profiling

## GPU Support

The code automatically detects and uses CUDA if available. To force CPU usage:

```python
# In your config or code
device = 'cpu'
```

GPU training can provide significant speedups, especially for large autoencoders.

## Citation

If you use this code in your research, please cite the relevant LaSDI papers:

[Add citation information here]

## Contributing

Contributions are welcome! Please follow the existing code structure and add tests for new features.

## License

[Add license information here]

## Acknowledgments

This work was performed under the auspices of the U.S. Department of Energy by Lawrence Livermore National Laboratory under Contract DE-AC52-07NA27344 (LLNL-CODE-xxxxxx).
