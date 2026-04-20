# Higher-Order LaSDI

Higher-Order LaSDI provides tools for building reduced-order models (ROMs) from full-order (FOM) simulations using latent-space dynamics identification with Gaussian Process-based greedy sampling. Higher-Order LaSDI is designed to be incredibly flexible. At its core, a reduced-order model (ROM) consists of:
1. A parameterized collection of time series that live in some high-dimensional, FOM space (Physics).
2. An Encoder/Decoder pair which can map elements from the FOM space to a lower dimensional latent space (EncoderDecoder).
3. A parameterized set of latent dynamics to describe how the encodings of a particular time series evolves (LatentDynamics).
4. A procedure to learn an encoder/decoder/latent dynamics given training data (Trainer).
5. A method to greedily pick new training elements (Sampler).
Higher-Order LaSDI implements a modular framework that allows the user to customize each one of these steps. Each one has an associated base class (specified by parentheses in the list) which defines the interface the rest of the code uses when interacting with that portion of the workflow. The library includes a few subclasses for each step, but you can customize any of them by defining a subclass of the corresponding base class. As long as the subclass interface matches what the library expects, it should integrate cleanly. Want to embed a domain-specific inductive bias into the latent dynamics? Define a `LatentDynamics` subclass. Need an EncoderDecoder structure tailored to your data? Define an `EncoderDecoder` subclass. Want to use a custom physics-inspired loss? Implement it in a `Physics` subclass. Higher-Order LaSDI is designed to be flexible, extensible, and modular.

## Key Features

- **Gaussian Process-based Greedy Sampling**: Adaptive parameter space exploration using GP uncertainty quantification
- **Higher-Order Dynamics**: Native support for systems requiring multiple time derivatives
- **Flexible Time Grids**: Handles both uniform and non-uniform temporal discretizations with automatic finite difference scheme selection
- **Multiple Loss Functions**: Reconstruction, latent dynamics, rollout, IC rollout, chain rule, and consistency losses with configurable weights
- **EncoderDecoder Architectures**: Standard autoencoder, paired autoencoder for higher-order systems
- **Rich Activation Functions**: Support for 20+ activation functions including sine, ReLU, tanh, GELU, and more
- **Visualization Tools**: Automated plotting of latent trajectories, error heatmaps, and solution animations
- **Training stability & diagnostics**: Gradient clipping (typically configured under `trainer.<TrainerType>.gradient_clip`) to prevent exploding gradients, and per-parameter loss logging to `results/*_loss_by_param.pkl` for post-hoc analysis

## Getting Started

For a command-line workflow, use `src/Workflow.py` together with a YAML configuration file. Example configurations are provided in `examples/*.yml`. 

### Basic Usage

```bash
python src/Workflow.py --config examples/KleinGordon.yml
```

This command will:
1. Generate or load training data from the physics solver
2. Train the autoencoder and latent dynamics using `Trainer`
3. Perform greedy sampling to adaptively select new training points
4. Evaluate the learned ROM on test data
5. Generate plots and error metrics

### Workflow Components

`Workflow.py` orchestrates the following pipeline:
- **Data Generation**: Calls physics solvers to generate training trajectories
- **Training**: Uses the `Trainer` class to train autoencoders and latent dynamics
- **Greedy Sampling**: Uses a configurable `Sampler` to select new training points (often via Gaussian Processes fit to latent dynamics coefficients)
- **Evaluation**: Computes rollout errors, relative errors, and standard deviations
- **Visualization**: Generates latent trajectory plots, error heatmaps, and solution animations

## Available Examples

The `examples/` directory contains configuration files for various physics problems:

| Example | Physics Type | Order | Grid Type |
|---------|-------------|-------|-----------|
| `Explicit.yml` | Custom explicit solver | 1st | Uniform |
| `Burgers.yml` | Burgers equation (second-order formulation) | 2nd | Uniform |
| `Burgers2D.yml` | 2D Burgers equation | 1st | Uniform |
| `Thermal.yml` | Thermal diffusion (HDF5) | 1st | Uniform |
| `Advection.yml` | Advection equation (PyMFEM) | 1st | Uniform |
| `WaveEquation.yml` | Wave equation (PyMFEM) | 2nd | Uniform |
| `KleinGordon.yml` | Klein-Gordon equation (PyMFEM) | 2nd | Uniform |
| `Telegraphers.yml` | Telegraphers equation (PyMFEM) | 2nd | Uniform |
| `NonlinearElasticity.yml` | Nonlinear elasticity (PyMFEM) | 2nd | Uniform |

**Note**: PyMFEM examples require PyMFEM installation (see below).

## Repository Layout

### EncoderDecoders and $n_{IC}$

Higher-Order LaSDI supports higher-order latent dynamics (via multiple time derivatives) and multi-stage training (via multiple decoders).

The key hyperparameters for an `EncoderDecoder` are:
- $n_{IC}$: the number of initial-condition components used by the latent dynamics. Practically, this means we treat the training data as a tuple containing the FOM frame and its first $n_{IC}-1$ time derivatives:
  \[
    \left(u,\ \dot{u},\ \ddot{u},\ \ldots,\ u^{(n_{IC}-1)}\right).
  \]
- $n_z$ (sometimes written $n_{latent}$): latent-space dimension, so each latent component lives in $\mathbb{R}^{n_z}$.
- $n_{Decoders}$: number of decoder *stages* used to enable multi-stage training (mLaSDI-style).

**Interface.** An `EncoderDecoder` is a `torch.nn.Module` with the following core methods:
- `Encode(*Xs) -> tuple[torch.Tensor, ...]`  
  Takes `n_IC` tensors `Xs = (X0, X1, ..., X_{n_IC-1})` (the FOM frame and its derivatives) and returns `n_IC` latent tensors `Zs = (Z0, Z1, ..., Z_{n_IC-1})`, each typically shaped `(batch, n_z)`.
- `Eval_Decoder(i_Decoder, *Zs) -> tuple[torch.Tensor, ...]`  
  Evaluates the *i-th* decoder stage on the latent tuple, returning a tuple of `n_IC` decoded tensors in FOM space.
- `Decode(*Zs)` (implemented in the base class)  
  Produces the decoded tuple by taking a **weighted sum** over the active decoder stages. Concretely, for each component index `j`:
  \[
    X_j \;=\; \sum_{d\ \text{active}} w_{j,d}\, \mathrm{Eval\_Decoder}(d, Z_0,\ldots,Z_{n_{IC}-1})_j.
  \]
  The weights $w_{j,d}$ live in `Decoder_Weight[j, d]`, and which decoders participate is controlled by `Decoder_Active[d]`. These settings are included in `export()`/`load()` so multi-stage state can be checkpointed and restored.

If you set `n_Decoders = 1` (the default/common case), this reduces to the original single-decoder behavior: `Decode(*Zs)` is exactly `Eval_Decoder(0, *Zs)` (up to the default unit weight).

In general, every Physics, EncoderDecoder, and Trainer object expects a specific $n_{IC}$ value. These must match for the code to work. Thus, for instance, if your Physics sub-class defines time series consisting of some time series and its first time derivative, then $n_{IC} = 2$ and you must select a Trainer and LatentDynamics model that also use $n_{IC} = 2$. 

### Core Components

- **`src/Workflow.py`** – Main command-line driver that loads configuration files, initializes components, and runs the training pipeline
- **`src/Trainer/`** - The class which defines how the encoder/decoder/latent dynamics train using the available data
    - `Trainer.py` – Base `Trainer` class: normalization helpers, checkpointing, loss logging, timing, and round-based training orchestration
    - `First_Order_Rollout.py` – `Trainer` subclass for first-order systems (`n_IC = 1`)
    - `Second_Order_Rollout.py` – `Trainer` subclass for second-order systems (`n_IC = 2`)
    - `Second_Order_Noise.py` – Similar to `Second_Order_Rollout` but can add noise to the data
    - `Second_Order_Noise_Weak.py` – Similar to `Second_Order_Noise`, but intended for weak-form latent dynamics
- **`src/Initialize.py`** – Factory functions for initializing trainers, EncoderDecoders, physics solvers, and latent dynamics from config files. You must register all new sub-classes in this file.
- **`src/EncoderDecoder`** – Neural network architectures:
  - `EncoderDecoder.py`: Base `EncoderDecoder` class.
  - `MLP.py`: Flexible MLP with customizable activations
  - `Autoencoder.py`: Standard autoencoder for first-order systems
  - `Autoencoder_Pair.py`: Paired autoencoder for higher-order systems (encodes multiple derivatives)
  - `CNN_3D_Autoencoder.py`: 3D convolutional autoencoder variant
- **`src/ParameterSpace.py`** – Parameter space management, grid generation, and train/test split utilities
- **`src/SolveROMs.py`** – ROM simulation functions:
  - `average_rom()`: Simulate using GP mean predictions
  - `sample_roms()`: Simulate using samples from GP posteriors
  - Error computation and uncertainty quantification
- **`src/Sample/`** – Greedy sampling logic:
  - `Sampler.py`: Base `Sampler` class (selects the next training parameter during greedy sampling)
  - `FOM_Variance.py`: Selects next point by maximizing predictive variance in decoded (FOM) space
  - `FOM_Rollout.py`: Selects next point by maximizing rollout error against the true FOM (intrusive)
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

### Trainer Settings (`trainer`)
- `trainer.type` selects the concrete `Trainer` subclass to use (e.g., `First_Order_Rollout`, `Second_Order_Rollout`, `Second_Order_Noise`, `Second_Order_Noise_Weak`).
- Round scheduling / greedy sampling limits:
  - `trainer.n_iter`, `trainer.max_iter`, `trainer.max_greedy_iter`
- Normalization:
  - `trainer.normalize` (if enabled, the library computes global mean/std and normalizes train/test data)
- Device placement:
  - `trainer.device` (optional; `"cpu"` by default)
- Subclass-specific settings live under `trainer.<TypeName>`. Example (`Second_Order_Rollout`):
  - learning rate + stability: `lr`, `gradient_clip`, `warmup_epochs`
  - rollout curriculum: `p_rollout_init`, `rollout_update_freq`, `dp_per_update`, `max_p_rollout`
  - rollout sampling: `n_rollouts`
  - IC rollout curriculum: `p_IC_rollout_init`, `IC_rollout_update_freq`, `IC_dp_per_update`, `max_p_IC_rollout`
  - loss weights / types:
    - weights: `loss_weights` (e.g. `recon`, `LD`, `rollout`, `IC_rollout`, `stab`, `coef`, plus higher-order losses)
    - types: `loss_types` (`"MSE"` or `"MAE"`)

### Sampler Settings (`sampler`)
- `sampler.type` selects a sampler implementation (e.g., `FOM_Variance` or `FOM_Rollout`).
- Each sampler has its own settings block under `sampler.<TypeName>`. For example:
  - `sampler.FOM_Variance.n_samples`
  - `sampler.FOM_Rollout.n_samples`, `sampler.FOM_Rollout.normalized_FOM`, `sampler.FOM_Rollout.error_normalization`

### Workflow Settings (`workflow`)
- Restart capability (load from checkpoint)
- Plotting options

### Parameter Space (`parameter_space`)
- Parameter definitions (uniform, list, or file-based)
- Test space configuration (grid, random, or file-based)

### EncoderDecoder Architecture (`EncoderDecoder`)
- EncoderDecoder type: `ae`/`autoencoder` (Autoencoder), `pair`/`autoencoder_pair` (Autoencoder_Pair), or `cnn_3d`/`cnn_3d_ae`/`cnn_3d_autoencoder` (CNN_3D_Autoencoder).
- Hidden layer widths
- Activation functions
- Latent dimension
- Number of decoder stages: `n_Decoders`

### Latent Dynamics (`latent_dynamics`)
- Type: `sindy`, `spring`, or `switch sindy`.
- Stability regularization (stability penalty)

### Physics (`physics`)
- Physics type (must match a key in `physics_dict`)
- Physics-specific parameters (grid sizes, time steps, domain bounds, etc.)

See **Extending the Code** below for details on adding new Physics / LatentDynamics / Sampler /
EncoderDecoder implementations and registering them in `src/Initialize.py`.

## Extending: Adding a new Trainer subclass

The training loop is split into a base class (`src/Trainer/Trainer.py`) and concrete subclasses
that implement one training strategy per greedy-sampling round.

### 1) Create a new subclass

Create a new file under `src/Trainer/`, for example `src/Trainer/MyTrainer.py`, and implement:

- `__init__(physics, encoder_decoder, latent_dynamics, param_space, config)`
- `Iterate(start_iter, end_iter)` (the actual per-epoch training logic)

Follow the existing trainers (`First_Order_Rollout`, `Second_Order_Rollout`, `Second_Order_Noise`, `Second_Order_Noise_Weak`) as templates. In particular, your
`Iterate(...)` method should:

- Log losses via `_store_loss_by_param(...)` / `_store_total_loss(...)`
- Call `_Save_Checkpoint(...)` when a new best model is found (so `train()` can restore it)

### 2) Register the trainer in `Initialize.py`

In `src/Initialize.py`:

1. Import your class:

```python
from MyTrainer import MyTrainer
```

2. Add it to `trainer_dict`:

```python
trainer_dict = {
    'First_Order_Rollout'      : First_Order_Rollout,
    'Second_Order_Rollout'     : Second_Order_Rollout,
    'Second_Order_Noise'       : Second_Order_Noise,
    'Second_Order_Noise_Weak'  : Second_Order_Noise_Weak,
    'MyTrainer'  : MyTrainer,
}
```

### 3) Add YAML config entries

In your YAML file:

```yaml
trainer:
  type: MyTrainer
  n_iter: 2500
  max_iter: 20000
  max_greedy_iter: 20000
  normalize: false

  MyTrainer:
    # your subclass-specific settings here
```

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

These packages are listed in the `requirements.txt` file.


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

# PyTorch (pick the right wheel for your platform/CUDA)
python -m pip install torch==2.5.1
```

Add optional dependencies to the same venv:

```bash
# HDF5 data loading (Thermal example)
python -m pip install h5py==3.14.0

# Jupyter for error diagnostics
python -m pip install jupyter==1.0.0

# PyMFEM: This one is tricky. Go to the PyMFEM build steps below.
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
- hdf5 (the system HDF5 library, not a Python package; 1.14.5)
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

PyMFEM (via `numba-scipy`/SciPy constraints) does not work on the latest Python versions. For this
guide, use Python 3.11.

If you aren't already, set up a new virtual environment:

```bash
python3.11 -m venv venv_mfem
source venv_mfem/bin/activate

python -m pip install --upgrade pip setuptools wheel
# Install Higher-Order-LaSDI Python dependencies into this venv
python -m pip install -r requirements.txt

python -m pip install torch==2.5.1
```

PyMFEM should be installed into the same `Higher-Order-LaSDI/venv_mfem`.


### 2. Clone and Checkout PyMFEM

```bash
cd ..
git clone https://github.com/mfem/PyMFEM.git
cd ./PyMFEM
git checkout v_4.7.0.1
```

### 3. Install Dependencies

Install all dependencies using the requirements file:

```bash
python -m pip install -r requirements.txt
```


### 4. Fix CMake Version

PyMFEM v_4.7.0.1 requires cmake < 4.0:

```bash
python -m pip uninstall cmake
python -m pip install cmake==3.31.10
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

**On LC:**

Note that you may need to use a different version for `intel-classic` and `openmpi`.

```bash
module --force purge
module load StdEnv
module load intel-classic/2021.6.0
module load openmpi/4.1.2

which mpicc
mpicc --showme:link 2>/dev/null || mpicc -show
```

### 6. Install mpi4py

**On most systems:**
```bash
pip install mpi4py==4.0.3
```

**On LC:**

On some LC systems, the Python toolchain injects an Anaconda-provided MPI (`libmpi.so.12`) into the
link/runtime search path. If `mpi4py` links against that, but you load OpenMPI at runtime, imports
can fail with missing OpenMPI symbols (e.g. `undefined symbol: ompi_mpi_real`).

The procedure below forces `mpi4py` to link against the OpenMPI module you loaded above.

First, install Cython and remove any existing `mpi4py`:
```bash
source ./venv_mfem/bin/activate

python -m pip install --upgrade "cython>=3.0"
pip uninstall -y mpi4py
```

Next, download the `mpi4py` source distribution:

```bash
cd /tmp
rm -rf mpi4py-src && mkdir mpi4py-src && cd mpi4py-src
python -m pip download --no-deps --no-binary=mpi4py mpi4py==4.0.3
tar -xf mpi4py-4.0.3.tar.gz
cd mpi4py-4.0.3
```

Create an `mpi.cfg` that points to your loaded OpenMPI `mpicc/mpicxx` and forces linkage against
OpenMPI's `libmpi.so.40`:

```bash
OMPI_PREFIX="$(dirname "$(dirname "$(which mpicc)")")"
OMPI_LIB="$OMPI_PREFIX/lib"

cat > mpi.cfg <<EOF
[mpi]
mpicc  = $OMPI_PREFIX/bin/mpicc
mpicxx = $OMPI_PREFIX/bin/mpicxx

# key: do NOT link -lmpi
libraries =

# ensure OpenMPI runtime path is present
runtime_library_dirs = $OMPI_LIB

# force the exact OpenMPI SONAME
extra_objects = $OMPI_LIB/libmpi.so.40
EOF
```

Install from source using that config:
```bash
# make sure build tooling is up to date inside the venv
python -m pip install -U pip setuptools wheel

# tell mpi4py to use your mpi.cfg
export MPI4PY_BUILD_MPICFG="$PWD/mpi.cfg"

# build/install
python -m pip install --no-build-isolation .
```

Verify the resulting `mpi4py` extension is linked only against OpenMPI (`libmpi.so.40`):

```bash
MPI_SO=$(python -c "import site,glob,os; sp=site.getsitepackages()[0]; print(glob.glob(os.path.join(sp,'mpi4py','MPI*.so'))[0])")
echo "$MPI_SO"

readelf -d "$MPI_SO" | egrep 'NEEDED.*libmpi|RPATH|RUNPATH'
ldd "$MPI_SO" | egrep 'libmpi|open-rte|open-pal|anaconda|not found'
```

The `readelf` output should include:

```
... (NEEDED) Shared library: [libmpi.so.40]
```

If you also see `libmpi.so.12` under `NEEDED` (and `ldd` resolves it from an Anaconda path),
imports may still fail. As a last resort, you can remove that unwanted dependency with `patchelf`:

```bash
cp -v "$MPI_SO" "${MPI_SO}.bak"
patchelf --remove-needed libmpi.so.12 "$MPI_SO"
readelf -d "$MPI_SO" | egrep 'NEEDED.*libmpi|RPATH|RUNPATH'
```

Now run a test import:
```bash
python -c "from mpi4py import MPI; print(MPI.Get_library_version())"
```

This should print something like the following: 

`Open MPI v4.1.2, package: Open MPI sly1@rzwhippet7 Distribution, ident: 4.1.2, repo rev: v4.1.2, Nov 24, 2021`

If so, mpi4py is now installed and linked correctly. Change back to the PyMFEM directory:

```bash
cd <path to PyMFEM directory>
```


**Suppressing warnings on LC:**

When running on some nodes, you may see OpenMPI warnings about OpenFabrics/InfiniBand initialization.
These are often benign for single-process runs. One way to silence the warning is:

```bash
export OMPI_MCA_btl_openib_warn_no_device_params_found=0
export OMPI_MCA_btl_openib_allow_ib=0
```

Avoid hard-coding `OMPI_MCA_btl="self,tcp"` unless you know your system provides the `btl:tcp`
component; on some systems it can cause `MPI_Init` failures on compute nodes.


### 7. Build PyMFEM

```bash
python setup.py install -v --with-parallel --with-gslib \
  --CC=gcc --CXX=g++ --MPICC=mpicc --MPICXX=mpic++ --with-lapack
```
**Note**: On some systems you may need to adjust compiler names (e.g., `gcc-11`, `g++-11`).

If this works, you have now installed PyMFEM!

### LC note: “Do I need to redo this every login?”

- The **venv is persistent**: once `venv_mfem` (and the patched `mpi4py` inside it) is created, it
  will keep working across logins *as long as you don't reinstall/upgrade `mpi4py`* inside that venv.
- Your **module environment is not persistent**: you generally need to `module load intel-classic`
  and `module load openmpi` again in each new shell / batch job.
- Any **environment variables** (e.g. `OMPI_MCA_btl`) must be set each session/job if you want them.
  (e.g. `OMPI_MCA_btl_openib_allow_ib`, `OMPI_MCA_btl_openib_warn_no_device_params_found`).

A simple solution is to create a small helper script (e.g. `env_lc.sh` in Higher-Order-LaSDI repo) with the following contents:


```bash
# env_lc.sh (example)
module --force purge
module load StdEnv
module load intel-classic/2021.6.0
module load openmpi/4.1.2
source ./venv_mfem/bin/activate
export OMPI_MCA_btl_openib_warn_no_device_params_found=0
export OMPI_MCA_btl_openib_allow_ib=0
```

Before running LaSDI, `source` the script:
```bash
source env_lc.sh
```


## Non-Uniform Time Grids

Physics objects expose a `Uniform_t_Grid` attribute which determines derivative computation:
- When `True`: Uses higher-order finite difference schemes (Order 4 when available)
- When `False`: Uses non-uniform grid schemes (currently Order 2)

The `Trainer` class automatically selects the appropriate finite difference method based on this flag. See `Derivative1_Order2_NonUniform` in `src/Utilities/FiniteDifference.py` for non-uniform grid implementation.

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

1. **Create a subclass** of `EncoderDecoder` in `src/EncoderDecoder/YourEncoderDecoder.py`.
2. **Implement the required interface** (minimal set):
   - `__init__(self, Frame_Shape: list[int], config: dict)`:
     - Parse `config['type']` and your type-specific sub-config.
     - Call `super().__init__(n_IC=..., n_z=..., n_Decoders=..., config=config)`.
     - Construct the encoder module(s) and the per-stage decoder module(s) (e.g., a `torch.nn.ModuleList` of length `n_Decoders`).
   - `Encode(self, *Xs) -> tuple[torch.Tensor, ...]`:
     - Accept `n_IC` FOM tensors (frame + derivatives) and return `n_IC` latent tensors.
   - `Eval_Decoder(self, i_Decoder: int, *Zs) -> tuple[torch.Tensor, ...]`:
     - Run the requested decoder stage `i_Decoder` and return `n_IC` decoded tensors in FOM space.
   - `export(self) -> dict`:
     - Return a dict that includes `super().export()` (so `Decoder_Active`/`Decoder_Weight` are checkpointed) plus your module state dicts and any architecture metadata needed to reconstruct the model.
   - A corresponding `load_YourEncoderDecoder(dict_) -> YourEncoderDecoder` free function:
     - Instantiate the model (using saved `Frame_Shape`/`config`/architecture metadata),
     - call `model.load(dict_['EncoderDecoder dict'])` to restore `Decoder_Active`/`Decoder_Weight`,
     - load module parameters from state dicts.

3. **Optional overrides** (only if you need custom behavior):
   - `Decode(self, *Zs)` if you want a non-linear combination of decoder stages (the base class implements a weighted sum over active stages).
   - `forward(self, *Xs)` (the base class calls `Encode` then `Decode`).
   - `latent_initial_conditions(...)` if you need custom IC handling (the base class provides a default implementation).
   - `Set_Decoder_Active` / `Set_Decoder_Weight` if you want custom validation or semantics.
4. **Register in `Initialize.py`**:
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
5. **Define how to train your architecture**:
   - If your model follows the standard `Encode` / `Decode` contract (and your losses don’t rely on architecture-specific internals), you may be able to reuse an existing trainer (e.g., `First_Order_Rollout` or `Second_Order_Rollout`).
   - If you need architecture-specific losses or training logic, implement a new `Trainer` subclass and register it in `trainer_dict` in `src/Initialize.py`.


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

To prevent exploding gradients during training, Trainer subclasses apply global gradient-norm clipping via `torch.nn.utils.clip_grad_norm_`. The threshold is typically configured under `trainer.<TrainerType>.gradient_clip` (default varies by trainer; see the corresponding trainer subclass). When clipping activates, a warning is logged.

### Per-parameter loss logging (`*_loss_by_param.pkl`)

During training, the trainer writes a pickle file (see `src/Trainer/Trainer.py`):

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
