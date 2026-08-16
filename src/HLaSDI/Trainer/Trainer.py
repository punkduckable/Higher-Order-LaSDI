# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

import  os;

import  json;
import  logging;
from    typing                              import Any;

import  torch;
import  numpy;

from    HLaSDI.EncoderDecoder              import  EncoderDecoder;
from    HLaSDI.Utilities.Timing            import  Timer;
from    HLaSDI.ParameterSpace              import  ParameterSpace;
from    HLaSDI.Physics                     import  Physics;
from    HLaSDI.LatentDynamics              import  LatentDynamics, InterpolatableLatentDynamics;
from    HLaSDI.Schemas                     import  BaseTrainerConfig;

# Setup Logger
LOGGER : logging.Logger = logging.getLogger(__name__);

# Should we profile a run of Iterate?
PROFILE_ITERATE : bool  = False
PROFILE_WAIT    : int   = 10        # Do nothing for this many epochs   
PROFILE_WARMUP  : int   = 1         # Run machinery, but discard results
PROFILE_ACTIVE  : int   = 10        # Run/log profiler stuff for this many epochs
PROFILE_REPEAT  : int   = 1         # Repeat this schedule how many times?


# -------------------------------------------------------------------------------------------------
# Trainer Base class
# -------------------------------------------------------------------------------------------------

class Trainer:
    r"""
    Base interface and shared state for HLaSDI training algorithms.

    A `Trainer` coordinates the learned parts of the reduced-order model: it owns the training and
    testing trajectories, the `Physics`, `EncoderDecoder`, `LatentDynamics`, and `ParameterSpace`
    objects, global normalization statistics, checkpointing, loss logging, and timing.  The base
    `train()` method runs one round of optimization by calling a subclass `Iterate(...)` method,
    then restores the encoder/decoder and latent-dynamics coefficients from the best checkpoint of
    that round so subsequent greedy sampling uses the best available ROM state.

    Class/instance variables
    ------------------------
    U_Train : list[list[torch.Tensor]]
        Training trajectories.  The outer index selects a parameter point; the inner index selects
        one of the `n_IC` state/derivative components; each tensor has a leading time dimension.
        If `noise_ratio > 0`, these trajectories are corrupted in-place before each training
        round. If normalization is enabled, noise is added to the normalized training data.
    U_Train_Clean : list[list[torch.Tensor]]
        Deep copy of `U_Train` before noise is applied. This is the authoritative clean backup
        used to re-sample noisy training data each round and when new greedy samples are added.
    noise_ratio : float
        Ratio of Gaussian noise standard deviation to signal RMS. Defaults to `0.0` (disabled).
        The first frame of each trajectory component is restored from `U_Train_Clean` after noise
        injection, so initial conditions remain exact.
    t_Train : list[torch.Tensor]
        Time grids corresponding to `U_Train`.
    U_Test : list[list[torch.Tensor]]
        Testing trajectories with the same nested structure as `U_Train`.
    t_Test : list[torch.Tensor]
        Time grids corresponding to `U_Test`.
    n_IC : int
        Number of state/derivative components expected by the physics, encoder/decoder, trainer,
        and latent dynamics.  The base initializer checks that these all agree.
    n_iter : int
        Maximum number of optimizer iterations performed in one training round.
    max_iter : int
        Global iteration limit for training.
    max_greedy_iter : int
        Global iteration limit for greedy sampling rounds.
    normalize : bool
        Whether generated FOM trajectories are normalized before training.
    config : dict
        The `trainer` configuration dictionary.
    timer : Timer
        Timing utility used by subclasses to record loss and backpropagation costs.
    device : str
        Device used for training (`"cpu"`, `"cuda:..."`, or `"mps"`).
    physics : Physics
        Full-order problem used to generate trajectories and initial conditions.
    encoder_decoder : EncoderDecoder
        Neural map between FOM space and latent space.
    latent_dynamics : LatentDynamics
        Parameterized latent ODE model whose coefficients are optimized during training.
    param_space : ParameterSpace
        Container for current train/test parameter sets used by training and greedy sampling.
    data_mean, data_std : list[torch.Tensor] | None
        Per-IC scalar normalization statistics when normalization is enabled.

    Subclassing
    -----------
    To implement a training strategy, subclass `Trainer`, call `super().__init__(...)`, parse any
    subclass-specific configuration, and implement:

    - `Iterate(start_iter, end_iter)`: perform optimizer steps for the requested global iteration
      range, compute reconstruction/latent/rollout/etc. losses appropriate to the strategy, update
      encoder/decoder/latent_dynamics parameters, record timing and per-parameter
      losses, and call `_Save_Checkpoint(...)` whenever a new best model for the round is found.

    Subclasses commonly use `_optimizer_parameters()` to build optimizers over both neural-network
    parameters and LD-owned coefficient tensors.  They may extend `export()` and `load()` for
    additional state, but should preserve the base training data, normalization, checkpoint, and
    iteration bookkeeping.
    """
    # An n_Train element list. The i'th element is is an n_IC element list whose j'th element is a
    # numpy ndarray of shape (n_t(i), Frame_Shape) holding a sequence of samples of the j'th 
    # derivative of the FOM solution when we use the i'th combination of training values. 
    # NOTE: these are initialized as instance variables in __init__ (do not share across instances).
    U_Train : list[list[torch.Tensor]];

    # A deep-copy of U_Train without any noise. If noise_ratio > 0, U_Train is re-noised from this
    # clean backup before each training round. It has the same nested shape as U_Train.
    U_Train_Clean : list[list[torch.Tensor]];

    # How much noise we add to the data.
    noise_ratio : float;

    # An n_Train element list whose i'th element is a torch.Tensor of shape (n_t(i)) whose j'th
    # element holds the time value for the j'th frame when we use the i'th combination of training 
    # parameters.
    t_Train : list[torch.Tensor];
    
    # Same as U_Test, but used for the test set.
    U_Test  : list[list[torch.Tensor]];

    # An n_Test element list whose i'th element is a torch.Tensor of shape (n_t(i)) whose j'th
    # element holds the time value for the j'th frame when we use the i'th combination of testing 
    # parameters.
    t_Test  : list[torch.Tensor];

    # number of IC's in the FOM solution.
    n_IC  : int;

    # Number of iterations per round of training
    n_iter : int;
    # We stop training if restart_iter goes above this number. 
    max_iter : int;

    # We stop performing greedy sampling if restart_iter goes above this number.
    max_greedy_iter : int;
    
    # If true, the Sampler will normalize the training data before storing it in this 
    # object. See Sampler/Sampler.py for details.
    normalize : bool;

    # The trainer configuration file.
    config : dict;

    # A timer object that Iterate should use to track how long each loss takes to compute.
    timer : Timer;

    # The trainer's device
    device : str;



    def __init__(   self, 
                    n_IC               : int, 
                    physics            : Physics, 
                    encoder_decoder    : EncoderDecoder, 
                    latent_dynamics    : LatentDynamics, 
                    param_space        : ParameterSpace, 
                    trainer_config     : BaseTrainerConfig):
        """
        Abstract base class that defines how each round of training proceeds (the loss functions, 
        and optimizer).

        In the HLaSDI framework, a ROM consists of an EncoderDecoder model and a LatentDynamics 
        object (acting as the Encoder/Decoder and Latent Dynamics portions of the ROM, respectively). 
        These are jointly trained via a Trainer object using data from a Physics object. The 
        LatentDynamics object holds the learnedLatentDynamics coefficients for the training set. 
        A Sampler object determines how the model picks which testing example to add to the 
        training set after each round of training.

        The trainer essentially defines how everything gets trained. It should do this by 
        initializing an optimizer on the EncoderDecoder parameters and trainable coefficients 
        in the LatentDynamics object (fetched via LatentDynamics.trainable_tensors). It 
        should train these parameters via a sequence of epochs. During each epoch, the Trainer 
        should evaluate a number of loss functions, add them together, then back-prop through the 
        loss to get the derivative of the loss with respect to each EncoderDecoder parameter and 
        latent dynamics coefficient, then use these derivatives to update the parameters and 
        coefficients. All of this is implemented in the sub-class defined "Iterate" method (which 
        is driven via the base class' Train method).

        The trainer also defines model checkpointing (via the _Save_Checkpoint method which 
        Iterate should call each time it finds a new best model).
        
        Trainer also control data normalization normalization; the base class defines several 
        methods for normalizing and de-normalizign data (set_normalization_stats_from_training, 
        set_normalization_stats_from_test, normalize_tensor, denormalize_tensor, 
        denormalize_np, denormalize_np, scale_std_np, and normalize_U_inplace); see 
        each one and their doc strings for details). Normalization generally works by re-centering
        and re-scaling training data before it is fed into the EncoderDecoder; this dramatically 
        improves EncoderDecoder performance (mostly because ML models tend to work best when their 
        data has 0 mean and unit variance), but creates extra book-keeping challenges. In 
        particular, with normalization, EncoderDecoder object natively predict normalized values 
        which need to be de-normalized before their predictions can be evaluated.

        Finally, Trainer objects generally track timing data (time spent computing each loss; this
        is managed by the timer attribute) and track losses (by training parameter!).

        In addition to defining how training works, a `Trainer` instance owns the state of a 
        Higher-Order-LaSDI run:

        - The training and testing datasets (`U_Train`, `t_Train`, `U_Test`, `t_Test`)
        - Optional global normalization statistics (`data_mean`, `data_std`)
        - The model objects (`physics`, `encoder_decoder`, `latent_dynamics`)
        - Bookkeeping for iterative training + greedy sampling (`restart_iter`, `n_iter`, etc.)


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        physics : Physics
            Encodes the FOM. It allows us to fetch the FOM solution and/or initial conditions 
            for a particular combination of parameters. We use this object to generate FOM 
            solutions which we then use to train the encoder_decoder and latent dynamics.
         
        encoder_decoder : EncoderDecoder
            use to compress the FOM state to a reduced, latent state.

        latent_dynamics : LatentDynamics
            A LatentDynamics object which describes how we specify the dynamics in the 
            EncoderDecoder's latent space.

        param_space: ParameterSpace
            holds the set of testing and training parameters. 

        trainer_config : dict
            The `trainer` sub-dictionary of the YAML config. The base class expects:

                - type
                - n_iter
                - max_iter
                - max_greedy_iter
                - normalize

            Optional keys:
                - device   (defaults to "cpu")
                - noise_ratio (defaults to 0.0; Gaussian noise std / signal RMS)

        
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Nothing!
        """
        
        # Checks.
        assert isinstance(n_IC, int) and n_IC > 0, "n_IC must be a positive int";
        assert latent_dynamics.n_IC         ==  n_IC, "latent_dynamics.n_IC = %d, n_IC = %d" % (latent_dynamics.n_IC, n_IC);
        assert encoder_decoder.n_IC         ==  n_IC, "encoder_decoder.n_IC = %d, n_IC = %d" % (encoder_decoder.n_IC, n_IC);
        assert physics.n_IC                 ==  n_IC, "physics.n_IC = %d, n_IC = %d" % (physics.n_IC, n_IC);
        self.n_IC                           =   n_IC;

        # Serialize stuff. 
        self.config                         = trainer_config;
        self.physics                        = physics;
        self.encoder_decoder                = encoder_decoder;
        self.latent_dynamics                = latent_dynamics;
        self.param_space                    = param_space;

        # Initialize datasets (instance variables; do NOT share across instances).
        self.U_Train                        = [];
        self.U_Train_Clean                  = [];
        self.t_Train                        = [];
        self.U_Test                         = [];
        self.t_Test                         = [];
        
        # Initialize a timer object. We will use this while training.
        self.timer                          = Timer();

        assert isinstance(trainer_config, BaseTrainerConfig), "trainer_config must be a BaseTrainerConfig, got %s" % str(type(trainer_config));

        # Fetch trainer class information.
        self.n_iter                 : int   = trainer_config.n_iter;             # Number of iterations for one train and greedy sampling
        self.max_iter               : int   = trainer_config.max_iter;           # We stop training if restart_iter goes above this number.
        self.max_greedy_iter        : int   = trainer_config.max_greedy_iter;    # We stop performing greedy sampling if restart_iter goes above this number.
        device                      : str   = trainer_config.device;  # The device we want to map the trainer and its attributes to (and where we will perform training).
        self.noise_ratio            : float = float(trainer_config.noise_ratio);
        assert self.noise_ratio >= 0.0, "trainer.noise_ratio must be non-negative";
        if self.noise_ratio > 0.0:
            LOGGER.info("Noise injection enabled: noise_ratio = %f" % self.noise_ratio);
        else:
            LOGGER.info("Noise injection disabled (noise_ratio = 0.0)");

        # Optional normalization (training-only stats).
        # If enabled, we compute a single mean/std across ALL training trajectories (per IC),
        # then normalize both training + testing trajectories using these values.
        self.normalize              : bool                      = trainer_config.normalize;
        self.data_mean              : list[torch.Tensor] | None = None;   # per-IC scalar tensors (CPU)
        self.data_std               : list[torch.Tensor] | None = None;   # per-IC scalar tensors (CPU)

        # Set the device to train on. We default to cpu.
        if (device.startswith('cuda')):
            assert(torch.cuda.is_available());
            self.device = device;
        elif (device == 'mps'):
            assert(torch.backends.mps.is_available());
            self.device = device;
        else:
            self.device = 'cpu';

        # Set paths for checkpointing/results.
        src_dir     = os.path.dirname(os.path.abspath(__file__));                       # .../Higher-Order-LaSDI/src/Trainer
        project_dir = os.path.abspath(os.path.join(src_dir, os.pardir, os.pardir));     # .../Higher-Order-LaSDI
        self.path_checkpoint    : str = os.path.join(project_dir, "checkpoint");
        self.path_results       : str = os.path.join(project_dir, "results");

        # Make sure the checkpoints and results directories exist.
        from pathlib import Path;
        Path(self.path_checkpoint).mkdir(   parents = True, exist_ok = True);
        Path(self.path_results).mkdir(      parents = True, exist_ok = True);
        LOGGER.info("Checkpoint directory: %s" % self.path_checkpoint);
        LOGGER.info("Results directory: %s" % self.path_results);

        # Build a loss cache; this will be a list whose entries are tuples of the form:
        #   (loss_name, param_tuple or "total", loss_value)
        # The _flush_loss_cache method post-processes/serializes the contents of this list.
        self._loss_cache                    = [];

        # Figure out where we will save cached losses.
        base_filename           : str       = self.physics.config.type;
        self.loss_by_param_path : str       = os.path.join(self.path_results, base_filename + '_loss_by_param.jsonl');
        
        # Final setup.
        self.restart_iter       = 0;                # Global iteration index at the start of the next training round
        self.best_epoch         = None;             # Optional: subclasses may set this when checkpointing

        # All done!
        return;




    # ---------------------------------------------------------------------------------------------
    # Methods to add noise
    # ---------------------------------------------------------------------------------------------

    @staticmethod
    def addNoise(x : torch.Tensor, noise_ratio : float) -> torch.Tensor:
        """
        Add Gaussian noise to a tensor, scaled by the signal's RMS power.

        sigma = noise_ratio * sqrt(mean(x^2))
        noise ~ N(0, sigma)

        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        x : torch.Tensor
            The clean signal to corrupt.

        noise_ratio : float
            The ratio of the noise standard deviation to the signal RMS.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        x_noisy : torch.Tensor
            The corrupted signal (same shape and dtype as x).
        """

        if noise_ratio <= 0.0:
            return x;
        
        signal_power    : float         = float(torch.sqrt(torch.mean(x**2)).item());
        sigma           : float         = noise_ratio * signal_power;
        noise           : torch.Tensor  = torch.normal(mean = 0.0, std = sigma, size = x.shape).to(dtype = x.dtype, device = x.device);
        return x + noise;



    def apply_noise_to_U_Train(self) -> None:
        """
        Apply Gaussian noise to the current training data (self.U_Train).

        Before corrupting the data, a deep copy of the clean training data is saved in
        self.U_Train_Clean so that noise-free references remain available. Repeated calls
        re-sample noise from this clean backup rather than adding noise on top of existing noisy
        data. Note that the first frame (IC) of every trajectory component is restored from the
        clean data because we assume perfect initial conditions.
        """

        if self.noise_ratio <= 0.0:
            return;

        LOGGER.info("Applying noise (ratio = %f) to %d training trajectories" % (self.noise_ratio, len(self.U_Train)));

        # Deep-copy clean data before corruption. Notably, if U_Train is longer than
        # U_Train_Clean, then the extra elements of U_Train were added by the sampler and do not
        # yet have any noise; we need to back them up in U_Train_Clean.
        for i in range( len(self.U_Train_Clean), len(self.U_Train) ):
            self.U_Train_Clean.append([u.clone() for u in self.U_Train[i]]);

        # Corrupt each trajectory, each IC derivative, but preserve the first frame (IC).
        for i in range(len(self.U_Train)):
            for j in range(len(self.U_Train[i])):
                clean_IC    : torch.Tensor  = self.U_Train_Clean[i][j][0:1, ...].clone();     # shape (1, ...)
                noisy_data  : torch.Tensor  = self.addNoise(self.U_Train_Clean[i][j].clone(), self.noise_ratio);
                noisy_data[0:1, ...]        = clean_IC;                                  # restore perfect IC
                self.U_Train[i][j]          = noisy_data;
                
                LOGGER.debug("  Trajectory %d, IC %d: signal_rms = %.6e, noise_std = %.6e" % (
                    i, j,
                    float(torch.sqrt(torch.mean(self.U_Train_Clean[i][j]**2)).item()),
                    float(self.noise_ratio * torch.sqrt(torch.mean(self.U_Train_Clean[i][j]**2)).item())));

        LOGGER.info("Noise injection complete. Clean data saved in U_Train_Clean.");
        return;




    # -------------------------------------------------------------------------------------------------
    # Normalization helpers
    # -------------------------------------------------------------------------------------------------

    def has_normalization(self) -> bool:
        return bool(self.normalize and (self.data_mean is not None) and (self.data_std is not None));



    def _compute_mean_std_from_U(self, U: list[list[torch.Tensor]], eps: float = 1.0e-12) -> tuple[list[float], list[float]]:
        """
        Compute mean/std across ALL entries in U for each IC separately (scalar values).

        We do this without concatenating everything to avoid large memory spikes.
        """
        assert isinstance(U, list) and len(U) > 0, "U must be a non-empty list";
        n_IC: int = len(U[0]);
        assert n_IC > 0, "n_IC must be positive";
        for i in range(len(U)):
            assert len(U[i]) == n_IC, "U[%d] has %d ICs but expected %d" % (i, len(U[i]), n_IC);

        sum_      : list[float] = [0.0] * n_IC;
        sum_sq    : list[float] = [0.0] * n_IC;
        count     : list[int]   = [0]   * n_IC;

        for i in range(len(U)):
            for j in range(n_IC):
                T: torch.Tensor = U[i][j];
                assert isinstance(T, torch.Tensor), "U[%d][%d] is not a torch.Tensor" % (i, j);
                Td = T.detach().double();
                sum_[j]   += float(Td.sum().item());
                sum_sq[j] += float((Td * Td).sum().item());
                count[j]  += int(Td.numel());

        means: list[float] = [];
        stds : list[float] = [];
        for j in range(n_IC):
            assert count[j] > 0, "No elements found for IC %d" % j;
            mean_j: float = sum_[j] / float(count[j]);
            var_j: float  = (sum_sq[j] / float(count[j])) - (mean_j * mean_j);
            if var_j < 0.0:
                # Numerical guard
                var_j = 0.0;
            std_j: float = float(numpy.sqrt(max(var_j, eps)));
            means.append(mean_j);
            stds.append(std_j);
        return means, stds;



    def set_normalization_stats_from_training(self) -> None:
        """
        Compute and store mean/std from current training trajectories.
        Stats live on the trainer only; downstream utilities should be passed the trainer.
        """
        assert self.normalize, "Normalization is disabled";
        means, stds = self._compute_mean_std_from_U(self.U_Train);
        self.data_mean = [torch.tensor(m, dtype = torch.float32) for m in means];
        self.data_std  = [torch.tensor(s, dtype = torch.float32) for s in stds];

        LOGGER.info("Normalization enabled (from TRAINING set). Per-IC mean/std:");
        for j in range(len(means)):
            LOGGER.info("  IC %d: mean = %.6e, std = %.6e" % (j, means[j], stds[j]));
        LOGGER.warning("Note: Stats computed from %d training points. Consider using test set for better global statistics." % len(self.U_Train));
        return;
    
    

    def set_normalization_stats_from_test(self) -> None:
        """
        Compute and store mean/std from ALL test trajectories (better global statistics).
        This is preferred over training-only stats when training set is small (e.g., 4 corners).
        """
        assert self.normalize, "Normalization is disabled";
        assert len(self.U_Test) > 0, "Test set is empty!";
        means, stds = self._compute_mean_std_from_U(self.U_Test);
        self.data_mean = [torch.tensor(m, dtype = torch.float32) for m in means];
        self.data_std  = [torch.tensor(s, dtype = torch.float32) for s in stds];

        LOGGER.info("Normalization enabled (from TEST set - better global statistics). Per-IC mean/std:");
        for j in range(len(means)):
            LOGGER.info("  IC %d: mean = %.6e, std = %.6e" % (j, means[j], stds[j]));
        LOGGER.info("Stats computed from %d test points (full parameter space)." % len(self.U_Test));
        return;



    def normalize_tensor(self, X: torch.Tensor, ic_idx: int) -> torch.Tensor:
        if not self.has_normalization():
            return X;
        assert self.data_mean is not None and self.data_std is not None;
        m = float(self.data_mean[ic_idx].item());
        s = float(self.data_std[ic_idx].item());
        return (X - m) / s;



    def denormalize_tensor(self, X: torch.Tensor, ic_idx: int) -> torch.Tensor:
        if not self.has_normalization():
            return X;
        assert self.data_mean is not None and self.data_std is not None;
        m = float(self.data_mean[ic_idx].item());
        s = float(self.data_std[ic_idx].item());
        return X * s + m;



    def denormalize_np(self, x: numpy.ndarray, ic_idx: int) -> numpy.ndarray:
        """
        De-normalize a numpy array using the trainer's stored stats (per IC).
        """
        if not self.has_normalization():
            return x;
        assert self.data_mean is not None and self.data_std is not None;
        m = float(self.data_mean[ic_idx].detach().cpu().item());
        s = float(self.data_std[ic_idx].detach().cpu().item());
        return x * s + m;



    def scale_std_np(self, std_x: numpy.ndarray, ic_idx: int) -> numpy.ndarray:
        """
        Convert a standard deviation computed in normalized units to physical units.
        """
        if not self.has_normalization():
            return std_x;
        assert self.data_std is not None;
        s = float(self.data_std[ic_idx].detach().cpu().item());
        return std_x * s;



    def normalize_U_inplace(self, U: list[list[torch.Tensor]]) -> None:
        """
        Normalize a dataset in-place (per IC) using stored mean/std.
        """
        if not self.has_normalization():
            return;
        assert self.data_mean is not None and self.data_std is not None;
        n_IC: int = len(self.data_mean);
        for i in range(len(U)):
            assert len(U[i]) == n_IC, "U[%d] has %d ICs but expected %d" % (i, len(U[i]), n_IC);
            for j in range(n_IC):
                U[i][j] = self.normalize_tensor(U[i][j], j);
        return;





    # ---------------------------------------------------------------------------------------------
    # Loss Tracking Helpers.
    # ---------------------------------------------------------------------------------------------

    def _cache_loss(self, 
                    loss_name   : str, 
                    loss_value  : torch.Tensor,
                    param_tuple : tuple | None = None) -> None:
        """
        Cache a loss tensor for deferred scalar logging. 

        `Iterate(...)` implementations should call this method at the point where a loss component
        is computed, but they should pass a detached tensor rather than calling `.item()`. At the 
        end of a step, they should use `_flush_loss_cache(epoch)` to write all losses from that 
        epoch to file. That method batches the device-to-CPU scalar transfer once per optimization 
        step instead of synchronizing the GPU repeatedly throughout the forward/loss code.

        This method can be used to cache a loss for a particular parameter, or the total (sum 
        across parameters). Expected use inside trainers:

            self._cache_loss("LD", loss_LD_i.detach(), param_tuple) # LD loss for a specific parameter
            self._cache_loss("LD", loss_LD.detach())                # Total LD loss

        Do not pass Python floats and do not call `.item()` before caching.  The tensor must contain
        exactly one scalar value and must already be detached from the autograd graph.  The matching
        `_flush_loss_cache(...)` call should run once per training step after `optimizer.step()`.


        -------------------------------------------------------------------------------------------
        Arguments:
        -------------------------------------------------------------------------------------------

        loss_name : str
            Name of the loss component (e.g., 'recon', 'rollout_ROM')
        loss_value : torch.Tensor
            Detached scalar tensor containing the loss value to cache.
        param_tuple : tuple | None
            Optional parameter combination as a tuple (can be used as dictionary key). If not 
            specified, we assume `total`. 
        """

        assert isinstance(loss_name, str),              "loss_name must be a string";
        if param_tuple is not None:
            assert isinstance(param_tuple, tuple),      "param_tuple must be a tuple or None";
        assert isinstance(loss_value, torch.Tensor),    "loss_value must be a torch.Tensor; pass loss.detach(), not loss.item()";
        assert loss_value.numel() == 1,                 "loss_value must contain exactly one scalar value";
        assert loss_value.requires_grad == False,       "loss_value must be detached before caching; pass loss.detach()";

        key : tuple | str = param_tuple if param_tuple is not None else 'total';
        self._loss_cache.append((loss_name, key, loss_value));
        return;


    def _process_latent_dynamics_losses(
            self,
            raw_loss_dict  : dict[str, list[torch.Tensor] | torch.Tensor],
            params         : numpy.ndarray,
            device         : torch.device | str,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        r"""
        Cache, aggregate, and weight an LD-owned loss dictionary.

        `LatentDynamics.compute_losses(...)` returns a dictionary whose keys must match
        `latent_dynamics.loss_weights`. Each value is either a length-n_train list of scalar tensors
        for parameter-specific losses or one scalar tensor for a global loss. This helper validates
        that contract, caches per-parameter entries when they exist, caches one total for every
        latent-dynamics loss key, and forms the weighted latent-dynamics contribution to the
        trainer objective.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        raw_loss_dict : dict[str, list[torch.Tensor] | torch.Tensor]
            The loss dictionary returned by `self.latent_dynamics.compute_losses(...)`. Its keys
            must exactly match `self.latent_dynamics.loss_weights`.

        params : numpy.ndarray, shape = (n_train, n_p)
            Training parameter values corresponding to any per-parameter loss lists in
            `raw_loss_dict`. The i'th row is used as the logging key for the i'th entry of each
            per-parameter loss list.

        device : torch.device or str
            Device on which to create the initial zero tensor for the weighted latent-dynamics loss.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        loss_dict : dict[str, torch.Tensor]
            Unweighted scalar total for each latent-dynamics loss key. Per-parameter loss lists are
            summed across parameters; global scalar losses are passed through unchanged.

        weighted_loss_sum : torch.Tensor
            Scalar tensor equal to `sum(self.latent_dynamics.loss_weights[key] * loss_dict[key])`.
        """

        # Ensure the keys in the loss dict match those in the LD loss weights dict.
        expected_keys : set[str] = set(self.latent_dynamics.loss_weights.keys());
        actual_keys   : set[str] = set(raw_loss_dict.keys());
        if actual_keys != expected_keys:
            raise ValueError("LatentDynamics.compute_losses returned keys %s, but loss_weights has keys %s" % (
                sorted(actual_keys),
                sorted(expected_keys),
            ));

        loss_dict          : dict[str, torch.Tensor] = {};
        weighted_loss_sum  : torch.Tensor           = torch.zeros((), dtype = torch.float32, device = device);

        # Process the weights item-by-item; summing per-parameter losses.
        for key, value in raw_loss_dict.items():
            if isinstance(value, list):
                assert len(value) == params.shape[0], "Loss `%s` has %d per-parameter entries, expected %d" % (key, len(value), params.shape[0]);
                for i, param_loss in enumerate(value):
                    assert isinstance(param_loss, torch.Tensor), "Loss `%s` entry %d is not a torch.Tensor" % (key, i);
                    param_tuple = tuple(params[i, :]);
                    self._cache_loss(key, param_loss.detach(), param_tuple);
                total_loss = torch.sum(torch.stack(value));
            else:
                assert isinstance(value, torch.Tensor), "Loss `%s` must be a torch.Tensor or list[torch.Tensor]" % key;
                total_loss = value;

            self._cache_loss(key, total_loss.detach());
            loss_dict[key] = total_loss;
            weighted_loss_sum = weighted_loss_sum + self.latent_dynamics.loss_weights[key] * total_loss;

        # All done :)
        return loss_dict, weighted_loss_sum;
    


    @staticmethod
    def _jsonable_param(param_tuple: tuple) -> list[Any]:
        """
        Convert a parameter tuple into JSON-compatible scalar values.
        """

        jsonable_param : list[Any] = [];
        for value in param_tuple:
            if isinstance(value, numpy.generic):
                value = value.item();
            assert isinstance(value, (int, float, str, bool)) or value is None, "parameter values must be JSON scalar values";
            jsonable_param.append(value);
        return jsonable_param;



    def _flush_loss_cache(self, epoch: int) -> dict[tuple[str, tuple | str], float]:
        """
        Flush cached loss tensors gathered during one epoch to a row in the jsonl file 
        self.loss_by_param_path.

        This method converts all cached detached scalar tensors into Python floats with a single
        batched CPU transfer, then appends one JSON object to `self.loss_by_param_path`. Trainer
        subclasses should call this exactly once per training step, after `optimizer.step()` and
        before checkpoint/report logic that needs scalar loss values.

        The returned dictionary maps `(loss_name, param_tuple_or_total)` to the flushed float for
        the current cache contents.  This lets trainers reuse the synchronized values for reporting
        and best-loss checkpoint decisions without calling `.item()` again.

        
        -------------------------------------------------------------------------------------------
        Arguments:
        -------------------------------------------------------------------------------------------

        epoch : int
            Epoch number

        
        -------------------------------------------------------------------------------------------
        Returns:
        -------------------------------------------------------------------------------------------

        flushed_values : dict[tuple[str, tuple | str], float]
            The scalar values flushed from the cache.  Total losses use the key
            `(loss_name, 'total')`.
        """

        # Checks
        assert isinstance(epoch, int),                  "epoch must be an int";

        # Setup 
        if len(self._loss_cache) == 0:
            return {};

        # Stack first, then transfer once.  Losses for a trainer step should all live on one device.
        first_device = self._loss_cache[0][2].device;
        for _, _, loss_value in self._loss_cache:
            assert loss_value.device == first_device, "cached loss tensors must live on the same device for batched flushing";

        # Fetch loss values, then convert to cpu/list.
        values_tensor : torch.Tensor = torch.stack([entry[2].reshape(()) for entry in self._loss_cache], dim = 0);
        assert bool(torch.isfinite(values_tensor).all()), "cached loss tensors must be finite for JSONL logging";
        values_list   : list[float]  = [float(x) for x in values_tensor.cpu().tolist()];

        flushed_values : dict[tuple[str, tuple | str], float] = {};
        loss_records   : list[dict[str, Any]] = [];
        for (loss_name, key, _), loss_float in zip(self._loss_cache, values_list):
            flushed_values[(loss_name, key)] = loss_float;
            if key == 'total':
                loss_records.append({
                    "loss_name" : loss_name,
                    "param"     : None,
                    "value"     : loss_float,
                });
            else:
                assert isinstance(key, tuple), "cached non-total loss keys must be parameter tuples";
                loss_records.append({
                    "loss_name" : loss_name,
                    "param"     : Trainer._jsonable_param(key),
                    "value"     : loss_float,
                });

        # Now write to file.
        with open(self.loss_by_param_path, "a", encoding = "utf-8") as handle:
            json.dump({"epoch" : epoch, "losses" : loss_records}, handle, sort_keys = True, allow_nan = False);
            handle.write("\n");
            LOGGER.debug("Saved losses from epoch %d to %s" % (epoch, self.loss_by_param_path));

        # Reset cache loss and return :) 
        self._loss_cache = [];
        return flushed_values;





    # ---------------------------------------------------------------------------------------------
    # Latent dynamics coefficient helpers
    # ---------------------------------------------------------------------------------------------

    def _check_train_coefficients(self) -> None:
        """
        Verify every training parameter has LD-owned native coefficients.

        This is intentionally a check, not synchronization; missing coefficients indicate a sampler
        or initialization bug and should stop execution.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Nothing!
        """
        for i in range(self.param_space.n_train()):
            params_i = self.param_space.train_space[i, :];
            coef_dict = self.latent_dynamics.get_train_coefs(params_i);
            assert isinstance(coef_dict, dict), "train_coefs[%s] must be a dict" % str(tuple(params_i));
            assert len(coef_dict) > 0, "train_coefs[%s] is empty" % str(tuple(params_i));
            for name, tensor in coef_dict.items():
                assert isinstance(name, str), "coefficient names must be strings";
                assert isinstance(tensor, torch.Tensor), "coefficient %s must be a torch.Tensor" % name;
        return;



    def _optimizer_parameters(self) -> list[torch.Tensor]:
        """
        Collect EncoderDecoder parameters and LD-owned coefficient tensors for optimization.

        The latent-dynamics coefficients live the latent dynamics object but can be fetched using 
        the `trainable_tensors` method (which should return all trainable tensors in the 
        LD object).


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        parameters : list[torch.Tensor]
            Trainable tensors that should be passed to a torch optimizer.
        """

        trainable_params = [];
        if self.encoder_decoder.trainable == True:
            trainable_params.extend(list(self.encoder_decoder.parameters()));
        if self.latent_dynamics.trainable == True:
            self._check_train_coefficients();
            trainable_params.extend(self.latent_dynamics.trainable_tensors());

        return trainable_params;



    # ---------------------------------------------------------------------------------------------
    # Checkpointing
    # ---------------------------------------------------------------------------------------------

    def _Save_Checkpoint(self, encoder_decoder : EncoderDecoder, iter : int) -> str:
        """
        Used to serialize a copy of the EncoderDecoder parameters and LatentDynamics state.

        The latent-dynamics coefficients are owned by `self.latent_dynamics`, so checkpointing now
        stores the LatentDynamics export dictionary rather than separate flattened train/test
        coefficient arrays. This includes the latent dynamics tensors, whose values are native
        coefficient dictionaries.



        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        encoder_decoder : EncoderDecoder
            The EncoderDecoder object whose state dictionary we want to serialize.

        iter : int
            The iteration number corresponding to when we obtained the best model/coefficients.



        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        checkpoint_path : str
            A string housing the path to the file housing the saved checkpoint.
        """

        # Run checks. This is intentionally strict: every training parameter should already have a
        # corresponding native coefficient dictionary in the LatentDynamics object.
        self._check_train_coefficients();

        # Set up the checkpoint path.
        checkpoint_path : str = self.path_checkpoint + '/' + 'checkpoint.pt';

        # Fetch a detached CPU copy of the encoder-decoder parameters without moving the live model.
        with torch.no_grad():
            model_state: dict[str, torch.Tensor] = {
                k: v.detach().cpu().clone()
                for k, v in encoder_decoder.state_dict().items()
            }

        # Serialize the encoder_decoder parameters and the LatentDynamics export dictionary.
        # The LatentDynamics export handles moving coefficient tensors to CPU and detaching them.
        torch.save({"EncoderDecoder_state_dict"     : model_state,
                    "latent_dynamics"               : self.latent_dynamics.export(),
                    "iteration number"              : iter},
                    checkpoint_path);

        return checkpoint_path;



    def Load_Checkpoint(self) -> tuple[EncoderDecoder, int]:
        """
        Deserializes the EncoderDecoder parameters and LatentDynamics state from the latest
        checkpoint. Note that the loaded encoder_decoder will always be on cpu, so you may need to
        manually move it to another device if cpu is insufficient.

        The LatentDynamics load method replaces the latent dynamics object's internal tensors 
        with those from the checkpoint and restores each trainable tensor as a trainable leaf
        tensor.



        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        encoder_decoder, iter

        encoder_decoder : EncoderDecoder
            The de-serialized EncoderDecoder object, mapped to the cpu.

        iter : int
            The iteration number corresponding to when the checkpoint was made.
        """

        # Set up the checkpoint path.
        checkpoint_path : str = self.path_checkpoint + '/' + 'checkpoint.pt';

        # Load the checkpoint.
        # NOTE: PyTorch >= 2.6 defaults `weights_only=True`, which disallows loading arbitrary
        # pickled objects. Our checkpoint intentionally stores dictionaries of tensors, so we must
        # set `weights_only=False`.
        checkpoint_dict : dict = torch.load(checkpoint_path, map_location = 'cpu', weights_only = False);

        # Load the EncoderDecoder state dictionary.
        self.encoder_decoder.cpu().load_state_dict(checkpoint_dict["EncoderDecoder_state_dict"]);

        # Restore the LatentDynamics metadata and native training coefficient dictionaries.
        self.latent_dynamics.load(checkpoint_dict["latent_dynamics"]);

        # Fetch the checkpoint iteration number.
        iter : int = checkpoint_dict["iteration number"];

        # All done!
        return self.encoder_decoder, iter;



    # ---------------------------------------------------------------------------------------------
    # Training.
    # ---------------------------------------------------------------------------------------------

    def Iterate(self, 
                start_iter  : int, 
                end_iter    : int, 
                profiler    : torch.profiler.profile | None = None) -> None:
        """
        Runs a round of training. It should train the encoder_decoder and training coefficients 
        from iteration = start_iter to iteration = end_iter. Along the way, it should make 
        checkpoints by calling `self._Save_Checkpoint(...)`. After training, we load the latest checkpoint
        and use the serialized encoder_decoder and coefficients to update the encoder_decoder 
        and latent dynamic coefficients, respectively. 

        The function should also track specific losses for each training parameter combination
        during each epoch using `_cache_loss`, then call `_flush_loss_cache` once per epoch after 
        the optimizer step.

        Finally, this function should record how long each part of the training process takes. 
        Specifically, it should track how long each loss function takes to compute, as well as how 
        long the back propagation step takes. It should record all of this using the self.timer
        attribute (see Utilities/Timing for details).

        Note that if normalization is enabled, the entires in U_Train and U_Test will already be 
        normalized when they are stored in the Trainer object. This also means that the 
        EncoderDecoder should be trained using normalized data (if you just fetch from self.U_Train,
        then this shouldn't be an issue). You may need to normalize data from the physics (such 
        as initial conditions) before passing them into the EncoderDecoder. 
        
        
        -------------------------------------------------------------------------------------------
        Arguments:
        -------------------------------------------------------------------------------------------

        start_iter : int
            The index of the first training iteration. Must have start_iter <= end_iter.

        end_iter : int 
            The index of the last training iteration. Must have start_iter <= end_iter.

        profiler : torch.profiler.profile
            An optional torch profiler that can be used to profile Iterate.

            
        -------------------------------------------------------------------------------------------
        Returns:
        -------------------------------------------------------------------------------------------

        None! 
        """

        raise RuntimeError("Abstract method Trainer.Iterate!");



    def train(self) -> None:
        """
        Runs one round of training and restores the in-memory state to the best checkpoint
        produced during that round.

        This method is "round-based": each call advances the global iteration counter
        `restart_iter` by at most `n_iter` (and never beyond `max_iter`). The concrete training
        behavior is implemented by the subclass `Iterate(...)` method.

        Important semantic: at the end of the round, the EncoderDecoder and latent-dynamics
        coefficients are restored from the *best epoch of the round* (not the final epoch). This is
        critical because greedy sampling should use the best available coefficients when fitting
        interpolators / evaluating errors.
        """
        
        # -------------------------------------------------------------------------------------
        # Setup. 

        # Make sure we have at least one training data point.
        assert len(self.U_Train) > 0, "len(self.U_Train) = %d" % len(self.U_Train);
        assert len(self.U_Train) == self.param_space.n_train(), "len(self.U_Train) = %d, self.param_space.n_train() = %d" % (len(self.U_Train), self.param_space.n_train());

        # Apply optional base-Trainer noise before subclasses build device copies or rollout
        # targets. This keeps all training losses for this round consistent with the same noisy
        # data. New greedy samples are clean when appended, so this call also backs them up in
        # U_Train_Clean before corruption.
        if self.noise_ratio > 0.0:
            self.apply_noise_to_U_Train();

        # Make sure the checkpoints and results directories exist.
        from pathlib import Path
        Path(self.path_checkpoint).mkdir(   parents = True, exist_ok = True);
        Path(self.path_results).mkdir(      parents = True, exist_ok = True);


        # -----------------------------------------------------------------------------------------
        # Initialize loss tracking
        
        # Delete existing files if starting fresh (restart_iter == 0)
        # This ensures we don't append to results from previous training runs
        if self.restart_iter == 0:
            if os.path.exists(self.loss_by_param_path):
                os.remove(self.loss_by_param_path);
                LOGGER.info("Deleted existing loss_by_param file: %s" % self.loss_by_param_path);

        # Reset loss cache.
        self._loss_cache = [];


        # -----------------------------------------------------------------------------------------
        # Run the iterations!

        n_train      : int  = self.param_space.n_train();
        start_iter   : int  = self.restart_iter;
        end_iter     : int  = min(self.restart_iter + self.n_iter, self.max_iter);
        assert end_iter >= start_iter;
        LOGGER.info("Training for %d epochs (starting at %d, going to %d) with %d training parameters" % (end_iter - start_iter, start_iter, end_iter, n_train));

        if PROFILE_ITERATE:
            # Iterate with profiler on
            profiler_activities = [torch.profiler.ProfilerActivity.CPU];
            if torch.cuda.is_available():
                profiler_activities.append(torch.profiler.ProfilerActivity.CUDA);

            with torch.profiler.profile(
                activities      = profiler_activities,
                schedule        = torch.profiler.schedule(
                    wait        = PROFILE_WAIT,
                    warmup      = PROFILE_WARMUP,
                    active      = PROFILE_ACTIVE,
                    repeat      = PROFILE_REPEAT,
                ),
                record_shapes   = True,
                profile_memory  = True,
            ) as prof:
                self.Iterate(start_iter = start_iter, end_iter = end_iter, profiler = prof);

            # Build a string to hold the profiler results.
            profiler_results_list : list[str] = [
                "=" * 120,
                "HLaSDI training profiler results",
                "=" * 120,
                "",
            ]
            if torch.cuda.is_available():
                profiler_results_list.extend(
                    [
                        "--- Sorted by CUDA time total ---",
                        "",
                        prof.key_averages().table(sort_by="cuda_time_total", row_limit=30),
                        "",
                    ]
                )
            profiler_results_list.extend(
                [
                    "--- Sorted by CPU time total ---",
                    "",
                    prof.key_averages().table(sort_by="cpu_time_total", row_limit=30),
                    "",
                    "=" * 120,
                    "",
                ]
            )

            # Write profile results to file.
            profiler_table_str: str = "\n".join(profiler_results_list)
            with open("./HLaSDI_train_profile.txt", "a", encoding="utf-8") as handle:
                handle.write(profiler_table_str)
            
            # Now print the profiling results!
            print(profiler_table_str, flush = True);
        else:
            self.Iterate(start_iter = start_iter, end_iter = end_iter);
        

        # -------------------------------------------------------------------------------------
        # Load model/params from checkpoint.

        # We are ready to wrap up the training procedure.
        self.timer.start("finalize");

        self.encoder_decoder, iter = self.Load_Checkpoint();
        LOGGER.info("We attained our best performance on epoch %d. Replacing encoder_decoder, latent dynamics coefficients with the checkpoint from that epoch" % iter);


        # -------------------------------------------------------------------------------------
        # Wrap up

        # Now that we have completed another round of training, update the restart iteration.
        self.restart_iter = end_iter;

        # Report timing information.
        self.timer.end("finalize");
        self.timer.log();

        # All done!
        return;





    # ---------------------------------------------------------------------------------------------
    # Save, Load
    # ---------------------------------------------------------------------------------------------

    def export(self) -> dict:
        """
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        dict_ : dict
            A dictionary housing most of the internal variables in self. You can pass this 
            dictionary to self (after initializing it using ParameterSpace, encoder_decoder, and 
            LatentDynamics objects) to make a GLaSDI object whose internal state matches that of 
            self.
        """

        config = self.config.model_dump(mode = "python", by_alias = True) if hasattr(self.config, "model_dump") else self.config;
        dict_ = {'U_Train'                  : self.U_Train,
                 'U_Train_Clean'            : self.U_Train_Clean,
                 'noise_ratio'              : self.noise_ratio,
                 'U_Test'                   : self.U_Test,
                 't_Train'                  : self.t_Train,
                 't_Test'                   : self.t_Test,
                 'restart_iter'             : self.restart_iter, 
                 'timer'                    : self.timer.export(), 
                 'config'                   : config,
                 'normalize'                : self.normalize,
                 'data_mean'                : None if self.data_mean is None else [float(m.detach().cpu().item()) for m in self.data_mean],
                 'data_std'                 : None if self.data_std  is None else [float(s.detach().cpu().item()) for s in self.data_std]};
        return dict_;



    def load(self, dict_ : dict) -> None:
        """
        Modifies self's internal state to match the one whose export method generated the dict_ 
        dictionary.


        -------------------------------------------------------------------------------------------
        Arguments 
        -------------------------------------------------------------------------------------------

        dict_ : dict 
            This should be a dictionary returned by calling the export method on another 
            GLaSDI object. We use this to make self hav the same internal state as the object that 
            generated dict_. 
            

        -------------------------------------------------------------------------------------------
        Returns  
        -------------------------------------------------------------------------------------------
        
        Nothing!
        """

        # Extract instance variables from dict_.
        self.U_Train            : list[list[torch.Tensor]]  = dict_['U_Train'];             # len = n_train, i'th element is an n_IC element list.  
        self.U_Train_Clean      : list[list[torch.Tensor]]  = dict_['U_Train_Clean'];       # len = n_train, i'th element is an n_IC element list.  
        self.noise_ratio        : float                     = float(dict_['noise_ratio']);
        self.U_Test             : list[list[torch.Tensor]]  = dict_['U_Test'];              # len = n_test, i'th element is an n_IC element list.

        self.t_Train            : list[torch.Tensor]        = dict_['t_Train'];             # len = n_train.
        self.t_Test             : list[torch.Tensor]        = dict_['t_Test'];              # len = n_test.

        self.restart_iter       : int                       = dict_['restart_iter'];

        # Restore normalization stats (if present).
        self.normalize = bool(dict_.get('normalize', False));
        dm = dict_.get('data_mean', None);
        ds = dict_.get('data_std', None);
        if self.normalize and (dm is not None) and (ds is not None):
            # Load scalar stats (handle both raw floats and scalar numpy arrays)
            self.data_mean = [torch.tensor(float(x) if not isinstance(x, numpy.ndarray) else float(x.item()), dtype = torch.float32) for x in dm];
            self.data_std  = [torch.tensor(float(x) if not isinstance(x, numpy.ndarray) else float(x.item()), dtype = torch.float32) for x in ds];
        else:
            self.data_mean = None;
            self.data_std  = None;

        # Next, compute n_IC.           
        self.n_IC = len(self.U_Test[0]);

        # Load the timer / optimizer. 
        self.timer.load(dict_['timer']);
        self._loss_cache = [];


        # All done!
        return;
