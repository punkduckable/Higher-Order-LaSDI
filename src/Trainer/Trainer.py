# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

import  sys;
import  os;
# Add sibling (src/*) directories to the search path. This file lives in src/Trainer/.
src_path            : str   = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir));
Physics_Path        : str   = os.path.join(src_path, "Physics");
LD_Path             : str   = os.path.join(src_path, "LatentDynamics");
EncoderDecoder_Path : str   = os.path.join(src_path, "EncoderDecoder");
Utils_Path          : str   = os.path.join(src_path, "Utilities");
sys.path.append(Physics_Path);
sys.path.append(LD_Path);
sys.path.append(EncoderDecoder_Path);
sys.path.append(Utils_Path);

import  logging;

import  torch;
import  numpy;
from    torch.optim                 import  Optimizer;
import  pickle;

from    EncoderDecoder              import  EncoderDecoder;
from    Timing                      import  Timer;
from    ParameterSpace              import  ParameterSpace;
from    Physics                     import  Physics;
from    LatentDynamics              import  LatentDynamics;

# Setup Logger
LOGGER : logging.Logger = logging.getLogger(__name__);



# -------------------------------------------------------------------------------------------------
# Trainer Base class
# -------------------------------------------------------------------------------------------------

class Trainer:
    # An n_Train element list. The i'th element is is an n_IC element list whose j'th element is a
    # numpy ndarray of shape (n_t(i), Frame_Shape) holding a sequence of samples of the j'th 
    # derivative of the FOM solution when we use the i'th combination of training values. 
    # NOTE: these are initialized as instance variables in __init__ (do not share across instances).
    U_Train : list[list[torch.Tensor]];

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

    # A tensor holding the coefficients for each testing parameter that we obtained during 
    # the best training iteration
    best_train_coefs : numpy.ndarray | None;               # shape (n_train, n_coef)

    # A tensor holding the coefficients for each testing parameter that we obtained during 
    # the best training iteration.
    test_coefs : torch.nn.parameter.Parameter;                 # shape (n_test, n_coef)

    # The trainer's device
    device : str;



    def __init__(   self, 
                    n_IC               : int, 
                    physics            : Physics, 
                    encoder_decoder    : EncoderDecoder, 
                    latent_dynamics    : LatentDynamics, 
                    param_space        : ParameterSpace, 
                    trainer_config     : dict):
        """
        Abstract base class for training strategies.

        A `Trainer` instance owns the state of a Higher-Order-LaSDI run:

        - The training and testing datasets (`U_Train`, `t_Train`, `U_Test`, `t_Test`)
        - Optional global normalization statistics (`data_mean`, `data_std`)
        - The model objects (`physics`, `encoder_decoder`, `latent_dynamics`)
        - Trainable latent-dynamics coefficients for every point in the test space (`test_coefs`)
        - Bookkeeping for iterative training + greedy sampling (`restart_iter`, `n_iter`, etc.)
        - Performance timing (`timer`) and per-parameter loss logging (`loss_by_param`)

        Subclasses implement `Iterate(start_iter, end_iter)` which performs optimization steps
        and calls `_Save_Checkpoint(...)` whenever a new best model is found within the round.
        The base class `train()` method drives the round-by-round schedule and ensures that, at
        the end of each round, `encoder_decoder` and `test_coefs` are restored to their *best*
        values from that round (not necessarily the final epoch).

        The YAML config convention is:

        - `trainer.type` selects the subclass (e.g., `"Rollout_1_IC"`)
        - Base settings live directly under `trainer` (e.g., `n_iter`, `max_iter`, `normalize`)
        - Subclass-specific settings live under `trainer[trainer.type]` (e.g., `trainer.Rollout_1_IC.lr`)


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
        self.t_Train                        = [];
        self.U_Test                         = [];
        self.t_Test                         = [];
        
        # Initialize a timer object. We will use this while training.
        self.timer                          = Timer();

        # Fetch trainer class information.
        self.n_iter                 : int   = trainer_config['n_iter'];             # Number of iterations for one train and greedy sampling
        self.max_iter               : int   = trainer_config['max_iter'];           # We stop training if restart_iter goes above this number. 
        self.max_greedy_iter        : int   = trainer_config['max_greedy_iter'];    # We stop performing greedy sampling if restart_iter goes above this number.
        device                      : str   = trainer_config.get('device', 'cpu');  # The device we want to map the trainer and its attributes to (and where we will perform training).

        # Optional normalization (training-only stats).
        # If enabled, we compute a single mean/std across ALL training trajectories (per IC),
        # then normalize both training + testing trajectories using these values.
        self.normalize              : bool                      = trainer_config['normalize'];
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

        # If we are learning the latent dynamics coefficients, then we need to set up 
        # a torch Parameter housing the the coefficients for each combination of testing 
        # parameters. If we aren't learning the coefficients, then this will never be used.
        self.test_coefs : torch.nn.parameter.Parameter = torch.nn.parameter.Parameter(
            torch.zeros(
                self.param_space.n_test(),
                self.latent_dynamics.n_coefs,
                dtype = torch.float32,
                device = self.device,
                requires_grad = True,
            )
        );

        # Set paths for checkpointing/results.
        # Put these in the Higher-Order-LaSDI project root.
        src_dir     = os.path.dirname(os.path.abspath(__file__));
        project_dir = os.path.abspath(os.path.join(src_dir, os.pardir));
        self.path_checkpoint    : str = os.path.join(project_dir, "checkpoint");
        self.path_results       : str = os.path.join(project_dir, "results");

        # Make sure the checkpoints and results directories exist.
        from pathlib import Path;
        Path(self.path_checkpoint).mkdir(   parents = True, exist_ok = True);
        Path(self.path_results).mkdir(      parents = True, exist_ok = True);
        LOGGER.info("Checkpoint directory: %s" % self.path_checkpoint);
        LOGGER.info("Results directory: %s" % self.path_results);

        # Final setup.
        self.best_train_coefs   = None;
        self.restart_iter       = 0;                # Global iteration index at the start of the next training round
        self.best_epoch         = None;             # Optional: subclasses may set this when checkpointing
        
        # All done!
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

    def _store_loss_by_param(self, loss_name: str, param_tuple: tuple, epoch: int, loss_value: float) -> None:
        """
        Helper function to store a loss value for a specific parameter combination.
        

        -------------------------------------------------------------------------------------------
        Arguments:
        -------------------------------------------------------------------------------------------

        loss_name : str
            Name of the loss component (e.g., 'recon', 'rollout_ROM')
        param_tuple : tuple
            Parameter combination as a tuple (can be used as dictionary key)
        epoch : int
            Epoch number
        loss_value : float
            Loss value to store
        """
        if loss_name not in self.loss_by_param:
            self.loss_by_param[loss_name] = {};
        if param_tuple not in self.loss_by_param[loss_name]:
            self.loss_by_param[loss_name][param_tuple] = {'epochs': [], 'losses': []};
        self.loss_by_param[loss_name][param_tuple]['epochs'].append(epoch);
        self.loss_by_param[loss_name][param_tuple]['losses'].append(loss_value);
    


    def _store_total_loss(self, loss_name: str, epoch: int, loss_value: float) -> None:
        """
        Helper function to store a total loss value (summed across all parameter combinations).
        


        -------------------------------------------------------------------------------------------
        Arguments:
        -------------------------------------------------------------------------------------------
        loss_name : str
            Name of the loss component
        epoch : int
            Epoch number
        loss_value : float
            Total loss value to store
        """
        if loss_name not in self.loss_by_param:
            self.loss_by_param[loss_name] = {};
        if 'total' not in self.loss_by_param[loss_name]:
            self.loss_by_param[loss_name]['total'] = {'epochs': [], 'losses': []};
        self.loss_by_param[loss_name]['total']['epochs'].append(epoch);
        self.loss_by_param[loss_name]['total']['losses'].append(loss_value);



    # ---------------------------------------------------------------------------------------------
    # Checkpointing
    # ---------------------------------------------------------------------------------------------

    def _Save_Checkpoint(self, encoder_decoder : EncoderDecoder, train_coefs : numpy.ndarray, test_coefs : torch.Tensor | numpy.ndarray, iter : int) -> str:
        """
        Used to serialize a copy of the EncoderDecoder parameters and latent dynamics
        coefficients.

        

        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        encoder_decoder : EncoderDecoder
            The EncoderDecoder object we want to serialize.
        
        train_coefs : numpy.ndarray, shape = (n_train, n_coefs).
            The array whose i'th row holds the training coefficients for the i'th training 
            parameter.
        
        test_coefs : torch.Tensor or numpy.ndarray, shape = (n_test, n_coefs)
            The coefficient matrix for the *test* parameter space. In most workflows this is
            simply `self.test_coefs`, which is a learnable torch Parameter.

        iter : int
            The iteration number corresponding to when we obtained the best model/coefficients.

            

        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        checkpoint_path : str
            A string housing the path to the file housing the saved checkpoint. 
        """

        # Run checks.
        assert isinstance(train_coefs, numpy.ndarray),                      "train_coefs must be a numpy.ndarray, not %s" % str(type(train_coefs));
        assert len(train_coefs.shape)   == 2,                               "train_coefs must have shape (%d, %d), got %s" % (self.param_space.n_train(), self.latent_dynamics.n_coefs, str(train_coefs.shape));
        assert train_coefs.shape[0]     == self.param_space.n_train(),      "train_coefs must have shape (%d, %d), got %s" % (self.param_space.n_train(), self.latent_dynamics.n_coefs, str(train_coefs.shape));
        assert train_coefs.shape[1]     == self.latent_dynamics.n_coefs,    "train_coefs must have shape (%d, %d), got %s" % (self.param_space.n_train(), self.latent_dynamics.n_coefs, str(train_coefs.shape));

        # Normalize test_coefs to numpy for checkpoint portability.
        if isinstance(test_coefs, torch.Tensor):
            test_coefs_np : numpy.ndarray = test_coefs.detach().cpu().numpy();
        else:
            test_coefs_np : numpy.ndarray = test_coefs;

        assert isinstance(test_coefs_np, numpy.ndarray),                    "test_coefs must be a torch.Tensor or numpy.ndarray, not %s" % str(type(test_coefs));
        assert len(test_coefs_np.shape)  == 2,                              "test_coefs must have shape (%d, %d), got %s" % (self.param_space.n_test(), self.latent_dynamics.n_coefs, str(test_coefs_np.shape));
        assert test_coefs_np.shape[0]    == self.param_space.n_test(),      "test_coefs must have shape (%d, %d), got %s" % (self.param_space.n_test(), self.latent_dynamics.n_coefs, str(test_coefs_np.shape));
        assert test_coefs_np.shape[1]    == self.latent_dynamics.n_coefs,   "test_coefs must have shape (%d, %d), got %s" % (self.param_space.n_test(), self.latent_dynamics.n_coefs, str(test_coefs_np.shape));

        # Set up the checkpoint path.
        checkpoint_path : str = self.path_checkpoint + '/' + 'checkpoint.pt';

        # First, fetch the device for the encoder_decoder.
        device = next(encoder_decoder.parameters()).device;

        # Serialize the encoder_decoder parameters and + coefficients.
        torch.save({"EncoderDecoder_state_dict"     : encoder_decoder.cpu().state_dict(),
                    "train coefficients"            : train_coefs,
                    "test coefficients"             : test_coefs_np,
                    "iteration number"              : iter},
                    checkpoint_path);

        # Move encoder_decoder back to original device after saving
        encoder_decoder.to(device);  

        return checkpoint_path;



    def Load_Checkpoint(self) -> tuple[EncoderDecoder, numpy.ndarray, torch.nn.parameter.Parameter, int]:
        """
        Deserializes the encoder_decoder and coefs attributes from the latest checkpoint. Note that 
        the loaded encoder_decoder will always be on cpu, so you will need to manually move it to 
        another device if cpu is insufficient.


        
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        encoder_decoder, train_coefs, test_coefs, iter

        encoder_decoder : EncoderDecoder
            The de-serialized EncoderDecoder object, mapped to the cpu.
        
        train_coefs : numpy.ndarray, shape = (n_train, n_coefs).
            The array whose i'th row holds the training coefficients for the i'th training 
            parameter.
        
        test_coefs : torch.Tensor, shape = (n_test, n_coefs)
            The tensor whose i'th row holds the training coefficients for the i'th testing 
            parameter.

        iter : int 
            The iteration number corresponding to when the checkpoint was made.
        """

        # Set up the checkpoint path.
        checkpoint_path : str = self.path_checkpoint + '/' + 'checkpoint.pt';

        # Load the checkpoint.
        # NOTE: PyTorch >= 2.6 defaults `weights_only=True`, which disallows loading arbitrary
        # pickled objects (like numpy arrays). Our checkpoint intentionally stores numpy arrays
        # for portability, so we must set `weights_only=False`.
        checkpoint_dict : dict = torch.load(checkpoint_path, map_location = 'cpu', weights_only = False);
        
        # Load the EncoderDecoder state dictionary.
        self.encoder_decoder.cpu().load_state_dict(checkpoint_dict["EncoderDecoder_state_dict"]);

        # Next, fetch the coefficients, iteration number.
        train_coefs      : numpy.ndarray = checkpoint_dict["train coefficients"];
        test_coefs_np    : numpy.ndarray = checkpoint_dict["test coefficients"];
        iter             : int           = checkpoint_dict["iteration number"];

        # Restore test coefficients into the existing learnable Parameter.
        test_coefs_t : torch.Tensor = torch.tensor(test_coefs_np, dtype = torch.float32, device = self.device);
        with torch.no_grad():
            self.test_coefs.data.copy_(test_coefs_t);

        # All done! 
        return self.encoder_decoder, train_coefs, self.test_coefs, iter;




    # ---------------------------------------------------------------------------------------------
    # Training.
    # ---------------------------------------------------------------------------------------------

    def Iterate(self, start_iter : int, end_iter : int) -> None:
        """
        Runs a round of training. It should train the encoder_decoder and training coefficients 
        from iteration = start_iter to iteration = end_iter. Along the way, it should make 
        checkpoints by calling `self._Save_Checkpoint(...)`. After training, we load the latest checkpoint
        and use the serialized encoder_decoder and coefficients to update the encoder_decoder 
        and latent dynamic coefficients, respectively. 

        The function should also track specific losses for each training parameter combination 
        during each epoch using the `_store_loss_by_param` and `_store_total_loss` methods.

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

        Important semantic: at the end of the round, `self.test_coefs` is restored to the value
        from the *best epoch of the round* (not the final epoch). This is critical because greedy
        sampling should use the best available coefficients when fitting GPs / evaluating errors.
        """
        
        # -------------------------------------------------------------------------------------
        # Setup. 

        # Make sure we have at least one training data point.
        assert len(self.U_Train) > 0, "len(self.U_Train) = %d" % len(self.U_Train);
        assert len(self.U_Train) == self.param_space.n_train(), "len(self.U_Train) = %d, self.param_space.n_train() = %d" % (len(self.U_Train), self.param_space.n_train());

        # Make sure the checkpoints and results directories exist.
        from pathlib import Path
        Path(self.path_checkpoint).mkdir(   parents = True, exist_ok = True);
        Path(self.path_results).mkdir(      parents = True, exist_ok = True);


        # -----------------------------------------------------------------------------------------
        # Initialize loss tracking
        
        # Set up filename for loss_by_param.
        # NOTE: must match the filename we save at the end of training so restarts work.
        base_filename       : str = self.physics.config['type'];
        loss_by_param_path  : str = os.path.join(self.path_results, base_filename + '_loss_by_param.pkl');
        
        # Delete existing files if starting fresh (restart_iter == 0)
        # This ensures we don't append to results from previous training runs
        if self.restart_iter == 0:
            if os.path.exists(loss_by_param_path):
                os.remove(loss_by_param_path);
                LOGGER.info("Deleted existing loss_by_param file: %s" % loss_by_param_path);

        # Load existing loss_by_param if restarting (always try to load, don't check hasattr)
        if os.path.exists(loss_by_param_path) and self.restart_iter > 0:
            LOGGER.info("Loading existing per-parameter loss tracking from %s" % loss_by_param_path);
            with open(loss_by_param_path, 'rb') as f:
                self.loss_by_param = pickle.load(f);
        else:
            # Initialize fresh if starting from scratch
            if not hasattr(self, 'loss_by_param'):
                self.loss_by_param = {};
        
        
        # -----------------------------------------------------------------------------------------
        # Run the iterations!

        n_train      : int  = self.param_space.n_train();
        start_iter   : int  = self.restart_iter;
        end_iter     : int  = min(self.restart_iter + self.n_iter, self.max_iter);
        assert end_iter >= start_iter;
        LOGGER.info("Training for %d epochs (starting at %d, going to %d) with %d training parameters" % (end_iter - start_iter, start_iter, end_iter, n_train));
        self.Iterate(start_iter = start_iter, end_iter = end_iter);
        

        # -------------------------------------------------------------------------------------
        # Serialize loss_by_param

        # We are ready to wrap up the training procedure.
        self.timer.start("finalize");
    
        # Keep filename consistent with the one used for restart/load above.
        base_filename       : str   = self.physics.config['type'];
        loss_by_param_path  : str   = os.path.join(self.path_results, base_filename + '_loss_by_param.pkl');

        # Save self.loss_by_param to file.     
        with open(loss_by_param_path, 'wb') as f:
            pickle.dump(self.loss_by_param, f);
        
        LOGGER.info("Saved per-parameter loss tracking to %s" % loss_by_param_path);


        # -------------------------------------------------------------------------------------
        # Load model/params from checkpoint.

        self.encoder_decoder, self.best_train_coefs, self.test_coefs, iter = self.Load_Checkpoint();
        LOGGER.info("We attained our best performance on epoch %d. Replacing encoder_decoder, coefficients with the checkpoint from that epoch" % iter);


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

        dict_ = {'U_Train'                  : self.U_Train, 
                 'U_Test'                   : self.U_Test,
                 't_Train'                  : self.t_Train,
                 't_Test'                   : self.t_Test,
                 'best_train_coefs'         : self.best_train_coefs,                   # Shape = (n_train, n_coefs).
                 # Store as numpy for portability across devices / torch versions.
                 'test_coefs'               : self.test_coefs.detach().cpu().numpy(),  # Shape = (n_test, n_coefs).
                 'restart_iter'             : self.restart_iter, 
                 'timer'                    : self.timer.export(), 
                 'config'                   : self.config,
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
        self.U_Test             : list[list[torch.Tensor]]  = dict_['U_Test'];              # len = n_test, i'th element is an n_IC element list.

        self.t_Train            : list[torch.Tensor]        = dict_['t_Train'];             # len = n_train.
        self.t_Test             : list[torch.Tensor]        = dict_['t_Test'];              # len = n_test.

        self.best_train_coefs   : numpy.ndarray | None      = dict_['best_train_coefs'];    # Shape = (n_train, n_coefs).

        # Restore test_coefs into the existing learnable Parameter (do not replace the Parameter
        # object, since optimizers and downstream code expect it to remain a Parameter).
        loaded_test_coefs = dict_['test_coefs'];     # numpy.ndarray or torch.Tensor
        if isinstance(loaded_test_coefs, torch.Tensor):
            test_coefs_t = loaded_test_coefs.detach().to(dtype = torch.float32, device = self.device);
        else:
            test_coefs_t = torch.tensor(loaded_test_coefs, dtype = torch.float32, device = self.device);
        with torch.no_grad():
            self.test_coefs.data.copy_(test_coefs_t);
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


        # All done!
        return;
    
