# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

import  sys;
import  os; 
Physics_Path        : str   = os.path.abspath(os.path.join(os.path.dirname(__file__), "Physics"));
LD_Path             : str   = os.path.abspath(os.path.join(os.path.dirname(__file__), "LatentDynamics"));
EncoderDecoder_Path : str   = os.path.abspath(os.path.join(os.path.dirname(__file__), "EncoderDecoder"));
Utils_Path          : str   = os.path.abspath(os.path.join(os.path.dirname(__file__), "Utilities"));
sys.path.append(Physics_Path);
sys.path.append(LD_Path);
sys.path.append(EncoderDecoder_Path);
sys.path.append(Utils_Path);

import  logging;

import  torch;
import  numpy;
from    torch.optim                 import  Optimizer;
from    sklearn.gaussian_process    import  GaussianProcessRegressor;
from    scipy                       import  interpolate;
import  pickle;
import  time;

from    GaussianProcess             import  sample_coefs, fit_gps;
from    EncoderDecoder              import  EncoderDecoder;
from    Autoencoder                 import  Autoencoder;
from    Autoencoder_Pair            import  Autoencoder_Pair;
from    Timing                      import  Timer;
from    ParameterSpace              import  ParameterSpace;
from    Physics                     import  Physics;
from    LatentDynamics              import  LatentDynamics;
from    FiniteDifference            import  Derivative1_Order4, Derivative1_Order2_NonUniform;
from    Logging                     import  Log_Dictionary;
from    MoveOptimizer               import  Move_Optimizer_To_Device;

# Setup Logger
LOGGER : logging.Logger = logging.getLogger(__name__);



# -------------------------------------------------------------------------------------------------
# Trainer class
# -------------------------------------------------------------------------------------------------

class Trainer:
    # An n_Train element list. The i'th element is is an n_IC element list whose j'th element is a
    # numpy ndarray of shape (n_t(i), Frame_Shape) holding a sequence of samples of the j'th 
    # derivative of the FOM solution when we use the i'th combination of training values. 
    U_Train : list[list[torch.Tensor]]  = [];  

    # An n_Train element list whose i'th element is a torch.Tensor of shape (n_t(i)) whose j'th
    # element holds the time value for the j'th frame when we use the i'th combination of training 
    # parameters.
    t_Train : list[torch.Tensor]        = []; 
    
    # Same as U_Test, but used for the test set.
    U_Test  : list[list[torch.Tensor]]  = [];  

    # An n_Test element list whose i'th element is a torch.Tensor of shape (n_t(i)) whose j'th
    # element holds the time value for the j'th frame when we use the i'th combination of testing 
    # parameters.
    t_Test  : list[torch.Tensor]        = [];

    # number of IC's in the FOM solution.
    n_IC  : int;



    def __init__(self, 
                 physics            : Physics, 
                 encoder_decoder    : EncoderDecoder, 
                 latent_dynamics    : LatentDynamics, 
                 param_space        : ParameterSpace, 
                 config             : dict):
        """
        This class runs a full round of training. As input, it takes a EncoderDecoder object to 
        encode and decode FOM frames, a Physics object to recover FOM ICs + information on the 
        time discretization, a latent dynamics object, and a parameter space object (which holds 
        the testing and training sets of parameters).

        The "train" method runs the active learning training loop, computes the reconstruction and 
        SINDy loss, trains the GPs, and samples a new FOM data point.


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

        config: dict
            houses the Trainer settings. This should contain a 'trainer' sub-dictionary.

        
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Nothing!
        """
        
        # Checks.
        n_IC    : int               = latent_dynamics.n_IC;
        assert encoder_decoder.n_IC == n_IC, "encoder_decoder.n_IC = %d, n_IC = %d" % (encoder_decoder.n_IC, n_IC);
        assert physics.n_IC         == n_IC, "physics.n_IC = %d, n_IC = %d" % (physics.n_IC, n_IC);
        self.n_IC                   = n_IC;
        assert('trainer' in config), "config must contain a 'trainer' sub-dictionary";

        LOGGER.info("Initializing a Trainer object"); 
        Log_Dictionary(LOGGER = LOGGER, D = config, level = logging.INFO);

        # Fetch the trainer sub-dictionary.
        self.config = config;
        trainer_config : dict = config['trainer'];

        self.physics                        = physics;
        self.encoder_decoder                = encoder_decoder;
        self.latent_dynamics                = latent_dynamics;
        self.param_space                    = param_space;
        
        # Initialize a timer object. We will use this while training.
        self.timer                          = Timer();

        # Extract training/loss hyperparameters from the configuration file. 
        self.lr                     : float     = trainer_config.get('lr', '0.001');                # Learning rate for the optimizer.
        self.gradient_clip          : float     = trainer_config.get('gradient_clip', 15.0);        # Maximum allowable gradient magnitude; will rescale gradientsif exceeded.
        self.n_samples              : int       = trainer_config.get('n_samples', 20);              # Number of samples to draw per coefficient per combination of parameters
        self.p_rollout_init         : float     = trainer_config.get('p_rollout_init', 0.01);       # The proportion of the simulated we simulate forward when computing the rollout loss.
        self.rollout_update_freq    : float     = trainer_config.get('rollout_update_freq', 10);    # We increase p_rollout after this many iterations.
        self.dp_per_update          : float     = trainer_config.get('dp_per_update', 0.005);       # We increase p_rollout by this much each time we increase it.
        self.rollout_spline_order   : int       = trainer_config.get('rollout_spline_order', 1);    # The order of the spline used to interpolate the rollout targets.
        self.n_rollout_targets      : int       = trainer_config.get('n_rollout_targets', 3);       # Number of random target times sampled per rollable frame per epoch for the rollout loss.
        self.p_IC_rollout_init      : float     = trainer_config.get('p_IC_rollout_init', 0.01);    # The proportion of the simulation we simulate forward when computing the IC rollout loss.
        self.IC_rollout_update_freq : float     = trainer_config.get('IC_rollout_update_freq', 10); # We increase p_IC_rollout after this many iterations.
        self.IC_dp_per_update       : float     = trainer_config.get('IC_dp_per_update', 0.005);    # We increase p_IC_rollout by this much each time we increase it.
        self.max_p_rollout          : float     = trainer_config.get('max_p_rollout', 0.75);        # Maximum value p_rollout is allowed to reach (curriculum ceiling for the frame rollout loss).
        self.max_p_IC_rollout       : float     = trainer_config.get('max_p_IC_rollout', 1.0);      # Maximum value p_IC_rollout is allowed to reach (curriculum ceiling for the IC rollout loss).
        self.warmup_epochs          : int       = trainer_config.get('warmup_epochs', 40);          # We warmup the learning rate for this many epochs after greedy sampling.
        self.n_iter                 : int       = trainer_config['n_iter'];                         # Number of iterations for one train and greedy sampling
        self.max_iter               : int       = trainer_config['max_iter'];                       # We stop training if restart_iter goes above this number. 
        self.max_greedy_iter        : int       = trainer_config['max_greedy_iter'];                # We stop performing greedy sampling if restart_iter goes above this number.
        self.loss_weights           : dict      = trainer_config['loss_weights'];                   # A dictionary housing the weights of the various parts of the loss function.
        self.loss_types             : dict      = trainer_config['loss_types'];                     # A dictionary housing the type of loss function (MSE or MAE) for each part of the loss function.
        self.learnable_coefs        : bool      = trainer_config.get('learnable_coefs', True);      # If True, the latent dynamics coefficients are learnable parameters. If false, we compute them using Least Squares.

        # Optional normalization (training-only stats).
        # If enabled, we compute a single mean/std across ALL training trajectories (per IC),
        # then normalize both training + testing trajectories using these values.
        self.normalize          : bool          = bool(trainer_config.get('normalize', False));
        self.data_mean                : list[torch.Tensor] | None = None;   # per-IC scalar tensors (CPU)
        self.data_std                 : list[torch.Tensor] | None = None;   # per-IC scalar tensors (CPU)

        # Set the device to train on. We default to cpu.
        device = trainer_config['device'] if 'device' in trainer_config else 'cpu';
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
        self.test_coefs : torch.Tensor = torch.nn.parameter.Parameter(torch.zeros(self.param_space.n_test(), self.latent_dynamics.n_coefs, dtype = torch.float32, device = self.device, requires_grad = True));

        # Set up the optimizer and loss function.
        LOGGER.info("Setting up the optimizer with a learning rate of %f" % (self.lr));
        self.optimizer          : Optimizer = torch.optim.Adam(list(encoder_decoder.parameters()) + [self.test_coefs], lr = self.lr, weight_decay = 1.0e-5);
        self.MSE                            = torch.nn.MSELoss(reduction = 'mean');
        self.MAE                            = torch.nn.L1Loss(reduction = 'mean');

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

        # Set up variables to aide checkpointing
        self.best_train_coefs   = None;
        self.best_epoch         = -1;
        self.restart_iter       = 0;                # Iteration number at the end of the last training period
        
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
        
        Arguments:
        ----------
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
        
        Arguments:
        ----------
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
    # Training Loop.
    # ---------------------------------------------------------------------------------------------


    def train(self, reset_optim : bool = True) -> None:
        """
        Runs a round of training on the encoder_decoder.

        
        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        reset_optim : bool
            If True, we re-initialize self's optimizer before training. 



        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Nothing!
        """
        
        # Make sure we have at least one training data point.
        assert len(self.U_Train) > 0, "len(self.U_Train) = %d" % len(self.U_Train);
        assert len(self.U_Train) == self.param_space.n_train(), "len(self.U_Train) = %d, self.param_space.n_train() = %d" % (len(self.U_Train), self.param_space.n_train());


        # Reset optimizer, if desirable. 
        if(reset_optim == True): self._reset_optimizer();



        # -------------------------------------------------------------------------------------
        # Setup. 

        # Fetch parameters. Note that p_rollout and p_IC_rollout can be negative.
        # IMPORTANT: Calculate rollout proportions using epochs within CURRENT round (not accumulated restart_iter).
        # This ensures rollout starts small after each greedy sampling and gradually increases.
        n_train                 : int               = self.param_space.n_train();
        epochs_in_round         : int               = 0;  # Will be updated each iteration
        p_rollout               : float             = min(self.max_p_rollout,    self.p_rollout_init    + self.dp_per_update   *(epochs_in_round//self.rollout_update_freq));
        p_IC_rollout            : float             = min(self.max_p_IC_rollout, self.p_IC_rollout_init + self.IC_dp_per_update*(epochs_in_round//self.IC_rollout_update_freq));
        best_loss               : float             = numpy.inf;                    # Stores the lowest loss we get in this round of training.

        # Map everything to self's device.
        device                  : str                       = self.device;
        encoder_decoder_device  : EncoderDecoder            = self.encoder_decoder.to(device);
        
        # Determine encoder_decoder type once (assumed constant throughout training)
        is_autoencoder_pair     : bool                      = isinstance(encoder_decoder_device, Autoencoder_Pair);
        
        U_Train_device          : list[list[torch.Tensor]]  = [];
        t_Train_device          : list[torch.Tensor]        = [];
        for i in range(n_train):
            t_Train_device.append(self.t_Train[i].to(device));
            
            ith_U_Train_device  : list[torch.Tensor] = [];
            for j in range(self.n_IC):
                ith_U_Train_device.append(self.U_Train[i][j].to(device));
            U_Train_device.append(ith_U_Train_device);

        # Make sure the checkpoints and results directories exist.
        from pathlib import Path
        Path(self.path_checkpoint).mkdir(   parents = True, exist_ok = True);
        Path(self.path_results).mkdir(      parents = True, exist_ok = True);

        # Rollout setup
        if(self.loss_weights['rollout'] > 0 and p_rollout > 0):
            self.timer.start("Rollout Setup");

            t_Grid_rollout, n_rollout_ICs, U_Target_Rollout_Trajectory = self._rollout_setup(
                                                                            t            = t_Train_device, 
                                                                            U            = U_Train_device, 
                                                                            p_rollout    = p_rollout);
            self.timer.end("Rollout Setup");

        # IC rollout setup
        if(self.loss_weights['IC_rollout'] > 0 and p_IC_rollout > 0):
            self.timer.start("IC Rollout Setup");

            t_Grid_IC_rollout, n_IC_rollout_frames, U_IC_Rollout_Targets = self._IC_rollout_setup(  t            = t_Train_device, 
                                                                                                    p_IC_rollout = p_IC_rollout);
            self.timer.end("IC Rollout Setup"); 

        # If we are learning the latent dynamics coefficients, then we need to determine 
        # which combinations of parameters are in the training set. Specifically, each 
        # element of the train space should also be in the test space. We need to figure out 
        # the index of each train space element within the test space.
        train_coefs_list : list[torch.Tensor] = [];
        if(self.learnable_coefs == True):
            for i in range(n_train):
                ith_train_in_test : bool = False;
                for j in range(self.param_space.n_test()):
                    if(numpy.allclose(self.param_space.test_space[j, :], self.param_space.train_space[i, :], rtol = 1e-12, atol = 1e-14)):
                        train_coefs_list.append(self.test_coefs[j, :]);
                        ith_train_in_test = True;
                        break;

                # Make sure we found the training combination of parameters in the test space.
                assert(ith_train_in_test == True);


        # -----------------------------------------------------------------------------------------
        # Initialize loss tracking
        
        # Set up filename for loss_by_param.
        # NOTE: must match the filename we save at the end of training so restarts work.
        base_filename       : str = self.config['physics']['type'];
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

        next_iter   : int = min(self.restart_iter + self.n_iter, self.max_iter);
        LOGGER.info("Training for %d epochs (starting at %d, going to %d) with %d parameters" % (next_iter - self.restart_iter, self.restart_iter, next_iter, n_train));
        
        for iter in range(self.restart_iter, next_iter):
            self.timer.start("train_step");
            LOGGER.debug("=" * 80);
            LOGGER.debug("Starting training iteration %d/%d" % (iter + 1, next_iter));


            # -------------------------------------------------------------------------------------
            # Warmup the learning rate for the first few epochs after greedy sampling.
            # NOTE: epochs_in_round will be recalculated later for rollout updates.

            epochs_in_round     : int = iter - self.restart_iter;  # Progress within current training round
            if self.warmup_epochs > 0 and epochs_in_round < self.warmup_epochs:
                # Reduce LR for warmup period
                warmup_scale = 0.1 + 0.9 * (float(epochs_in_round) / float(self.warmup_epochs));
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr * warmup_scale;
                LOGGER.info("Warmup: LR scaled to %.6f (epoch %d/%d in round)" % (self.lr * warmup_scale, epochs_in_round, next_iter - self.restart_iter));
            else:
                # Restore full LR
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr;


            # -------------------------------------------------------------------------------------
            # Check if we need to update p_rollout. If so, then we also need to update 
            # t_Grid_rollout, n_rollout_ICs, and U_Target_Rollout_Trajectory
            # NOTE: Use epochs_in_round (not iter) to reset rollout progression each training round
            
            if(self.loss_weights['rollout'] > 0 and epochs_in_round > 0 and ((epochs_in_round % self.rollout_update_freq) == 0)):
                self.timer.start("Rollout Setup");
                LOGGER.debug("Rollout Setup");

                # Recalculate p_rollout based on progress within current round
                p_rollout   = min(self.max_p_rollout, self.p_rollout_init + self.dp_per_update*(epochs_in_round//self.rollout_update_freq));

                LOGGER.info("p_rollout is now %f (epoch %d/%d in current round)" % (p_rollout, epochs_in_round, next_iter - self.restart_iter));

                if(p_rollout > 0):
                    t_Grid_rollout, n_rollout_ICs, U_Target_Rollout_Trajectory = self._rollout_setup(
                                                                                    t            = t_Train_device, 
                                                                                    U            = U_Train_device, 
                                                                                    p_rollout    = p_rollout);
                
                LOGGER.debug("Rollout Setup complete");
                self.timer.end("Rollout Setup");


            # -------------------------------------------------------------------------------------
            # Check if we need to update IC rollout parameters
            # NOTE: Use epochs_in_round (not iter) to reset IC rollout progression each training round

            if(self.loss_weights['IC_rollout'] > 0 and epochs_in_round > 0 and ((epochs_in_round % self.IC_rollout_update_freq) == 0)):
                self.timer.start("IC Rollout Setup");

                # Recalculate p_IC_rollout based on progress within current round
                p_IC_rollout   = min(self.max_p_IC_rollout, self.p_IC_rollout_init + self.IC_dp_per_update*(epochs_in_round//self.IC_rollout_update_freq));

                LOGGER.info("p_IC_rollout is now %f (epoch %d/%d in current round)" % (p_IC_rollout, epochs_in_round, next_iter - self.restart_iter));

                # Setup IC rollout time grids and targets
                if(p_IC_rollout > 0):
                    t_Grid_IC_rollout, n_IC_rollout_frames, U_IC_Rollout_Targets = self._IC_rollout_setup(  t            = t_Train_device, 
                                                                                                            p_IC_rollout = p_IC_rollout);
                
                self.timer.end("IC Rollout Setup"); 


            # -------------------------------------------------------------------------------------
            # Zero gradients.
            
            self.optimizer.zero_grad();
            LOGGER.debug("Zeroed gradients for iteration %d" % (iter + 1));


            # -------------------------------------------------------------------------------------
            # Compute losses
            
            if not is_autoencoder_pair:
                # Initialize losses. 
                loss_recon              : torch.Tensor = torch.zeros(1, dtype = torch.float32, device = device);
                loss_LD                 : torch.Tensor = torch.zeros(1, dtype = torch.float32, device = device);
                loss_stab               : torch.Tensor = torch.zeros(1, dtype = torch.float32, device = device);
                loss_rollout_FOM        : torch.Tensor = torch.zeros(1, dtype = torch.float32, device = device);
                loss_rollout_ROM        : torch.Tensor = torch.zeros(1, dtype = torch.float32, device = device);
                loss_IC_rollout_FOM     : torch.Tensor = torch.zeros(1, dtype = torch.float32, device = device);
                loss_IC_rollout_ROM     : torch.Tensor = torch.zeros(1, dtype = torch.float32, device = device);

                # Setup. 
                Latent_States           : list[list[torch.Tensor]]  = [];       # len = n_train. i'th element is 1 element list of (n_t_i, n_z) arrays.
                ROM_Rollout_ICs         : list[list[torch.Tensor]]  = [];       # len = n_train. i'th element is 1 element list of (n_rollout_ICs[i], n_z) arrays.
                FOM_Rollout_Targets     : list[list[torch.Tensor]]  = [];       # len = n_train. i'th element is 1 element list of (n_rollout_ICs[i], physics.Frame_Shape) arrays holding the FOM rollout targets.
                ROM_Rollout_Targets     : list[list[torch.Tensor]]  = [];       # len = n_train. i'th element is 1 element list of (n_rollout_ICs[i], n_z) arrays holding the ROM rollout targets.
                Rollout_Indices         : list[int]                 = [];       # len = n_train. i'th element is an array of shape (n_rollout_ICs[i]) specifying the indices (in rollout trajectories) of the frames we use as rollout targets.
                
                # Cycle through the combinations of parameter values
                for i in range(n_train):
                    # Setup. 
                    U_i         : torch.Tensor  = U_Train_device[i][0];
                    t_Grid_i    : torch.Tensor  = t_Train_device[i];
                    n_t_i       : int           = t_Grid_i.shape[0];


                    # -----------------------------------------------------------------------------
                    # Forward pass

                    self.timer.start("Forward Pass");
                    LOGGER.debug("Forward Pass (Autoencoder) - start for parameter combination %d" % i);

                    # Run the forward pass. This results in an n_train element list whose i'th 
                    # element is a 1 element list whose only element is a tensor of shape 
                    # (n_t(i), physics.Frame_Shape) whose [k, ...] slice holds our prediction for 
                    # the FOM solution at time t_Grid[i][k] when we use the i'th combination of 
                    # parameter values. Here, n_t(i) is the number of time steps in the solution 
                    # for the i'th combination of parameter values. 
                    Z_i         : torch.Tensor  = encoder_decoder_device.Encode(U_i)[0];
                    
                    # Log latent state statistics to diagnose potential autoencoder collapse
                    if iter % 100 == 0 or iter == self.restart_iter:  # Log every 100 iters and first iter
                        LOGGER.info("Epoch %d, Param %d: Z shape=%s, min=%.6e, max=%.6e, mean=%.6e, std=%.6e, device=%s" % (
                            iter + 1, i, str(Z_i.shape), 
                            float(Z_i.min().item()), float(Z_i.max().item()), 
                            float(Z_i.mean().item()), float(Z_i.std().item()),
                            str(Z_i.device)));
                    
                    Latent_States.append([Z_i]);
                    U_Pred_i    : torch.Tensor  = encoder_decoder_device.Decode(Z_i)[0];

                    LOGGER.debug("Forward Pass (Autoencoder) - complete for parameter combination %d" % i);
                    self.timer.end("Forward Pass");


                    # ----------------------------------------------------------------------------
                    # Reconstruction loss

                    if(self.loss_weights['recon'] > 0):
                        self.timer.start("Reconstruction Loss");
                        LOGGER.debug("Reconstruction Loss (Autoencoder) - start for parameter combination %d" % i);

                        # Reconstruction residual (data is either physical units or normalized).
                        diff = (U_i - U_Pred_i);
                        
                        # Compute loss from normalized difference
                        if(self.loss_types['recon'] == "MSE"):
                            recon_loss_ith_param = torch.mean(diff**2);
                        elif(self.loss_types['recon'] == "MAE"):
                            recon_loss_ith_param = torch.mean(torch.abs(diff));
                        else:
                            raise ValueError("Invalid reconstruction loss type: %s" % self.loss_types['recon']);
                        
                        loss_recon += recon_loss_ith_param;
                        
                        # Store recon loss for this parameter combination.
                        ith_param_tuple = tuple(self.param_space.train_space[i, :]);
                        self._store_loss_by_param('recon', ith_param_tuple, iter + 1, recon_loss_ith_param.item());
                        
                        LOGGER.debug("Reconstruction Loss (Autoencoder) - complete for parameter combination %d" % i);
                        self.timer.end("Reconstruction Loss");


                    # ----------------------------------------------------------------------------
                    # Setup Rollout losses.

                    if(self.loss_weights['rollout'] > 0 and p_rollout > 0):
                        self.timer.start("Rollout Setup");
                        LOGGER.debug("Rollout Setup (Autoencoder) - start for parameter combination %d" % i);

                        # Select the latent states we want to use as initial conditions for the i'th 
                        # combination of parameter values. This should be the first 
                        # n_rollout_ICs[i] frames (n_rollout_ICs[i] is computed such that if we 
                        # simulate the first n_rollout_ICs[i] frames, the final times are less than 
                        # the final time for this combination of parameter values. Each element of 
                        # ROM_Rollout_IC is a 1 element list of torch.Tensor objects of shape 
                        # (n_rollout_ICs[i], n_z).
                        ROM_Rollout_ICs.append([Z_i[:n_rollout_ICs[i], :]]);

                        # For each rollable frame (IC), sample n_rollout_targets target step indices
                        # uniformly from [1, n_steps). We start at 1 (not 0) because index 0
                        # corresponds to zero-duration rollout: the predicted state equals the IC
                        # itself, so the loss reduces to reconstruction error rather than testing
                        # forward dynamics. Shape: (n_rollout_ICs[i], n_rollout_targets).
                        n_steps_i           : int           = t_Grid_rollout[i].shape[0];
                        Rollout_Indices_i   : numpy.ndarray = numpy.random.randint(1, n_steps_i, (n_rollout_ICs[i], self.n_rollout_targets));
                        Rollout_Indices.append(Rollout_Indices_i);

                        # Fetch the corresponding FOM targets for the i'th combination of parameter 
                        # values. For each time-derivative component j, we build a flat array of shape 
                        # (n_rollout_ICs[i] * n_rollout_targets, Frame_Shape) by iterating over every 
                        # (IC index k, target index m) pair and looking up the pre-interpolated FOM 
                        # frame at that step. Flattening k and m together lets us encode all targets 
                        # in a single batch call below.
                        FOM_Rollout_Targets_i : list[torch.Tensor] = [];
                        for j in range(self.n_IC):
                            FOM_Rollout_Targets_ij : numpy.ndarray = numpy.empty(
                                (n_rollout_ICs[i] * self.n_rollout_targets,) + tuple(self.physics.Frame_Shape), 
                                dtype = numpy.float32);
                            for k in range(n_rollout_ICs[i]):
                                for m in range(self.n_rollout_targets):
                                    FOM_Rollout_Targets_ij[k * self.n_rollout_targets + m, ...] = \
                                        U_Target_Rollout_Trajectory[i][j][k, Rollout_Indices_i[k, m], ...];
                            FOM_Rollout_Targets_i.append(torch.tensor(FOM_Rollout_Targets_ij, dtype = torch.float32, device = device));
                        FOM_Rollout_Targets.append(FOM_Rollout_Targets_i);

                        # Encode all (n_rollout_ICs * n_rollout_targets) FOM targets in one batch.
                        # Result shape: (n_rollout_ICs[i] * n_rollout_targets, n_z).
                        ROM_Rollout_Targets.append([encoder_decoder_device.Encode(*FOM_Rollout_Targets_i)[0]]);
                    
                        LOGGER.debug("Rollout Setup (Autoencoder) - complete for parameter combination %d" % i);
                        self.timer.end("Rollout Setup");

                # Store total recon loss.
                self._store_total_loss('recon', iter + 1, loss_recon.item());


                # --------------------------------------------------------------------------------
                # Latent Dynamics, Stability losses

                self.timer.start("Calibration");

                # Compute the latent dynamics and stability losses. Also fetch the current latent
                # dynamics coefficients for each training point. The latter is stored in a 2d array 
                # called "train_coefs" of shape (n_train, n_coefs), where n_train = number of 
                # training combinations of parameters and n_coefs denotes the number of 
                # coefficients in the latent dynamics model. 
                train_coefs, loss_LD_list, loss_stab_list = self.latent_dynamics.calibrate(
                                                                Latent_States    = Latent_States, 
                                                                t_Grid           = t_Train_device,
                                                                input_coefs      = train_coefs_list,
                                                                loss_type        = self.loss_types['LD'],
                                                                params           = self.param_space.train_space);

                # Log coefficient statistics to diagnose constant dynamics issue
                if iter % 100 == 0 or iter == self.restart_iter:  # Log every 100 iters and first iter
                    LOGGER.info("Epoch %d: Coefs shape=%s, min=%.6e, max=%.6e, mean=%.6e, std=%.6e, abs_mean=%.6e" % (
                        iter + 1, str(train_coefs.shape),
                        float(train_coefs.min().item()), float(train_coefs.max().item()),
                        float(train_coefs.mean().item()), float(train_coefs.std().item()),
                        float(torch.abs(train_coefs).mean().item())));

                # Append the LD and stability losses to loss_by_param.
                for i in range(n_train):
                    param_tuple = tuple(self.param_space.train_space[i, :]);
                    self._store_loss_by_param('LD',   param_tuple, iter + 1, loss_LD_list[i].item());
                    self._store_loss_by_param('stab', param_tuple, iter + 1, loss_stab_list[i].item());

                # Compute the total loss.
                loss_LD   = torch.sum(torch.stack(loss_LD_list));
                loss_stab = torch.sum(torch.stack(loss_stab_list));

                # Append the total loss to loss_by_param.
                self._store_total_loss('LD', iter + 1, loss_LD.item());
                self._store_total_loss('stab', iter + 1, loss_stab.item());

                self.timer.end("Calibration");


                # ---------------------------------------------------------------------------------
                # Rollout loss. Note that we need the coefficients before we can compute this.
                
                if(self.loss_weights['rollout'] > 0 and p_rollout > 0):
                    self.timer.start("Rollout Loss");
                    LOGGER.debug("Rollout Loss (Autoencoder) - start");

                    # Simulate the frames forward in time. This should return an n_param element list
                    # whose i'th element is a 1 element list whose only element has shape (n_t_i, 
                    # n_rollout_ICs[i], n_z) whose p, q, r element of the should hold the r'th 
                    # component of the p'th frame of the j'th time derivative of the solution
                    # when we use the p'th initial condition for the i'th combination of parameter 
                    # values.
                    #
                    # Note that the latent dynamics are autonomous. Further, because we are simulating 
                    # each IC for the same amount of time, the specific values of the time are
                    # irreverent. The simulate function exploits this by solving one big IVP for each 
                    # combination of parameter values, rather than n(i) smaller ones.                         
                    ROM_Predicted_Rollout_Trajectories  : list[list[torch.Tensor]]  = self.latent_dynamics.simulate(  
                                                                                            coefs   = train_coefs, 
                                                                                            IC      = ROM_Rollout_ICs, 
                                                                                            t_Grid  = t_Grid_rollout,
                                                                                            params  = self.param_space.train_space);            

                    # Now cycle through the training examples.
                    for i in range(n_train):
                        # Fetch the predicted rollout trajectory for the i'th parameter combination.
                        # Shape: (n_steps, n_rollout_ICs[i], n_z).
                        ROM_Predicted_Rollout_Trajectories_i : torch.Tensor = ROM_Predicted_Rollout_Trajectories[i][0];

                        # Gather the predicted latent states at the randomly sampled target steps.
                        # Rollout_Indices[i] has shape (n_rollout_ICs, n_rollout_targets); each entry
                        # is a step index in [1, n_steps).
                        #
                        # Naive equivalent (Python loops):
                        #   for k in range(n_rollout_ICs[i]):
                        #       for m in range(self.n_rollout_targets):
                        #           out[k, m, :] = ROM_Predicted_Rollout_Trajectories_i[Rollout_Indices[i][k,m], k, :]
                        #
                        # Vectorized via torch.gather:
                        #   1. Permute trajectory to (n_rollout_ICs, n_steps, n_z) so ICs are the
                        #      leading dim instead of steps.
                        #   2. Expand the index array from (n_rollout_ICs, n_rollout_targets) to
                        #      (n_rollout_ICs, n_rollout_targets, n_z) by repeating along the last dim.
                        #   3. torch.gather(dim=1) picks, for every (IC, target, latent-component)
                        #      triple, the single requested step — all in one fused GPU kernel call
                        #      rather than a Python loop over n_rollout_ICs iterations.
                        n_z_i       : int           = ROM_Predicted_Rollout_Trajectories_i.shape[2];
                        traj_i      : torch.Tensor  = ROM_Predicted_Rollout_Trajectories_i.permute(1, 0, 2);                   # (n_rollout_ICs, n_steps, n_z)
                        idx_i       : torch.Tensor  = torch.tensor(Rollout_Indices[i], dtype = torch.long, device = device);   # (n_rollout_ICs, n_rollout_targets)
                        idx_exp     : torch.Tensor  = idx_i.unsqueeze(-1).expand(-1, -1, n_z_i);                               # (n_rollout_ICs, n_rollout_targets, n_z)
                        ROM_Rollout_Predict_i       : torch.Tensor = torch.gather(traj_i, dim = 1, index = idx_exp);           # (n_rollout_ICs, n_rollout_targets, n_z)

                        # Flatten (n_rollout_ICs, n_rollout_targets) into one batch dimension so all
                        # targets can be decoded together and compared against FOM_Rollout_Targets,
                        # which was stored in the same (k * n_rollout_targets + m) order during setup.
                        ROM_Rollout_Predict_i_flat  : torch.Tensor = ROM_Rollout_Predict_i.reshape(-1, n_z_i);                 # (n_rollout_ICs * n_rollout_targets, n_z)

                        # Fetch the corresponding encoded targets (already flat from setup).
                        ROM_Rollout_Targets_i       : torch.Tensor = ROM_Rollout_Targets[i][0];    # shape = (n_rollout_ICs[i] * n_rollout_targets, n_z)

                        # Decode the latent predictions to get FOM predictions.
                        FOM_Rollout_Predict_i       : torch.Tensor = encoder_decoder_device.Decode(ROM_Rollout_Predict_i_flat)[0];
                    
                        # Get the corresponding FOM targets.
                        FOM_Rollout_Target_i        : torch.Tensor = FOM_Rollout_Targets[i][0];    # shape = (n_rollout_ICs[i] * n_rollout_targets, *physics.Frame_Shape)
                    
                        # Compute differences once
                        diff_ROM = ROM_Rollout_Targets_i - ROM_Rollout_Predict_i_flat;
                        diff_FOM = (FOM_Rollout_Predict_i - FOM_Rollout_Target_i);
                        
                        # Compute losses from normalized differences
                        if(self.loss_types['rollout'] == "MSE"):
                            loss_rollout_ROM_ith_param = torch.mean(diff_ROM**2);
                            loss_rollout_FOM_ith_param = torch.mean(diff_FOM**2);
                        elif(self.loss_types['rollout'] == "MAE"):
                            loss_rollout_ROM_ith_param = torch.mean(torch.abs(diff_ROM));
                            loss_rollout_FOM_ith_param = torch.mean(torch.abs(diff_FOM));
                        else:
                            loss_rollout_ROM_ith_param = torch.zeros(1, dtype = torch.float32, device = device);
                            loss_rollout_FOM_ith_param = torch.zeros(1, dtype = torch.float32, device = device);
                        
                        loss_rollout_ROM += loss_rollout_ROM_ith_param;
                        loss_rollout_FOM += loss_rollout_FOM_ith_param;
                        
                        # Store per-parameter-combination loss
                        param_tuple = tuple(self.param_space.train_space[i, :]);
                        self._store_loss_by_param('rollout_ROM', param_tuple, iter + 1, loss_rollout_ROM_ith_param.item());
                        self._store_loss_by_param('rollout_FOM', param_tuple, iter + 1, loss_rollout_FOM_ith_param.item());
                        
                    # Store total rollout loss.
                    self._store_total_loss('rollout_ROM', iter + 1, loss_rollout_ROM.item());
                    self._store_total_loss('rollout_FOM', iter + 1, loss_rollout_FOM.item());

                    LOGGER.debug("Rollout Loss (Autoencoder) - complete");
                    self.timer.end("Rollout Loss");


                # --------------------------------------------------------------------------------
                # IC Rollout loss. This simulates forward from the FOM initial conditions.

                # Cycle through the training examples for IC rollout
                if(self.loss_weights['IC_rollout'] > 0 and p_IC_rollout > 0):
                    self.timer.start("IC Rollout Loss");
                    LOGGER.debug("IC Rollout Loss (Autoencoder) - start");

                    for i in range(n_train):
                        # Fetch the FOM initial conditions for this combination of parameters
                        param_i           : numpy.ndarray             = self.param_space.train_space[i, :]; 
                        FOM_IC_i          : list[numpy.ndarray]       = self.physics.initial_condition(param_i);    # len = 1

                        # Convert to tensors and reshape for encoding
                        U_IC_i            : torch.Tensor              = torch.tensor(FOM_IC_i[0], dtype = torch.float32, device = device).reshape((1,) + FOM_IC_i[0].shape);
                        if self.has_normalization():
                            U_IC_i = self.normalize_tensor(U_IC_i, 0);
                        
                        # Encode the FOM initial conditions
                        Z_IC_i : torch.Tensor = encoder_decoder_device.Encode(U_IC_i)[0];
                        
                        # Get the coefficients for this combination of parameters
                        train_coef_i            : torch.Tensor              = train_coefs[i, :].reshape(1, -1);
                        
                        # Simulate the latent dynamics forward in time
                        Z_IC_Rollout_i    : list[list[torch.Tensor]]  = self.latent_dynamics.simulate(  coefs   = train_coef_i, 
                                                                                                        IC      = [[Z_IC_i]], 
                                                                                                        t_Grid  = [t_Grid_IC_rollout[i]], 
                                                                                                        params  = param_i.reshape(1, -1));
                        
                        # Extract the predicted trajectory, remove the singleton dimension
                        Z_IC_Predict_i      : torch.Tensor              = Z_IC_Rollout_i[0][0].squeeze(1);    # shape = (n_t_IC_rollout[i], n_z)

                        # Decode the predicted trajectory to get FOM predictions
                        U_IC_Predict_i      : torch.Tensor              = encoder_decoder_device.Decode(Z_IC_Predict_i)[0];
                        
                        # Get the corresponding FOM targets
                        U_IC_Target_i       : list[torch.Tensor]        = U_IC_Rollout_Targets[i][0];         # shape = (n_t_IC_rollout[i], physics.Frame_Shape)

                        # Encode the FOM targets for latent space comparison
                        Z_IC_Target_i : torch.Tensor = encoder_decoder_device.Encode(U_IC_Target_i)[0];

                        # Compute differences once
                        diff_ROM = Z_IC_Target_i - Z_IC_Predict_i;
                        diff_FOM = (U_IC_Predict_i - U_IC_Target_i);
                        
                        # Compute losses from normalized differences
                        if(self.loss_types['IC_rollout'] == "MSE"):
                            loss_IC_rollout_ROM_ith_param = torch.mean(diff_ROM**2);
                            loss_IC_rollout_FOM_ith_param = torch.mean(diff_FOM**2);
                        elif(self.loss_types['IC_rollout'] == "MAE"):
                            loss_IC_rollout_ROM_ith_param = torch.mean(torch.abs(diff_ROM));
                            loss_IC_rollout_FOM_ith_param = torch.mean(torch.abs(diff_FOM));
                        else:
                            loss_IC_rollout_ROM_ith_param = torch.zeros(1, dtype = torch.float32, device = device);
                            loss_IC_rollout_FOM_ith_param = torch.zeros(1, dtype = torch.float32, device = device);
                        
                        loss_IC_rollout_ROM += loss_IC_rollout_ROM_ith_param;
                        loss_IC_rollout_FOM += loss_IC_rollout_FOM_ith_param;
                        
                        # Store per-parameter-combination loss
                        param_tuple = tuple(self.param_space.train_space[i, :]);
                        self._store_loss_by_param('IC_rollout_ROM', param_tuple, iter + 1, loss_IC_rollout_ROM_ith_param.item());
                        self._store_loss_by_param('IC_rollout_FOM', param_tuple, iter + 1, loss_IC_rollout_FOM_ith_param.item());

                    # Store total IC rollout loss.
                    self._store_total_loss('IC_rollout_ROM', iter + 1, loss_IC_rollout_ROM.item());
                    self._store_total_loss('IC_rollout_FOM', iter + 1, loss_IC_rollout_FOM.item());

                    LOGGER.debug("IC Rollout Loss (Autoencoder) - complete");
                    self.timer.end("IC Rollout Loss");


                # --------------------------------------------------------------------------------
                # Total loss

                loss_rollout    : torch.Tensor  = loss_rollout_ROM    + loss_rollout_FOM;
                loss_IC_rollout : torch.Tensor  = loss_IC_rollout_ROM + loss_IC_rollout_FOM;


                # Compute the final loss.
                LOGGER.debug("Computing total loss (Autoencoder)");
                loss = (self.loss_weights['recon']      * loss_recon + 
                        self.loss_weights['LD']         * loss_LD + 
                        self.loss_weights['rollout']    * loss_rollout + 
                        self.loss_weights['IC_rollout'] * loss_IC_rollout + 
                        self.loss_weights['stab']       * loss_stab);
                self._store_total_loss('total', iter + 1, loss.item());
                LOGGER.debug("Total loss (Autoencoder) computed: %f" % loss.item());



            else:  # is_autoencoder_pair
                # Initialize losses. 
                loss_LD                 : torch.Tensor = torch.zeros(1, dtype = torch.float32, device = device);
                loss_stab               : torch.Tensor = torch.zeros(1, dtype = torch.float32, device = device);
                loss_recon_D            : torch.Tensor = torch.zeros(1, dtype = torch.float32, device = device);
                loss_recon_V            : torch.Tensor = torch.zeros(1, dtype = torch.float32, device = device);
                loss_consistency_Z      : torch.Tensor = torch.zeros(1, dtype = torch.float32, device = device);
                loss_consistency_U      : torch.Tensor = torch.zeros(1, dtype = torch.float32, device = device);
                loss_chain_rule_U       : torch.Tensor = torch.zeros(1, dtype = torch.float32, device = device);
                loss_chain_rule_Z       : torch.Tensor = torch.zeros(1, dtype = torch.float32, device = device);
                loss_rollout_FOM_D      : torch.Tensor = torch.zeros(1, dtype = torch.float32, device = device);
                loss_rollout_FOM_V      : torch.Tensor = torch.zeros(1, dtype = torch.float32, device = device);
                loss_rollout_ROM_D      : torch.Tensor = torch.zeros(1, dtype = torch.float32, device = device);
                loss_rollout_ROM_V      : torch.Tensor = torch.zeros(1, dtype = torch.float32, device = device);
                loss_IC_rollout_D       : torch.Tensor = torch.zeros(1, dtype = torch.float32, device = device);
                loss_IC_rollout_V       : torch.Tensor = torch.zeros(1, dtype = torch.float32, device = device);
                loss_IC_rollout_Z_D     : torch.Tensor = torch.zeros(1, dtype = torch.float32, device = device);
                loss_IC_rollout_Z_V     : torch.Tensor = torch.zeros(1, dtype = torch.float32, device = device);
                # Setup. 
                Latent_States       : list[list[torch.Tensor]]  = [];       # len = n_train. i'th element is 2 element list of (n_t_i, n_z) arrays.

                ROM_Rollout_ICs     : list[list[torch.Tensor]]  = [];       # len = n_train. i'th element is 2 element list of (n_rollout_ICs[i], n_z) arrays.
                FOM_Rollout_Targets : list[list[torch.Tensor]]  = [];       # len = n_train. i'th element is 2 element list of (n_rollout_ICs[i], physics.Frame_Shape) arrays holding the FOM rollout targets.
                ROM_Rollout_Targets : list[list[torch.Tensor]]  = [];       # len = n_train. i'th element is 2 element list of (n_rollout_ICs[i], n_z) arrays holding the ROM rollout targets.
                Rollout_Indices     : list[int]                 = [];       # len = n_train. i'th element is an array of shape (n_rollout_ICs[i]) specifying the indices (in rollout trajectories) of the frames we use as rollout targets.
                


                # Cycle through the combinations of parameter values.
                for i in range(n_train):
                    # Setup. 
                    D_i         : torch.Tensor  = U_Train_device[i][0];
                    V_i         : torch.Tensor  = U_Train_device[i][1];

                    t_Grid_i    : torch.Tensor  = t_Train_device[i];
                    n_t_i       : int           = t_Grid_i.shape[0];


                    # -----------------------------------------------------------------------------
                    # Forward pass

                    self.timer.start("Forward Pass");
                    LOGGER.debug("Forward Pass (Autoencoder_Pair) - start for parameter combination %d" % i);

                    # Run the forward pass. This results in an n_train element list whose i'th 
                    # element is a 2 element list whose j'th element is a tensor of shape 
                    # (n_t(i), physics.Frame_Shape) whose [k, ...] slice holds our prediction for 
                    # the j'th time derivative of the FOM solution at time t_Grid[i][k] when we use 
                    # the i'th combination of parameter values. Here, n_t(i) is the number of time 
                    # steps in the solution for the i'th combination of parameter values. 
                    Z_i     : list[torch.Tensor]        = list(encoder_decoder_device.Encode(*U_Train_device[i]));
                    Z_D_i   : torch.Tensor              = Z_i[0];       # shape (n_t(i), n_z)
                    Z_V_i   : torch.Tensor              = Z_i[1];       # shape (n_t(i), n_z)
                    
                    # Log latent state statistics to diagnose potential autoencoder collapse
                    if iter % 100 == 0 or iter == self.restart_iter:  # Log every 100 iters and first iter
                        LOGGER.info("Epoch %d, Param %d: Z_D shape=%s, min=%.6e, max=%.6e, mean=%.6e, std=%.6e, device=%s" % (
                            iter + 1, i, str(Z_D_i.shape), 
                            float(Z_D_i.min().item()), float(Z_D_i.max().item()), 
                            float(Z_D_i.mean().item()), float(Z_D_i.std().item()),
                            str(Z_D_i.device)));
                        LOGGER.info("Epoch %d, Param %d: Z_V shape=%s, min=%.6e, max=%.6e, mean=%.6e, std=%.6e, device=%s" % (
                            iter + 1, i, str(Z_V_i.shape), 
                            float(Z_V_i.min().item()), float(Z_V_i.max().item()), 
                            float(Z_V_i.mean().item()), float(Z_V_i.std().item()),
                            str(Z_V_i.device)));
                    
                    Latent_States.append(Z_i);

                    U_Pred_i    : list[torch.Tensor]    = list(encoder_decoder_device.Decode(*Z_i));
                    D_Pred_i    : torch.Tensor          = U_Pred_i[0];  # shape = (n_t(i), physics.Frame_Shape)
                    V_Pred_i    : torch.Tensor          = U_Pred_i[1];  # shape = (n_t(i), physics.Frame_Shape)

                    LOGGER.debug("Forward Pass (Autoencoder_Pair) - complete for parameter combination %d" % i);
                    self.timer.end("Forward Pass");


                    # ----------------------------------------------------------------------------
                    # Reconstruction loss

                    if(self.loss_weights['recon'] > 0):
                        self.timer.start("Reconstruction Loss");
                        LOGGER.debug("Reconstruction Loss (Autoencoder_Pair) - start for parameter combination %d" % i);

                        # Compute differences once
                        diff_D = (D_i - D_Pred_i);
                        diff_V = (V_i - V_Pred_i);
                        
                        # Compute losses from normalized differences
                        if(self.loss_types['recon'] == "MSE"):
                            recon_D_loss_ith_param = torch.mean(diff_D**2);
                            recon_V_loss_ith_param = torch.mean(diff_V**2);
                        elif(self.loss_types['recon'] == "MAE"):
                            recon_D_loss_ith_param = torch.mean(torch.abs(diff_D));
                            recon_V_loss_ith_param = torch.mean(torch.abs(diff_V));
                        else:
                            recon_D_loss_ith_param = torch.zeros(1, dtype = torch.float32, device = device);
                            recon_V_loss_ith_param = torch.zeros(1, dtype = torch.float32, device = device);
                        
                        loss_recon_D += recon_D_loss_ith_param;
                        loss_recon_V += recon_V_loss_ith_param;
                        
                        # Store per-parameter-combination loss
                        param_tuple = tuple(self.param_space.train_space[i, :]);
                        self._store_loss_by_param('recon_D', param_tuple, iter + 1, recon_D_loss_ith_param.item());
                        self._store_loss_by_param('recon_V', param_tuple, iter + 1, recon_V_loss_ith_param.item());

                        LOGGER.debug("Reconstruction Loss (Autoencoder_Pair) - complete for parameter combination %d" % i);
                        self.timer.end("Reconstruction Loss");


                    # --------------------------------------------------------------------------------
                    # Consistency losses

                    if(self.loss_weights['consistency'] > 0):
                        self.timer.start("Consistency Loss");
                        LOGGER.debug("Consistency Loss (Autoencoder_Pair) - start for parameter combination %d" % i);

                        # Make sure Z_V actually looks like the time derivative of Z_D. 
                        if(self.physics.Uniform_t_Grid == True):
                            h               : float             = t_Grid_i[1] - t_Grid_i[0];
                            dZ_Di_dt        : torch.Tensor      = Derivative1_Order4(U = Z_D_i, h = h);
                        else:
                            dZ_Di_dt        : torch.Tensor      = Derivative1_Order2_NonUniform(U = Z_D_i, t_Grid = t_Grid_i);
                        
                        # Compute differences once
                        diff_Z = dZ_Di_dt - Z_V_i;
                        
                        # Compute loss from difference
                        consistency_Z_loss_ith_param = torch.mean(diff_Z**2) if self.loss_types['consistency'] == "MSE" else torch.mean(torch.abs(diff_Z));
                        loss_consistency_Z          += consistency_Z_loss_ith_param;
                        
                        # Store per-parameter-combination loss
                        param_tuple = tuple(self.param_space.train_space[i, :]);
                        self._store_loss_by_param('consistency_Z', param_tuple, iter + 1, consistency_Z_loss_ith_param.item());

                        # Next, make sure that V_Pred actually looks like the derivative of D_Pred. 
                        if(self.physics.Uniform_t_Grid  == True):
                            h               : float             = t_Grid_i[1] - t_Grid_i[0];
                            dD_Pred_i_dt    : torch.Tensor      = Derivative1_Order4(U = D_Pred_i, h = h);
                        else:
                            dD_Pred_i_dt    : torch.Tensor      = Derivative1_Order2_NonUniform(U = D_Pred_i, t_Grid = t_Grid_i);

                        # Compute difference once
                        diff_U = dD_Pred_i_dt - V_Pred_i;
                        
                        # Compute loss from difference
                        consistency_U_loss_ith_param = torch.mean(diff_U**2) if self.loss_types['consistency'] == "MSE" else torch.mean(torch.abs(diff_U));
                        loss_consistency_U          += consistency_U_loss_ith_param;
                        
                        # Store per-parameter-combination loss
                        self._store_loss_by_param('consistency_U', param_tuple, iter + 1, consistency_U_loss_ith_param.item());

                        LOGGER.debug("Consistency Loss (Autoencoder_Pair) - complete for parameter combination %d" % i);
                        self.timer.end("Consistency Loss");


                    # ----------------------------------------------------------------------------
                    # Chain Rule Losses

                    if(self.loss_weights['chain_rule'] > 0):
                        self.timer.start("Chain Rule Loss");
                        LOGGER.debug("Chain Rule Loss (Autoencoder_Pair) - start for parameter combination %d" % i);

                        # First, we compute the U portion of the chain rule loss. This stems from the 
                        # fact that 
                        #       (d/dt)U(t) \approx (d/dt)\phi_D,D(Z_D(t)) 
                        #                   = (d/dz)\phi_D,D(Z_D(t)) Z_V(t)
                        # Here, \phi_D,D is the displacement portion of the decoder. We can use torch 
                        # to compute jacobian-vector products. Note that the jvp function expects a 
                        # function as its first arguments (to define the forward pass). It passes the 
                        # inputs through func, then computes the jacobian-vector product (using 
                        # reverse mode AD) of inputs with v. It returns the result of the forward pass 
                        # and the associated jacobian-vector product. We only keep the latter.
                        d_dz_D_Pred__Z_V_i  : torch.Tensor  = torch.autograd.functional.jvp(
                                                                    func    = lambda Z_D : encoder_decoder_device.Displacement_Autoencoder.Decode(Z_D)[0], 
                                                                    inputs  = Z_D_i, 
                                                                    v       = Z_V_i)[1];
                        
                        # Compute difference once
                        diff_U = V_i - d_dz_D_Pred__Z_V_i;
                        
                        # Compute loss from difference
                        chain_rule_U_loss_ith_param = torch.mean(diff_U**2) if self.loss_types['chain_rule'] == "MSE" else torch.mean(torch.abs(diff_U));
                        loss_chain_rule_U          += chain_rule_U_loss_ith_param;
                        
                        # Store per-parameter-combination loss
                        param_tuple = tuple(self.param_space.train_space[i, :]);
                        self._store_loss_by_param('chain_rule_U', param_tuple, iter + 1, chain_rule_U_loss_ith_param.item());

                        # Next, we compute the Z portion of the chain rule loss:
                        #       (d/dt)Z(t) \approx (d/dt)\phi_E,D(D(t))
                        #                   = (d/dX)\phi_E,D(D(t)) V(t)
                        # Here, \phi_E,D is the displacement portion of the encoder.
                        d_dx_Z_D__V         : torch.Tensor  = torch.autograd.functional.jvp(
                                                                    func    = lambda D : encoder_decoder_device.Displacement_Autoencoder.Encode(D)[0],
                                                                    inputs  = D_i, 
                                                                    v       = V_i)[1];
                        
                        # Compute difference once
                        diff_Z = Z_V_i - d_dx_Z_D__V;
                        
                        # Compute loss from difference
                        chain_rule_Z_loss_ith_param = torch.mean(diff_Z**2) if self.loss_types['chain_rule'] == "MSE" else torch.mean(torch.abs(diff_Z));
                        loss_chain_rule_Z          += chain_rule_Z_loss_ith_param;
                        
                        # Store per-parameter-combination loss
                        self._store_loss_by_param('chain_rule_Z', param_tuple, iter + 1, chain_rule_Z_loss_ith_param.item());

                        LOGGER.debug("Chain Rule Loss (Autoencoder_Pair) - complete for parameter combination %d" % i);
                        self.timer.end("Chain Rule Loss");


                    # ----------------------------------------------------------------------------
                    # Setup Rollout losses.

                    if(self.loss_weights['rollout'] > 0 and p_rollout > 0):
                        self.timer.start("Rollout Setup");
                        LOGGER.debug("Rollout Setup (Autoencoder_Pair) - start for parameter combination %d" % i);

                        # Select the latent states we want to use as initial conditions for the i'th 
                        # combination of parameter values. This should be the first 
                        # n_rollout_ICs[i] frames (n_rollout_ICs[i] is computed such that if we 
                        # simulate the first n_rollout_ICs[i] frames, the final times are less than 
                        # the final time for this combination of parameter values. Each element of 
                        # ROM_Rollout_IC is a 1 element list of torch.Tensor objects of shape 
                        # (n_rollout_ICs[i], n_z).
                        ROM_Rollout_ICs.append([Z_i[0][:n_rollout_ICs[i], :], Z_i[1][:n_rollout_ICs[i], :]]);

                        # Generate the indices of the frames we want to use as the targets.
                        Rollout_Indices_i   : numpy.ndarray = numpy.random.randint(0, t_Grid_rollout[i].shape[0], n_rollout_ICs[i]);
                        Rollout_Indices.append(Rollout_Indices_i);

                        # Fetch the corresponding targets.
                        FOM_Rollout_Targets_i : list[torch.Tensor] = [];
                        for j in range(self.n_IC):
                            FOM_Rollout_Targets_ij : numpy.ndarray = numpy.empty((n_rollout_ICs[i],) + tuple(self.physics.Frame_Shape), dtype = numpy.float32);
                            
                            # Fetch the target solution at the target time for each IC.
                            for k in range(n_rollout_ICs[i]):
                                FOM_Rollout_Targets_ij[k, ...] = U_Target_Rollout_Trajectory[i][j][k, Rollout_Indices_i[k], ...];
                            FOM_Rollout_Targets_i.append(torch.tensor(FOM_Rollout_Targets_ij, dtype = torch.float32, device = device));
                        FOM_Rollout_Targets.append(FOM_Rollout_Targets_i);

                        # Fetch the corresponding target by encoding the FOM targets using the 
                        # current encoder.
                        ROM_Rollout_Targets.append(list(encoder_decoder_device.Encode(*FOM_Rollout_Targets_i)));
                    
                        LOGGER.debug("Rollout Setup (Autoencoder_Pair) - complete for parameter combination %d" % i);
                        self.timer.end("Rollout Setup");

                # Store the total recon, consistency, and chain rule losses.
                self._store_total_loss('recon_D', iter + 1, loss_recon_D.item());
                self._store_total_loss('recon_V', iter + 1, loss_recon_V.item());
                self._store_total_loss('consistency_Z', iter + 1, loss_consistency_Z.item());
                self._store_total_loss('consistency_U', iter + 1, loss_consistency_U.item());
                self._store_total_loss('chain_rule_U', iter + 1, loss_chain_rule_U.item());
                self._store_total_loss('chain_rule_Z', iter + 1, loss_chain_rule_Z.item());


                # --------------------------------------------------------------------------------
                # Latent Dynamics, Stability losses

                self.timer.start("Calibration");
                LOGGER.debug("Calibration (Autoencoder_Pair) - start");

                # Compute the latent dynamics and stability losses. Also fetch the current latent
                # dynamics coefficients for each training point. The latter is stored in a 2d array 
                # called "train_coefs" of shape (n_train, n_coefs), where n_train = number of training 
                # parameter parameters and n_coefs denotes the number of coefficients in the latent
                # dynamics model. 
                train_coefs, loss_LD_list, loss_stab_list   = self.latent_dynamics.calibrate(   
                                                            Latent_States    = Latent_States, 
                                                            t_Grid           = t_Train_device,
                                                            input_coefs      = train_coefs_list,
                                                            loss_type        = self.loss_types['LD'],
                                                            params           = self.param_space.train_space);

                # Log coefficient statistics to diagnose constant dynamics issue
                if iter % 100 == 0 or iter == self.restart_iter:  # Log every 100 iters and first iter
                    LOGGER.info("Epoch %d: Coefs shape=%s, min=%.6e, max=%.6e, mean=%.6e, std=%.6e, abs_mean=%.6e" % (
                        iter + 1, str(train_coefs.shape),
                        float(train_coefs.min().item()), float(train_coefs.max().item()),
                        float(train_coefs.mean().item()), float(train_coefs.std().item()),
                        float(torch.abs(train_coefs).mean().item())));

                # Append the LD and stability losses to loss_by_param.
                for i in range(n_train):
                    param_tuple = tuple(self.param_space.train_space[i, :]);
                    self._store_loss_by_param('LD', param_tuple, iter + 1, loss_LD_list[i].item());
                    self._store_loss_by_param('stab', param_tuple, iter + 1, loss_stab_list[i].item());

                # Compute the total loss.
                loss_LD     = torch.sum(torch.stack(loss_LD_list));
                loss_stab   = torch.sum(torch.stack(loss_stab_list));

                # Append the total loss to loss_by_param.
                self._store_total_loss('LD', iter + 1, loss_LD.item());
                self._store_total_loss('stab', iter + 1, loss_stab.item());

                LOGGER.debug("Calibration (Autoencoder_Pair) - complete");
                self.timer.end("Calibration");


                # ---------------------------------------------------------------------------------
                # Rollout loss. Note that we need the coefficients before we can compute this.

                if(self.loss_weights['rollout'] > 0 and p_rollout > 0):
                    self.timer.start("Rollout Loss");
                    LOGGER.debug("Rollout Loss (Autoencoder_Pair) - start");

                    # Simulate the frames forward in time. This should return an n_param element list
                    # whose i'th element is a 2 element list whose j'th element has shape (n_t_i, 
                    # n_rollout_ICs[i], n_z) whose p, q, r element should hold the r'th component 
                    # of the p'th frame of the j'th time derivative of the solution when we use the 
                    # p'th initial condition for the i'th combination of parameter values.
                    #
                    # Note that the latent dynamics are autonomous. Further, because we are simulating 
                    # each IC for the same amount of time, the specific values of the time are
                    # irreverent. The simulate function exploits this by solving one big IVP for each 
                    # combination of parameter values, rather than n(i) smaller ones. 
                    ROM_Predict_Rollout_Trajectory  : list[list[torch.Tensor]]  = self.latent_dynamics.simulate(
                                                                                                coefs   = train_coefs, 
                                                                                                IC      = ROM_Rollout_ICs, 
                                                                                                t_Grid  = t_Grid_rollout,
                                                                                                params  = self.param_space.train_space);            

                    # Now cycle through the training examples.
                    for i in range(n_train):
                        # Fetch the latent displacement/velocity for the i'th combination of parameter
                        # values. 
                        ROM_Predict_Rollout_Trajectory_i    : list[torch.Tensor]  = ROM_Predict_Rollout_Trajectory[i];
                        ROM_D_Predict_Rollout_Trajectory_i  : torch.Tensor        = ROM_Predict_Rollout_Trajectory_i[0];               # shape = (len(t_Grid_rollout[i]), n_rollout_ICs[i], n_z)
                        ROM_V_Predict_Rollout_Trajectory_i  : torch.Tensor        = ROM_Predict_Rollout_Trajectory_i[1];               # shape = (len(t_Grid_rollout[i]), n_rollout_ICs[i], n_z)

                        # Fetch Rollout_Indices[i][j]'th frame from the rollout trajectory for the 
                        # j'th IC, which represents an approximation of the solution at 
                        # self.t_Train[i][j] + t_Grid_rollout[i][Rollout_Indices[i][j]]. We will 
                        # compare this to the interpolated FOM solution at the same time, which 
                        # should be stored in FOM_Rollout_Targets[i][0][j, ...]

                        # First, fetch the predicted solution at the target times.
                        ROM_D_Rollout_Predict_i : torch.Tensor = torch.empty((n_rollout_ICs[i], encoder_decoder_device.n_z), dtype = torch.float32, device = device);
                        ROM_V_Rollout_Predict_i : torch.Tensor = torch.empty((n_rollout_ICs[i], encoder_decoder_device.n_z), dtype = torch.float32, device = device);
                        
                        for j in range(n_rollout_ICs[i]):
                            ROM_D_Rollout_Predict_i[j, :] = ROM_D_Predict_Rollout_Trajectory_i[Rollout_Indices[i][j], j, :];
                            ROM_V_Rollout_Predict_i[j, :] = ROM_V_Predict_Rollout_Trajectory_i[Rollout_Indices[i][j], j, :];

                        # Now fetch the corresponding FOM targets.
                        FOM_Rollout_Targets_i   : list[torch.Tensor]    = FOM_Rollout_Targets[i];
                        FOM_D_Rollout_Target_i  : torch.Tensor          = FOM_Rollout_Targets_i[0];     # shape = (n_rollout_ICs[i], physics.Frame_Shape)
                        FOM_V_Rollout_Target_i  : torch.Tensor          = FOM_Rollout_Targets_i[1];     # shape = (n_rollout_ICs[i], physics.Frame_Shape)

                        # And the corresponding ROM targets.
                        ROM_Rollout_Targets_i   : list[torch.Tensor]    = ROM_Rollout_Targets[i];
                        ROM_D_Rollout_Target_i  : torch.Tensor          = ROM_Rollout_Targets_i[0];     # shape = (n_rollout_ICs[i], physics.Frame_Shape)
                        ROM_V_Rollout_Target_i  : torch.Tensor          = ROM_Rollout_Targets_i[1];     # shape = (n_rollout_ICs[i], physics.Frame_Shape)

                        # Decode the latent predictions to get FOM predictions.
                        FOM_D_Rollout_Predict_i, FOM_V_Rollout_Predict_i = encoder_decoder_device.Decode(ROM_D_Rollout_Predict_i, ROM_V_Rollout_Predict_i);
                    
                        # Get the corresponding FOM targets.
                        FOM_Rollout_Targets_i   : list[torch.Tensor]    = FOM_Rollout_Targets[i];         # shape = (n_rollout_ICs[i], physics.Frame_Shape)
                        FOM_D_Rollout_Target_i  : torch.Tensor          = FOM_Rollout_Targets_i[0];       # shape = (n_rollout_ICs[i], physics.Frame_Shape)
                        FOM_V_Rollout_Target_i  : torch.Tensor          = FOM_Rollout_Targets_i[1];       # shape = (n_rollout_ICs[i], physics.Frame_Shape)
                    
                        # Compute differences once
                        diff_ROM_D = ROM_D_Rollout_Target_i - ROM_D_Rollout_Predict_i;
                        diff_ROM_V = ROM_V_Rollout_Target_i - ROM_V_Rollout_Predict_i;
                        diff_FOM_D = (FOM_D_Rollout_Target_i - FOM_D_Rollout_Predict_i);
                        diff_FOM_V = (FOM_V_Rollout_Target_i - FOM_V_Rollout_Predict_i);
                        
                        # Compute losses from normalized differences
                        if(self.loss_types['rollout'] == "MSE"):
                            rollout_ROM_D_loss_ith_param = torch.mean(diff_ROM_D**2);
                            rollout_ROM_V_loss_ith_param = torch.mean(diff_ROM_V**2);
                            rollout_FOM_D_loss_ith_param = torch.mean(diff_FOM_D**2);
                            rollout_FOM_V_loss_ith_param = torch.mean(diff_FOM_V**2);
                        elif(self.loss_types['rollout'] == "MAE"):
                            rollout_ROM_D_loss_ith_param = torch.mean(torch.abs(diff_ROM_D));
                            rollout_ROM_V_loss_ith_param = torch.mean(torch.abs(diff_ROM_V));
                            rollout_FOM_D_loss_ith_param = torch.mean(torch.abs(diff_FOM_D));
                            rollout_FOM_V_loss_ith_param = torch.mean(torch.abs(diff_FOM_V));
                        else:
                            rollout_ROM_D_loss_ith_param = torch.zeros(1, dtype = torch.float32, device = device);
                            rollout_ROM_V_loss_ith_param = torch.zeros(1, dtype = torch.float32, device = device);
                            rollout_FOM_D_loss_ith_param = torch.zeros(1, dtype = torch.float32, device = device);
                            rollout_FOM_V_loss_ith_param = torch.zeros(1, dtype = torch.float32, device = device);
                        
                        loss_rollout_ROM_D  += rollout_ROM_D_loss_ith_param;
                        loss_rollout_ROM_V  += rollout_ROM_V_loss_ith_param;
                        loss_rollout_FOM_D  += rollout_FOM_D_loss_ith_param;
                        loss_rollout_FOM_V  += rollout_FOM_V_loss_ith_param;
                        
                        # Store per-parameter-combination loss
                        param_tuple = tuple(self.param_space.train_space[i, :]);
                        self._store_loss_by_param('rollout_ROM_D', param_tuple, iter + 1, rollout_ROM_D_loss_ith_param.item());
                        self._store_loss_by_param('rollout_ROM_V', param_tuple, iter + 1, rollout_ROM_V_loss_ith_param.item());
                        self._store_loss_by_param('rollout_FOM_D', param_tuple, iter + 1, rollout_FOM_D_loss_ith_param.item());
                        self._store_loss_by_param('rollout_FOM_V', param_tuple, iter + 1, rollout_FOM_V_loss_ith_param.item());

                    # Store total rollout loss.
                    self._store_total_loss('rollout_ROM_D', iter + 1, loss_rollout_ROM_D.item());
                    self._store_total_loss('rollout_ROM_V', iter + 1, loss_rollout_ROM_V.item());
                    self._store_total_loss('rollout_FOM_D', iter + 1, loss_rollout_FOM_D.item());
                    self._store_total_loss('rollout_FOM_V', iter + 1, loss_rollout_FOM_V.item());

                    LOGGER.debug("Rollout Loss (Autoencoder_Pair) - complete");
                    self.timer.end("Rollout Loss");


                # ---------------------------------------------------------------------------------
                # IC Rollout loss. This simulates forward from the FOM initial conditions.

                if(self.loss_weights['IC_rollout'] > 0 and p_IC_rollout > 0):
                    self.timer.start("IC Rollout Loss");
                    LOGGER.debug("IC Rollout Loss (Autoencoder_Pair) - start");

                    # Cycle through the training examples for IC rollout
                    for i in range(n_train):
                        # Fetch the FOM initial conditions for this combination of parameters
                        param_i           : numpy.ndarray             = self.param_space.train_space[i, :];
                        FOM_IC_i          : list[numpy.ndarray]       = self.physics.initial_condition(param_i);
                        
                        # Convert to tensors and reshape for encoding
                        D_IC_i            : torch.Tensor              = torch.tensor(FOM_IC_i[0], dtype = torch.float32, device = device).reshape((1,) + FOM_IC_i[0].shape);
                        V_IC_i            : torch.Tensor              = torch.tensor(FOM_IC_i[1], dtype = torch.float32, device = device).reshape((1,) + FOM_IC_i[1].shape);
                        if self.has_normalization():
                            D_IC_i = self.normalize_tensor(D_IC_i, 0);
                            V_IC_i = self.normalize_tensor(V_IC_i, 1);
                        
                        # Encode the FOM initial conditions
                        Z_D_IC_i, Z_V_IC_i = encoder_decoder_device.Encode(D_IC_i, V_IC_i);
                        
                        # Get the coefficients for this combination of parameters
                        train_coef_i            : torch.Tensor              = train_coefs[i, :].reshape(1, -1);
                        
                        # Simulate the latent dynamics forward in time
                        Z_IC_Rollout_i    : list[list[torch.Tensor]]  = self.latent_dynamics.simulate(  coefs   = train_coef_i, 
                                                                                                        IC      = [[Z_D_IC_i, Z_V_IC_i]], 
                                                                                                        t_Grid  = [t_Grid_IC_rollout[i]], 
                                                                                                        params  = param_i.reshape(1, -1));
                        
                        # Extract the predicted trajectory
                        Z_D_IC_Predict_i  : torch.Tensor              = Z_IC_Rollout_i[0][0];  # shape = (n_t_IC_rollout[i], 1, n_z)
                        Z_V_IC_Predict_i  : torch.Tensor              = Z_IC_Rollout_i[0][1];  # shape = (n_t_IC_rollout[i], 1, n_z)
                        
                        # Remove the singleton dimension
                        Z_D_IC_Predict_i  : torch.Tensor              = Z_D_IC_Predict_i.squeeze(1);  # shape = (n_t_IC_rollout[i], n_z)
                        Z_V_IC_Predict_i  : torch.Tensor              = Z_V_IC_Predict_i.squeeze(1);  # shape = (n_t_IC_rollout[i], n_z)

                        # Decode the predicted trajectory to get FOM predictions
                        D_IC_Predict_i, V_IC_Predict_i = encoder_decoder_device.Decode(Z_D_IC_Predict_i, Z_V_IC_Predict_i);
                        
                        # Get the corresponding FOM targets
                        U_IC_Target_i     : list[torch.Tensor]        = U_IC_Rollout_Targets[i];
                        D_IC_Target_i     : torch.Tensor              = U_IC_Target_i[0];  # shape = (n_t_IC_rollout[i], physics.Frame_Shape)
                        V_IC_Target_i     : torch.Tensor              = U_IC_Target_i[1];  # shape = (n_t_IC_rollout[i], physics.Frame_Shape)

                        # Encode the FOM targets for latent space comparison
                        Z_D_IC_Target_i, Z_V_IC_Target_i = encoder_decoder_device.Encode(D_IC_Target_i, V_IC_Target_i);

                        # Compute differences once
                        diff_Z_D = Z_D_IC_Target_i - Z_D_IC_Predict_i;
                        diff_Z_V = Z_V_IC_Target_i - Z_V_IC_Predict_i;
                        diff_D = (D_IC_Target_i - D_IC_Predict_i);
                        diff_V = (V_IC_Target_i - V_IC_Predict_i);
                        
                        # Compute losses from normalized differences
                        if(self.loss_types['IC_rollout'] == "MSE"):
                            IC_rollout_Z_D_loss_ith_param = torch.mean(diff_Z_D**2);
                            IC_rollout_Z_V_loss_ith_param = torch.mean(diff_Z_V**2);
                            IC_rollout_D_loss_ith_param = torch.mean(diff_D**2);
                            IC_rollout_V_loss_ith_param = torch.mean(diff_V**2);
                        elif(self.loss_types['IC_rollout'] == "MAE"):
                            IC_rollout_Z_D_loss_ith_param = torch.mean(torch.abs(diff_Z_D));
                            IC_rollout_Z_V_loss_ith_param = torch.mean(torch.abs(diff_Z_V));
                            IC_rollout_D_loss_ith_param = torch.mean(torch.abs(diff_D));
                            IC_rollout_V_loss_ith_param = torch.mean(torch.abs(diff_V));
                        else:
                            IC_rollout_Z_D_loss_ith_param = torch.zeros(1, dtype = torch.float32, device = device);
                            IC_rollout_Z_V_loss_ith_param = torch.zeros(1, dtype = torch.float32, device = device);
                            IC_rollout_D_loss_ith_param = torch.zeros(1, dtype = torch.float32, device = device);
                            IC_rollout_V_loss_ith_param = torch.zeros(1, dtype = torch.float32, device = device);
                        
                        loss_IC_rollout_Z_D  += IC_rollout_Z_D_loss_ith_param;
                        loss_IC_rollout_Z_V  += IC_rollout_Z_V_loss_ith_param;
                        loss_IC_rollout_D    += IC_rollout_D_loss_ith_param;
                        loss_IC_rollout_V    += IC_rollout_V_loss_ith_param;
                        
                        # Store per-parameter-combination loss
                        param_tuple = tuple(self.param_space.train_space[i, :]);
                        self._store_loss_by_param('IC_rollout_Z_D', param_tuple, iter + 1, IC_rollout_Z_D_loss_ith_param.item());
                        self._store_loss_by_param('IC_rollout_Z_V', param_tuple, iter + 1, IC_rollout_Z_V_loss_ith_param.item());
                        self._store_loss_by_param('IC_rollout_D', param_tuple, iter + 1, IC_rollout_D_loss_ith_param.item());
                        self._store_loss_by_param('IC_rollout_V', param_tuple, iter + 1, IC_rollout_V_loss_ith_param.item());

                    # Store total IC rollout loss.
                    self._store_total_loss('IC_rollout_Z_D', iter + 1, loss_IC_rollout_Z_D.item());
                    self._store_total_loss('IC_rollout_Z_V', iter + 1, loss_IC_rollout_Z_V.item());
                    self._store_total_loss('IC_rollout_D', iter + 1, loss_IC_rollout_D.item());
                    self._store_total_loss('IC_rollout_V', iter + 1, loss_IC_rollout_V.item());

                    LOGGER.debug("IC Rollout Loss (Autoencoder_Pair) - complete");
                    self.timer.end("IC Rollout Loss");


                # --------------------------------------------------------------------------------
                # Total loss

                loss_recon          : torch.Tensor  = loss_recon_D          + loss_recon_V;
                loss_consistency    : torch.Tensor  = loss_consistency_Z    + loss_consistency_U;
                loss_chain_rule     : torch.Tensor  = loss_chain_rule_U     + loss_chain_rule_Z;
                loss_rollout        : torch.Tensor  = loss_rollout_FOM_D    + loss_rollout_FOM_V + loss_rollout_ROM_D + loss_rollout_ROM_V;
                loss_IC_rollout     : torch.Tensor  = loss_IC_rollout_D     + loss_IC_rollout_V + loss_IC_rollout_Z_D + loss_IC_rollout_Z_V;

                # Compute the final loss.
                LOGGER.debug("Computing total loss (Autoencoder_Pair)");
                loss = (self.loss_weights['recon']          * loss_recon + 
                        self.loss_weights['consistency']    * loss_consistency +
                        self.loss_weights['chain_rule']     * loss_chain_rule + 
                        self.loss_weights['rollout']        * loss_rollout +
                        self.loss_weights['IC_rollout']     * loss_IC_rollout +
                        self.loss_weights['LD']             * loss_LD + 
                        self.loss_weights['stab']           * loss_stab);
                self._store_total_loss('total', iter + 1, loss.item());
                LOGGER.debug("Total loss (Autoencoder_Pair) computed: %f" % loss.item());

            # Convert coefs to numpy and find the maximum element.
            # Store a detached copy for reporting (needed after backprop), but keep original for gradient flow
            with torch.no_grad():
                train_coefs_detached    : numpy.ndarray = train_coefs.detach().cpu().numpy();                # Shape = (n_train, n_coefs).
                max_train_coef          : numpy.float32 = numpy.abs(train_coefs_detached).max();




            # -------------------------------------------------------------------------------------
            # Backward Pass

            self.timer.start("Backwards Pass");
            LOGGER.debug("Backward Pass - start (iteration %d)" % (iter + 1));

            #  Run back propagation and update the encoder_decoder parameters. 
            # Note: optimizer.zero_grad() is already called at the start of the iteration (line 373)
            loss.backward();
            
            # Clip gradients to prevent explosion during latent dynamics rollout.
            grad_norm = torch.nn.utils.clip_grad_norm_(self.encoder_decoder.parameters(), max_norm = self.gradient_clip);
            
            # Log if gradient clipping activates (indicates potential instability)
            if grad_norm > self.gradient_clip:
                LOGGER.warning("Gradient norm %.2f exceeded threshold, clipped to %f (iter %d)" % (grad_norm, self.gradient_clip, iter + 1));
            
            LOGGER.debug("Backward Pass - backward() complete, calling optimizer.step()");
            self.optimizer.step();
            LOGGER.debug("Backward Pass - complete (iteration %d)" % (iter + 1));

            # Check if we hit a new minimum loss. If so, make a checkpoint, record the loss and 
            # the iteration number. 
            # NOTE: Skip checkpointing during warmup period to avoid saving "lucky" early epochs
            # that benefit from distribution shift before encoder_decoder has adapted.
            
            if loss.item() < best_loss:
                if epochs_in_round >= self.warmup_epochs:
                    LOGGER.info("Got a new lowest loss (%f) on epoch %d" % (loss.item(), iter + 1));
                    torch.save(encoder_decoder_device.cpu().state_dict(), self.path_checkpoint + '/' + 'checkpoint.pt');
                    encoder_decoder_device.to(device);  # Move encoder_decoder back to original device after saving
                    
                    # Update the best set of parameters. 
                    self.best_train_coefs   = train_coefs_detached.copy();     # Shape = (n_train, n_coefs).
                    self.best_epoch         = iter;
                    best_loss               = loss.item();
                else:
                    LOGGER.debug("Skipping checkpoint during warmup period (epoch %d/%d in round, warmup ends at %d)" % 
                               (epochs_in_round, next_iter - self.restart_iter, self.warmup_epochs));

            self.timer.end("Backwards Pass");
            


            # -------------------------------------------------------------------------------------
            # Report Results from this iteration 

            self.timer.start("Report");

            # Report the current iteration number and losses
            if not is_autoencoder_pair:
                info_str : str = "Iter: %05d/%d, Total: %3.10f" % (iter + 1, self.max_iter, loss.item());
                if(self.loss_weights['recon'] > 0):         info_str += ", Recon: %3.6f"                            % loss_recon.item();
                if(self.loss_weights['rollout'] > 0):       info_str += ", Roll FOM: %3.6f, Roll ROM: %3.6f"        % (loss_rollout_FOM.item(),    loss_rollout_ROM.item());
                if(self.loss_weights['IC_rollout'] > 0):    info_str += ", IC Roll FOM: %3.6f, IC Roll ROM: %3.6f"  % (loss_IC_rollout_FOM.item(), loss_IC_rollout_ROM.item());
                if(self.loss_weights['LD'] > 0):            info_str += ", LD: %3.6f"                               % loss_LD.item();
                if(self.loss_weights['stab'] > 0):          info_str += ", Stab: %3.6f"                             % loss_stab.item();
                info_str += ", max|c|: %.3f" % max_train_coef;
                LOGGER.info(info_str);
            
            else:  # is_autoencoder_pair
                info_str : str = "Iter: %05d/%d, Total: %3.6f" % (iter + 1, self.max_iter, loss.item());
                if(self.loss_weights['recon'] > 0):         info_str += ", Recon D: %3.6f, Recon V: %3.6f"                                              % (loss_recon_D.item(),       loss_recon_V.item());
                if(self.loss_weights['consistency'] > 0):   info_str += ", Consistency Z: %3.6f, Consistency U: %3.6f"                                  % (loss_consistency_Z.item(), loss_consistency_U.item());
                if(self.loss_weights['chain_rule'] > 0):    info_str += ", CR U: %3.6f, CR Z: %3.6f"                                                    % (loss_chain_rule_U.item(),  loss_chain_rule_Z.item());
                if(self.loss_weights['rollout'] > 0):       info_str += ", Roll FOM D: %3.6f, Roll FOM V: %3.6f, Roll ROM D: %3.6f, Roll ROM V: %3.6f"  % (loss_rollout_FOM_D.item(), loss_rollout_FOM_V.item(),  loss_rollout_ROM_D.item(),  loss_rollout_ROM_V.item());
                if(self.loss_weights['IC_rollout'] > 0):    info_str += ", IC Roll D: %3.6f, IC Roll V: %3.6f, IC Roll ZD: %3.6f, IC Roll ZV: %3.6f"    % (loss_IC_rollout_D.item(),  loss_IC_rollout_V.item(),   loss_IC_rollout_Z_D.item(), loss_IC_rollout_Z_V.item());
                if(self.loss_weights['LD'] > 0):            info_str += ", LD: %3.6f"                                                                   % loss_LD.item();
                if(self.loss_weights['stab'] > 0):          info_str += ", Stab: %3.6f"                                                                 % loss_stab.item();
                info_str += ", max|c|: %.3f" % max_train_coef;
                LOGGER.info(info_str);

            # Report the set of parameter combinations using scientific notation.
            def format_param(p: numpy.ndarray) -> str:
                # Format parameter array in scientific notation.
                return '[' + ', '.join(['%.2e' % val for val in p]) + ']';
            
            param_string : str  = 'Param: ' + format_param(self.param_space.train_space[0, :]);
            for i in range(1, n_train):
                param_string    = param_string + ', ' + format_param(self.param_space.train_space[i, :]);

            LOGGER.debug(param_string);

            self.timer.end("Report");
            
            LOGGER.debug("Completed training iteration %d/%d" % (iter + 1, next_iter));
            self.timer.end("train_step");
        
        # We are ready to wrap up the training procedure.
        self.timer.start("finalize");
        

        # -------------------------------------------------------------------------------------
        # Save loss history to file after training is complete
        # -------------------------------------------------------------------------------------
        
        # Component losses are stored in loss_by_param (saved separately as pickle)

        # Keep filename consistent with the one used for restart/load above.
        base_filename       : str   = self.config['physics']['type'];
        loss_by_param_path  : str   = os.path.join(self.path_results, base_filename + '_loss_by_param.pkl');

        # Save self.loss_by_param to file.     
        with open(loss_by_param_path, 'wb') as f:
            pickle.dump(self.loss_by_param, f);
        
        LOGGER.info("Saved per-parameter loss tracking to %s" % loss_by_param_path);

        # Now that we have completed another round of training, update the restart iteration.
        self.restart_iter += self.n_iter;

        # Recover the encoder_decoder + coefficients which attained the lowest loss from this round of 
        # training.
        assert(self.best_train_coefs is not None);
        LOGGER.info("encoder_decoder attained it's best performance on epoch %d. Replacing encoder_decoder with the checkpoint from that epoch" % self.best_epoch);
        state_dict  = torch.load(self.path_checkpoint + '/' + 'checkpoint.pt', map_location = 'cpu');
        self.encoder_decoder.load_state_dict(state_dict);

        # Report timing information.
        self.timer.end("finalize");
        self.timer.print();

        # All done!
        return;


    def _reset_optimizer(self) -> None:
        """
        Set the optimizer's m_t and v_t attributes (first and second moments) to zero. After each 
        training round, the momentum from the previous epoch may point us in the wrong direction. 
        Resetting the momentum eliminates this problem.
        """

        # Cycle through the optimizer's parameter groups.
        for group in self.optimizer.param_groups:

            # Cycle through the parameters in the group.
            for p in group['params']:
                state : dict = self.optimizer.state[p];

                # If the state is empty, skip this parameter.
                if not state:
                    continue;
                
                # zero the biased first moment estimate
                state['exp_avg'].zero_();

                # zero the biased second moment estimate
                state['exp_avg_sq'].zero_();
                
                # if you're using amsgrad:
                if 'max_exp_avg_sq' in state:
                    state['max_exp_avg_sq'].zero_();


    
    def _rollout_setup( self, 
                        t           : list[torch.Tensor], 
                        U           : list[list[torch.Tensor]], 
                        p_rollout   : float) -> tuple[list[torch.Tensor], list[int], list[list[torch.Tensor]]]:
        """
        An internal function that sets up the rollout loss. Specifically, it finds the t_grid for 
        simulating each latent frame we plan to rollout, the number of frames we can rollout for 
        each parameter value, the final time of each frame we rollout, and a target FOM frame for 
        each frame we rollout. The user should not call this function directly; only the train 
        method should call this.

        
        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        t: : list[torch.Tensor], len = n_param
            i'th element is a 1d torch.Tensor of shape (n_t_i) whose j'th element specifies the 
            time of the j'th frame in the FOM solution for the i'th combination of parameter 
            values. We assume the values in the j'th element are in increasing order and unique.

        U : list[list[torch.Tensor]], len = n_param
            i'th element is a n_IC element list whose j'th element is a torch.Tensor of shape 
            (n_t_i, ...) whose k'th element specifies the value of the j'th time derivative of the 
            FOM frame when using the i'th combination of parameter values.

        p_rollout : float
            A number between 0 and 1 specifying the ratio of the rollout time for a particular 
            combination of parameter values to the length of the time interval for that combination 
            of parameter values.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------
        
        t_Grid_rollout, n_rollout_ICs, U_Target_Rollout_Trajectory

        t_Grid_rollout : list[torch.Tensor], len = n_param
            i'th element is a 1d array whose j'th entry holds the j'th time at which we want to 
            rollout the first frame for the i'th combination of parameter values. Why do we do 
            this? When we rollout the latent states, we take advantage of the fact that the 
            governing dynamics is autonomous. Specifically, the actual at which times we solve the 
            ODE do not matter. All that matters is amount of time we solve for. This allows us to 
            use the same time grid for all latent states that we rollout, which dramatically 
            improves runtime. 

        n_rollout_ICs : list[int], len = n_param
            i'th element specifies how many frames we can rollout from the FOM solution for the 
            i'th combination of parameter values. Specifically, the first n_rollout_ICs[i] 
            frames from the i'th FOM solution are such that the time for each frame after rollout 
            will be less than the final time for that FOM solution.

        U_Target_Rollout_Trajectory : list[list[torch.Tensor]], len = n_param
            i'th element is an n_IC element list whose j'th element is a torch.Tensor of shape 
            (n_rollout_ICs[i], n_rollout_steps[i], physics.Frame_Shape), where n_rollout_steps[i] is 
            len(t_Grid_rollout[i]). The U_Target_Rollout_Trajectory[i][j][k, l] holds the 
            interpolated j'th derivative of the FOM solution at the l'th time step of the rollout 
            for the k'th IC for the i'th combination of parameter values. 
        """

        # Checks
        assert isinstance(p_rollout, float),    "type(p_rollout) = %s" % str(type(p_rollout));
        assert isinstance(U, list),             "type(U) = %s" % str(type(U));
        assert isinstance(t, list),             "type(t) = %s" % str(type(t));
        assert len(t) == len(U),                "len(t) = %d, len(U) = %d" % (len(t), len(U));

        assert isinstance(U[0], list),          "type(U[0]) = %s" % str(type(U[0]));
        n_param     : int   = len(U);

        for i in range(n_param):
            assert isinstance(U[i], list),          "type(U[%d]) = %s" % (i, str(type(U[i])));
            assert isinstance(t[i], torch.Tensor),  "type(t[%d]) = %s" % (i, str(type(t[i])));
            assert len(U[i])        == self.n_IC,   "len(U[%d]) = %d, self.n_IC = %d" % (i, len(U[i]), self.n_IC);
            assert len(t[i].shape)  == 1,           "len(t[%d].shape) = %d" % (i, len(t[i].shape));

            n_t_i : int = t[i].shape[0];
            for j in range(self.n_IC):
                assert isinstance(U[i][j], torch.Tensor), "type(U[%d][%d]) = %s" % (i, j, str(type(U[i][j])));
                assert U[i][j].shape[0]     == n_t_i,     "U[%d][%d].shape[0] = %d, n_t_i = %d" % (i, j, U[i][j].shape[0], n_t_i);


        # Other setup.        
        t_Grid_rollout                  : list[torch.Tensor]        = [];   # n_train element list whose i'th element is 1d array of times for rollout solve.
        n_rollout_ICs                   : list[int]                 = [];   # n_train element list whose i'th element specifies how many frames we should simulate forward.
        U_Target_Rollout_Trajectory     : list[list[torch.Tensor]]  = [];   # n_train element list whose i'th element is n_IC element list whose j'th element is a torch.Tensor of shape (n_rollout_ICs[i], n_rollout_steps[i], physics.Frame_Shape) holding target trajectories when we rollout the IC's for the j'th time derivative/i'th combination of parameters. 


        # -----------------------------------------------------------------------------------------
        # Find t_Grid_rollout, and n_rollout_ICs.

        for i in range(n_param):
            # Determine the amount of time that passes in the FOM simulation corresponding to the 
            # i'th combination of parameter values. 
            t_i                 : torch.Tensor  = t[i];
            n_t_i               : int           = t_i.shape[0];
            t_0_i               : float         = t_i[0].item();
            t_final_i           : float         = t_i[-1].item();

            # The final rollout time for this combination of parameter values. Remember that 
            # t_rollout is the proportion of t_final_i - t_0_i over which we simulate.
            t_rollout_i         : float         = p_rollout*(t_final_i - t_0_i);
            t_rollout_final_i   : float         = t_rollout_i + t_0_i;
            LOGGER.info("We will rollout the first frame for parameter combination #%d to t = %f" % (i, t_rollout_final_i));

            # Now figure out how many time steps occur before t_rollout_final_i.
            num_before_rollout_final_i  : int           = 0;
            for j in range(n_t_i):
                if(t_i[j] > t_rollout_final_i):
                    break; 
                
                num_before_rollout_final_i += 1;
            LOGGER.info("We will rollout each frame for parameter combination #%d over %d time steps" % (i, num_before_rollout_final_i));


            # Now define the rollout time grid for the i'th combination of parameter values.
            t_Grid_rollout.append(torch.linspace(start = t_0_i, end = t_rollout_final_i, steps = num_before_rollout_final_i));

            # Now figure out how many times occur less than t_rollout_i from t_final_i. If 
            # a time value satisifes this, then the corresponding frame is rollable.
            n_rollout_ICs_i : int = 0;
            for j in range(n_t_i):
                if(t_i[j] + t_rollout_i > t_final_i):
                    break;

                n_rollout_ICs_i += 1;
            n_rollout_ICs.append(n_rollout_ICs_i);
            LOGGER.info("We will rollout %d FOM frames for parameter combination #%d." % (n_rollout_ICs_i, i));


        # -----------------------------------------------------------------------------------------
        # Find U_Target_Rollout_Trajectory.

        for i in range(n_param):
            LOGGER.debug("Making interpolators for parameter combination #%d" % i);

            # Interpolate each U_Train[i][j], then evaluate it at the target times.
            U_Train_i               : list[torch.Tensor]            = U[i];                         # len = n_IC, i'th element is a torch.Tensor of shape (n_t(i), ...)
            # NOTE: SciPy interpolation expects NumPy arrays on CPU.
            t_Train_i               : numpy.ndarray                 = t[i].detach().cpu().numpy();  # shape = (n_t(i))
            t_Rollout_i             : numpy.ndarray                 = t_Grid_rollout[i].detach().cpu().numpy();  # shape = (n_rollout_steps_i,)

            # Fetch the number of frames we will rollout and the number of time 
            # steps we will rollout.
            n_rollout_ICs_i         : int = n_rollout_ICs[i];
            n_rollout_steps_i       : int = len(t_Grid_rollout[i]);

            # Fetch the targets for the i'th combination of parameter values.
            U_Target_Rollout_Trajectory_i       : list[torch.Tensor]            = [];
            for j in range(self.n_IC):
                # Interpolate the time series for the j'th derivative of the FOM solution when we 
                # use the i'th combination of parameter values.
                U_Train_ij  : numpy.ndarray = U_Train_i[j].detach().cpu().numpy();  # shape = (n_t(i), Physics.Frame_Shape)

                # Interpolate the time series for the j'th derivative of the FOM solution when we 
                # use the i'th combination of parameter values.
                U_Train_ij          : numpy.ndarray = U_Train_i[j].detach().numpy();        # shape = (n_t(i), Physics.Frame_Shape)
                U_Train_ij_interp                   = interpolate.make_interp_spline(x = t_Train_i, y = U_Train_ij, k = self.rollout_spline_order);

                U_Target_Rollout_Trajectory_ij : numpy.ndarray = numpy.empty((n_rollout_ICs_i, n_rollout_steps_i) + tuple(self.physics.Frame_Shape), dtype = numpy.float32);
                for k in range(n_rollout_ICs_i):
                    # Evaluate the i,j interpolator at the target times for the k'th rollout IC.
                    # The target times for the k'th IC rollout are the rollout timnes for the 1st 
                    # frame (t_Grid_rollout) plus the time of the k'th IC (t_Train_i[k]).
                    target_times = t_Rollout_i + t_Train_i[k];
                    U_Target_Rollout_Trajectory_ij[k, ...] = U_Train_ij_interp(target_times).astype(numpy.float32, copy = False);

                U_Target_Rollout_Trajectory_i.append(torch.tensor(U_Target_Rollout_Trajectory_ij, dtype = torch.float32, device = self.device));
            U_Target_Rollout_Trajectory.append(U_Target_Rollout_Trajectory_i);
    

        # All done!
        return t_Grid_rollout, n_rollout_ICs, U_Target_Rollout_Trajectory;



    def _IC_rollout_setup( self, 
                           t            : list[torch.Tensor], 
                           p_IC_rollout : float) -> tuple[list[torch.Tensor], list[int], list[list[torch.Tensor]]]:
        """
        An internal function that sets up the IC rollout loss. This is similar to _rollout_setup but
        for simulating forward from the FOM initial conditions. The user should not call this 
        function directly; only the train method should call this.

        
        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        t : list[torch.Tensor], len = n_param
            i'th element is a 1d torch.Tensor of shape (n_t_i) whose j'th element specifies the 
            time of the j'th frame in the FOM solution for the i'th combination of parameter 
            values. We assume the values in the j'th element are in increasing order and unique.

        p_IC_rollout : float
            A number between 0 and 1 specifying the ratio of the IC rollout time for a particular 
            combination of parameter values to the length of the time interval for that combination 
            of parameter values.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------
        
        t_Grid_IC_rollout, n_IC_rollout_frames, U_IC_Rollout_Targets

        t_Grid_IC_rollout : list[torch.Tensor], len = n_param
            i'th element is a 1d array whose j'th entry holds the j'th time at which we want to 
            rollout the initial condition for the i'th combination of parameter values.

        n_IC_rollout_frames : list[int], len = n_param
            i'th element specifies how many time steps we simulate forward from the initial condition
            for the i'th combination of parameter values.

        U_IC_Rollout_Targets : list[list[torch.Tensor]], len = n_param
            i'th element is an n_IC element list whose j'th element is a torch.Tensor of shape 
            (n_IC_rollout_frames[i], physics.Frame_Shape) consisting of the first 
            n_IC_rollout_frames[i] frames of the j'th time derivative of the FOM solution for the 
            i'th combination of parameter values.
        """

        # Checks
        assert isinstance(p_IC_rollout, float), "type(p_IC_rollout) = %s" % str(type(p_IC_rollout));
        assert isinstance(t, list),             "type(t) = %s" % str(type(t));
        assert p_IC_rollout >= 0.0 and p_IC_rollout <= 1.0, "p_IC_rollout = %f" % p_IC_rollout;

        n_param     : int   = len(t);

        # Other setup.        
        t_Grid_IC_rollout          : list[torch.Tensor]         = [];   # n_train element list whose i'th element is 1d array of times for IC rollout solve.
        n_IC_rollout_frames        : list[int]                  = [];   # n_train element list whose i'th element specifies how many time steps we should simulate forward.
        U_IC_Rollout_Targets       : list[list[torch.Tensor]]   = [];   # n_train element list whose i'th element is n_IC element list whose j'th element is a tensor of shape (n_IC_rollout_frames[i], ...) specifying FOM IC rollout targets


        # -----------------------------------------------------------------------------------------
        # Find t_Grid_IC_rollout and n_IC_rollout_frames.

        for i in range(n_param):
            # Determine the amount of time that passes in the FOM simulation corresponding to the 
            # i'th combination of parameter values. 
            t_i                 : torch.Tensor  = t[i];
            n_t_i               : int           = t_i.shape[0];
            t_0_i               : float         = t_i[0].item();
            t_final_i           : float         = t_i[-1].item();

            # The final IC rollout time for this combination of parameter values. Remember that 
            # t_IC_rollout is the proportion of t_final_i - t_0_i over which we simulate.
            t_IC_rollout_i      : float         = p_IC_rollout*(t_final_i - t_0_i);
            t_IC_rollout_final_i: float         = t_IC_rollout_i + t_0_i;
            LOGGER.info("We will rollout the initial condition for parameter combination #%d to t = %f" % (i, t_IC_rollout_final_i));

            # Now figure out how many time steps occur before t_IC_rollout_final_i.
            num_before_IC_rollout_final_i  : int           = 0;
            for j in range(n_t_i):
                if(t_i[j] > t_IC_rollout_final_i):
                    break; 
                
                num_before_IC_rollout_final_i += 1;
            LOGGER.info("We will rollout the initial condition for parameter combination #%d over %d time steps" % (i, num_before_IC_rollout_final_i));

            # Now define the IC rollout time grid for the i'th combination of parameter values.
            t_Grid_IC_rollout.append(torch.linspace(start = t_0_i, end = t_IC_rollout_final_i, steps = num_before_IC_rollout_final_i));

            # The number of frames we simulate forward from the initial condition
            n_IC_rollout_frames.append(num_before_IC_rollout_final_i);
            LOGGER.info("We will simulate %d time steps from the initial condition for parameter combination #%d." % (num_before_IC_rollout_final_i, i));

            # Fetch the first n_IC_rollout_frames[i] FOM frames.
            U_IC_Rollout_Targets_i : list[torch.Tensor] = [];
            for j in range(self.n_IC):
                U_IC_Rollout_Targets_i.append(self.U_Train[i][j][:num_before_IC_rollout_final_i]);
            U_IC_Rollout_Targets.append(U_IC_Rollout_Targets_i);

        # All done!
        return t_Grid_IC_rollout, n_IC_rollout_frames, U_IC_Rollout_Targets;



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
                 'best_coefs'               : self.best_train_coefs,                # Shape = (n_train, n_coefs).
                 'max_iter'                 : self.max_iter, 
                 'restart_iter'             : self.restart_iter, 
                 'timer'                    : self.timer.export(), 
                 'test_coefs'               : self.test_coefs,
                 'optimizer'                : self.optimizer.state_dict(),
                # Normalization (training-only stats).
                 'normalize'                : self.normalize,
                # Store as plain floats (scalars) for portability.
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

        self.best_train_coefs   : numpy.ndarray             = dict_['best_coefs'];          # Shape = (n_train, n_coefs).
        self.best_epoch         : int                       = dict_['restart_iter'];        # The current encoder_decoder has the best loss so far.
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

        # Set the test coefs.
        with torch.no_grad():
            for i in range(len(self.test_coefs)):
                self.test_coefs[i] = dict_['test_coefs'][i];

        # Load the timer / optimizer. 
        self.timer.load(dict_['timer']);
        self.optimizer.load_state_dict(dict_['optimizer']);
        if (self.device != 'cpu'):
            Move_Optimizer_To_Device(self.optimizer, self.device);

        # All done!
        return;
    