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
from    ParameterSpace              import  ParameterSpace;
from    Physics                     import  Physics;
from    LatentDynamics              import  LatentDynamics;
from    Optimizer                   import  Reset_Optimizer;
from    Trainer                     import  Trainer;

# Setup Logger
LOGGER : logging.Logger = logging.getLogger(__name__);



# -------------------------------------------------------------------------------------------------
# Trainer class
# -------------------------------------------------------------------------------------------------

class Second_Order_Rollout(Trainer):
    def __init__(self, 
                 physics            : Physics, 
                 encoder_decoder    : EncoderDecoder, 
                 latent_dynamics    : LatentDynamics, 
                 param_space        : ParameterSpace, 
                 config             : dict):
        """
        This defines a Trainer sub-class which is designed to run Rollouts on latent dynamics
        that have one initial condition (n_IC = 1). It uses the following loss functions:

            - reconstruction (autoencoder)
            - coefficient (Frobenius norm of latent dynamics coefficients)
            - LD (standard LaSDI latent dynamics loss)
            - stability (maximum eigenvalue of the symmetric part of the latent dynamics system matrix)
            - rollout (standard rollout loss)
            - IC_rollout (initial condition rollout loss)

        It can only be paired with Latent_Dynamics, Physics, and EncoderDecoder sub-classes which 
        also have n_IC = 1.

        **Configuration format**

        This trainer follows the standard Higher-Order-LaSDI convention:

        - `config['trainer']` contains base trainer settings such as `n_iter`, `max_iter`,
          `max_greedy_iter`, `normalize`, and `device`.
        - Subclass-specific hyperparameters live under `config['trainer']['First_Order_Rollout']`
          (learning rate, rollout curriculum settings, and loss weights/types).

        **Coefficient semantics**

        This trainer assumes coefficients are *trainable* by default. It stores a learnable
        matrix `self.test_coefs` of shape `(n_test, n_coefs)`, i.e. a coefficient row for every
        point in the test parameter space.

        During training, the training set is a subset of the test set. For each training parameter
        value, we locate its index in the test space and use the corresponding row of
        `self.test_coefs` inside the loss. This means:

        - gradients flow into `self.test_coefs` for rows corresponding to training parameters
        - the *full* `self.test_coefs` matrix is checkpointed so that, after each training round,
          the trainer can restore the best epoch's coefficients (used by greedy sampling / GP fits)

        **Checkpointing**

        The implementation of `Iterate(...)` is responsible for calling the base-class
        `_Save_Checkpoint(...)` method whenever a new best epoch is found.


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
        n_IC                        =  1;
        assert latent_dynamics.n_IC == n_IC, "latent_dynamics.n_IC = %d, n_IC = %d" % (latent_dynamics.n_IC, n_IC);
        assert encoder_decoder.n_IC == n_IC, "encoder_decoder.n_IC = %d, n_IC = %d" % (encoder_decoder.n_IC, n_IC);
        assert physics.n_IC         == n_IC, "physics.n_IC = %d, n_IC = %d" % (physics.n_IC, n_IC);
        self.n_IC                   =  n_IC; 

        assert 'trainer' in config,                                 "config must contain a 'trainer' sub-dictionary";
        assert 'type' in config['trainer'],                         "trainer dictionary must contain a 'type' attribute";
        assert config['trainer']['type'] == "First_Order_Rollout",  "config['trainer']['type'] = %s, should be First_Order_Rollout" % config['trainer']['type'];
        assert "First_Order_Rollout" in config['trainer'],          "First_Order_Rollout must be in config['trainer']";

        LOGGER.info("Initializing a First_Order_Rollout object"); 

        # Fetch the trainer sub-dictionary.
        trainer_config          : dict      = config['trainer'];
        sub_config              : dict      = trainer_config['First_Order_Rollout'];

        # Call the super class initializer.
        super().__init__(   n_IC            = n_IC,
                            physics         = physics,
                            encoder_decoder = encoder_decoder,
                            latent_dynamics = latent_dynamics,
                            param_space     = param_space,
                            trainer_config  = trainer_config);


        # Fetch training hyperparameters 
        self.lr                     : float     = float(sub_config.get('lr', 0.001));               # Learning rate for the optimizer.
        self.gradient_clip          : float     = float(sub_config.get('gradient_clip', 10.0));     # Maximum allowable gradient magnitude; will rescale gradients if exceeded.
        self.warmup_epochs          : int       = int(sub_config.get('warmup_epochs', 40));         # We warmup the learning rate for this many epochs after greedy sampling.


        # Fetch rollout hyperparameters
        self.p_rollout_init         : float     = float(sub_config.get('p_rollout_init', 0.01));    # The proportion of the simulated we simulate forward when computing the rollout loss.
        self.rollout_update_freq    : int       = int(sub_config.get('rollout_update_freq', 10));   # We increase p_rollout after this many iterations.
        self.dp_per_update          : float     = float(sub_config.get('dp_per_update', 0.005));    # We increase p_rollout by this much each time we increase it.
        self.max_p_rollout          : float     = float(sub_config.get('max_p_rollout', 0.75));     # Maximum value p_rollout is allowed to reach (curriculum ceiling for the frame rollout loss).


        # Rollout supervision (frame-rollout mode; safe for non-autonomous latent dynamics):
        #
        # Randomly select `n_rollouts` rollable start frames per training trajectory per epoch,
        # rollout each one using the *true* absolute-time grid slice t[k:j], and compare full
        # predicted trajectories against the true trajectory slice (no interpolation).
        assert 'n_rollouts' in sub_config, "First_Order_Rollout config must include `n_rollouts` (int > 0) for rollout supervision";
        self.n_rollouts             : int       = int(sub_config['n_rollouts']);
        assert self.n_rollouts > 0, "trainer.n_rollouts must be > 0";
        
        # Fetch IC rollout hyperparameters.
        self.p_IC_rollout_init      : float     = float(sub_config.get('p_IC_rollout_init', 0.01));    # The proportion of the simulation we simulate forward when computing the IC rollout loss.
        self.IC_rollout_update_freq : int       = int(sub_config.get('IC_rollout_update_freq', 10));   # We increase p_IC_rollout after this many iterations.
        self.IC_dp_per_update       : float     = float(sub_config.get('IC_dp_per_update', 0.005));    # We increase p_IC_rollout by this much each time we increase it.
        self.max_p_IC_rollout       : float     = float(sub_config.get('max_p_IC_rollout', 1.0));      # Maximum value p_IC_rollout is allowed to reach (curriculum ceiling for the IC rollout loss).

        # Fetch loss information.
        self.loss_weights           : dict      = sub_config['loss_weights'];                   # A dictionary housing the weights of the various parts of the loss function.
        self.loss_types             : dict      = sub_config['loss_types'];                     # A dictionary housing the type of loss function (MSE or MAE) for each part of the loss function.

        # Set default values for 'coef', 'stab' entries of loss_weights
        if 'coef' not in self.loss_weights.keys():
            self.loss_weights['coef'] = 0.0;
        if 'stab' not in self.loss_weights.keys():
            self.loss_weights['stab'] = 0.0;

        # Set up the optimizer and loss function.
        LOGGER.info("Setting up the optimizer with a learning rate of %f" % (self.lr));
        self.optimizer          : Optimizer = torch.optim.Adam(list(encoder_decoder.parameters()) + [self.test_coefs], lr = self.lr, weight_decay = 1.0e-5);
        self.MSE                            = torch.nn.MSELoss(reduction = 'mean');
        self.MAE                            = torch.nn.L1Loss(reduction = 'mean');

        # All done!
        return;



    # ---------------------------------------------------------------------------------------------
    # _IC_rollout_setup
    # ---------------------------------------------------------------------------------------------

    def _IC_rollout_setup( self, 
                           t            : list[torch.Tensor], 
                           p_IC_rollout : float) -> tuple[list[torch.Tensor], list[int], list[list[torch.Tensor]]]:
        """
        An internal function that sets up the IC rollout loss. This simulates forward from the FOM
        initial conditions. The user should not call this 
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
            LOGGER.info("We will rollout the initial condition for parameter combination #%d to t <= %f" % (i, t_IC_rollout_final_i));

            # Now figure out how many time steps occur before t_IC_rollout_final_i.
            num_before_IC_rollout_final_i  : int           = 0;
            for j in range(n_t_i):
                if(t_i[j] > t_IC_rollout_final_i):
                    break; 
                
                num_before_IC_rollout_final_i += 1;
            LOGGER.info("We will rollout the initial condition for parameter combination #%d over %d time steps" % (i, num_before_IC_rollout_final_i));

            # Now define the IC rollout time grid for the i'th combination of parameter values.
            #
            # IMPORTANT:
            # Use the *true* FOM time stamps for the first num_before_IC_rollout_final_i frames
            # rather than a linspace. This keeps the rollout simulation times aligned with
            # U_IC_Rollout_Targets, which are taken directly from U_Train[i][:num_before...].
            #
            # This is especially important for time-dependent / switched latent dynamics models
            # (e.g., SwitchSINDy), where the absolute time values affect which dynamics regime
            # (laser on/off) applies.
            assert num_before_IC_rollout_final_i > 0, "IC rollout produced 0 time steps (unexpected when p_IC_rollout > 0)";
            t_Grid_IC_rollout.append(t_i[:num_before_IC_rollout_final_i].clone());

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



    # ---------------------------------------------------------------------------------------------
    # Iterate.
    # ---------------------------------------------------------------------------------------------


    def Iterate(self, 
                start_iter      : int, 
                end_iter        : int) -> None:
        """
        Run one training round for a first-order system (`n_IC = 1`).

        This method performs gradient-based training over the epoch range
        `[start_iter, end_iter)`. Each epoch:

        1. Encodes the training trajectories to latent states `Z(t)` and decodes them back to
           reconstructed states `U_hat(t)` (reconstruction loss).
        2. Calls `latent_dynamics.calibrate(...)` to evaluate the latent-dynamics loss given the
           current coefficient values (which are taken from `self.test_coefs` at the rows
           corresponding to training parameters).
        3. Optionally computes rollout-based losses by simulating the latent dynamics forward in
           time and comparing decoded trajectories against either:
              - trajectory slices (frame rollouts), and/or
              - rollouts from true initial conditions (IC rollouts).
        4. Aggregates the weighted loss, performs backpropagation, gradient clipping, and an
           optimizer step.

        **Checkpointing (important)**

        Whenever this method finds a new best (lowest) loss *within this round*, it calls the
        base-class `_Save_Checkpoint(...)` method. The checkpoint stores:

        - the EncoderDecoder parameters
        - the best training coefficients (`train_coefs`, shape `(n_train, n_coefs)`)
        - the full test-space coefficient matrix (`self.test_coefs`, shape `(n_test, n_coefs)`)

        At the end of the round, `Trainer.train()` loads that checkpoint so that
        `trainer.test_coefs` reflects the best epoch of the round (not necessarily the last epoch).

        **Loss logging**

        This method records both per-parameter losses and totals using the base-class helpers
        `_store_loss_by_param(...)` and `_store_total_loss(...)`.


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
        
        # -------------------------------------------------------------------------------------
        # Setup. 

        # Reset optimizer.
        Reset_Optimizer(self.optimizer);

        # Fetch parameters. Note that p_rollout and p_IC_rollout can be negative.
        # IMPORTANT: Calculate rollout proportions using epochs within CURRENT round (not accumulated restart_iter).
        # This ensures rollout starts small after each greedy sampling and gradually increases.
        n_train                 : int               = self.param_space.n_train();
        epochs_in_round         : int               = 0;  # Will be updated each iteration
        p_rollout               : float             = min(self.max_p_rollout,    self.p_rollout_init    + self.dp_per_update   *(epochs_in_round//self.rollout_update_freq));
        p_IC_rollout            : float             = min(self.max_p_IC_rollout, self.p_IC_rollout_init + self.IC_dp_per_update*(epochs_in_round//self.IC_rollout_update_freq));
        best_loss               : float             = numpy.inf;                    # Stores the lowest loss we get in this round of training.
        checkpoint_saved        : bool              = False;                        # Ensure we save at least one checkpoint per round.
        last_train_coefs_detached : numpy.ndarray | None = None;
        last_iter_idx             : int | None         = None;

        # Map everything to self's device.
        device                  : str                       = self.device;
        encoder_decoder_device  : EncoderDecoder            = self.encoder_decoder.to(device);

        U_Train_device          : list[list[torch.Tensor]]  = [];
        t_Train_device          : list[torch.Tensor]        = [];
        for i in range(n_train):
            t_Train_device.append(self.t_Train[i].to(device));
            
            ith_U_Train_device  : list[torch.Tensor] = [];
            for j in range(self.n_IC):
                ith_U_Train_device.append(self.U_Train[i][j].to(device));
            U_Train_device.append(ith_U_Train_device);

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
        # Run the iterations!
        for iter in range(start_iter, end_iter):
            self.timer.start("train_step");
            LOGGER.debug("=" * 80);
            LOGGER.debug("Starting training iteration %d/%d" % (iter + 1, end_iter));


            # -------------------------------------------------------------------------------------
            # Warmup the learning rate for the first few epochs after greedy sampling.
            # NOTE: epochs_in_round will be recalculated later for rollout updates.

            epochs_in_round     : int = iter - self.restart_iter;  # Progress within current training round
            if self.warmup_epochs > 0 and epochs_in_round < self.warmup_epochs:
                # Reduce LR for warmup period
                warmup_scale = 0.1 + 0.9 * (float(epochs_in_round) / float(self.warmup_epochs));
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr * warmup_scale;
                LOGGER.info("Warmup: LR scaled to %.6f (epoch %d/%d in round)" % (self.lr * warmup_scale, epochs_in_round, end_iter - start_iter));
            else:
                # Restore full LR
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr;


            # -------------------------------------------------------------------------------------
            # Check if we need to update p_rollout (curriculum over horizon length).
            # NOTE: Use epochs_in_round (not iter) to reset rollout progression each training round.

            if(self.loss_weights['rollout'] > 0 and epochs_in_round > 0 and ((epochs_in_round % self.rollout_update_freq) == 0)):
                p_rollout = min(self.max_p_rollout, self.p_rollout_init + self.dp_per_update*(epochs_in_round//self.rollout_update_freq));
                LOGGER.info("p_rollout is now %f (epoch %d/%d in current round)" % (p_rollout, epochs_in_round, end_iter - start_iter));


            # -------------------------------------------------------------------------------------
            # Check if we need to update IC rollout parameters
            # NOTE: Use epochs_in_round (not iter) to reset IC rollout progression each training round

            if(self.loss_weights['IC_rollout'] > 0 and epochs_in_round > 0 and ((epochs_in_round % self.IC_rollout_update_freq) == 0)):
                self.timer.start("IC Rollout Setup");

                # Recalculate p_IC_rollout based on progress within current round
                p_IC_rollout   = min(self.max_p_IC_rollout, self.p_IC_rollout_init + self.IC_dp_per_update*(epochs_in_round//self.IC_rollout_update_freq));

                LOGGER.info("p_IC_rollout is now %f (epoch %d/%d in current round)" % (p_IC_rollout, epochs_in_round, end_iter - start_iter));

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
            # Set up losses
            
            # Initialize losses. 
            loss_recon              : torch.Tensor = torch.zeros(1, dtype = torch.float32, device = device);
            loss_LD                 : torch.Tensor = torch.zeros(1, dtype = torch.float32, device = device);
            loss_stab               : torch.Tensor = torch.zeros(1, dtype = torch.float32, device = device);
            loss_coef               : torch.Tensor = torch.zeros(1, dtype = torch.float32, device = device);
            loss_rollout_FOM        : torch.Tensor = torch.zeros(1, dtype = torch.float32, device = device);
            loss_rollout_ROM        : torch.Tensor = torch.zeros(1, dtype = torch.float32, device = device);
            loss_IC_rollout_FOM     : torch.Tensor = torch.zeros(1, dtype = torch.float32, device = device);
            loss_IC_rollout_ROM     : torch.Tensor = torch.zeros(1, dtype = torch.float32, device = device);

            # Setup. 
            Latent_States           : list[list[torch.Tensor]]  = [];       # len = n_train. i'th element is 1 element list of (n_t_i, n_z) arrays.

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
                if iter % 100 == 0 or iter == start_iter:  # Log every 100 iters and first iter
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
            train_coefs, loss_LD_list, loss_coef_list, loss_stab_list = self.latent_dynamics.calibrate(
                                                                            Latent_States    = Latent_States, 
                                                                            t_Grid           = t_Train_device,
                                                                            input_coefs      = train_coefs_list,
                                                                            loss_type        = self.loss_types['LD'],
                                                                            params           = self.param_space.train_space);

            # Log coefficient statistics to diagnose constant dynamics issue
            if iter % 100 == 0 or iter == start_iter:  # Log every 100 iters and first iter
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
                self._store_loss_by_param('coef', param_tuple, iter + 1, loss_coef_list[i].item());


            # Compute the total loss.
            loss_LD   = torch.sum(torch.stack(loss_LD_list));
            loss_stab = torch.sum(torch.stack(loss_stab_list));
            loss_coef = torch.sum(torch.stack(loss_coef_list));

            # Append the total loss to loss_by_param.
            self._store_total_loss('LD', iter + 1, loss_LD.item());
            self._store_total_loss('stab', iter + 1, loss_stab.item());
            self._store_total_loss('coef', iter + 1, loss_coef.item());


            self.timer.end("Calibration");


            # ---------------------------------------------------------------------------------
            # Rollout loss. Note that we need the coefficients before we can compute this.
            
            if(self.loss_weights['rollout'] > 0 and p_rollout > 0):
                self.timer.start("Rollout Loss");
                LOGGER.debug("Rollout Loss (Autoencoder) - start");

                # For each training parameter combination, randomly select a small number of
                # rollable start frames, rollout each one on the *true* absolute time grid,
                # and compare full trajectories (no interpolation / no random target points).
                for i in range(n_train):
                    t_i     : torch.Tensor  = t_Train_device[i];
                    n_t_i   : int           = t_i.shape[0];

                    if n_t_i < 2:
                        continue;

                    # Rollout duration for this parameter combination.
                    t0      : float = float(t_i[0].item());
                    tf      : float = float(t_i[-1].item());
                    dur     : float = float(p_rollout * (tf - t0));
                    if dur <= 0.0:
                        continue;
                    
                    # Find the set of rollable frames.
                    t_i_np      = t_i.detach().cpu().numpy();
                    rollable    = numpy.where(t_i_np + dur <= tf)[0];
                    if rollable.size == 0:
                        continue;
                    
                    # Pick out which frames we will roll out.
                    n_roll_i    = min(int(self.n_rollouts), int(rollable.size));
                    start_idx   = numpy.random.choice(rollable, size = n_roll_i, replace = False);

                    # Set up
                    loss_rollout_ROM_i : torch.Tensor = torch.zeros(1, dtype = torch.float32, device = device);
                    loss_rollout_FOM_i : torch.Tensor = torch.zeros(1, dtype = torch.float32, device = device);

                    param_i     = self.param_space.train_space[i, :].reshape(1, -1);
                    coef_i      = train_coefs[i, :].reshape(1, -1);

                    # Cycle through the frames we plan to rollout.
                    for k in start_idx:
                        k_int           : int   = int(k);
                        t_start         : float = float(t_i[k_int].item());
                        t_end_target    : float = t_start + dur;

                        # Find j: index of time closest to t_end_target; the time closest to 
                        # t_end_target will have the smallest absolute distance from 
                        # t_end_target, so j for which t_i_np[j] - t_end_target is smallest
                        # is the index we want. Ensure j > k when possible.
                        j_int = int(numpy.argmin(numpy.abs(t_i_np - t_end_target)));
                        if j_int < k_int:
                            j_int = k_int;
                        if j_int == k_int and (k_int + 1) < n_t_i:
                            j_int = k_int + 1;

                        # Pick out the times we will rollout over.
                        t_win : torch.Tensor = t_i[k_int:(j_int + 1)];

                        # Targets (ROM + FOM) on the same time grid slice.
                        Z_tgt_list : list[torch.Tensor] = [];
                        U_tgt_list : list[torch.Tensor] = [];
                        Z0_list    : list[torch.Tensor] = [];
                        for d in range(self.n_IC):
                            Z_d = Latent_States[i][d];                      # (n_t_i, n_z)
                            U_d = U_Train_device[i][d];                     # (n_t_i, ...)
                            Z_tgt_list.append(Z_d[k_int:(j_int + 1), :]);
                            U_tgt_list.append(U_d[k_int:(j_int + 1), ...]);
                            Z0_list.append(Z_d[k_int:(k_int + 1), :]);      # (1, n_z)

                        # Simulate latent dynamics using the absolute-time grid slice.
                        Z_pred_list_all : list[list[torch.Tensor]] = self.latent_dynamics.simulate(
                            coefs  = coef_i,
                            IC     = [Z0_list],      # one param -> list[list[tensor]] of len n_IC
                            t_Grid = [t_win],
                            params = param_i);

                        # Prepare trajectory for decoding
                        Z_pred_comp : list[torch.Tensor] = [];
                        for d in range(self.n_IC):
                            Z_pd = Z_pred_list_all[0][d];
                            assert Z_pd.ndim == 3 and Z_pd.shape[1] == 1, f"Expected (n_t, 1, n_z), got {tuple(Z_pd.shape)}";
                            Z_pred_comp.append(Z_pd.squeeze(1));  # (n_t_win, n_z)

                        # Decoded predicted trajectory.
                        U_pred_tuple : tuple[torch.Tensor, ...] = encoder_decoder_device.Decode(*Z_pred_comp);

                        # Accumulate losses over all IC components.
                        for d in range(self.n_IC):
                            diff_ROM = Z_tgt_list[d] - Z_pred_comp[d];
                            diff_FOM = U_pred_tuple[d] - U_tgt_list[d];

                            if self.loss_types['rollout'] == "MSE":
                                loss_rollout_ROM_i = loss_rollout_ROM_i + torch.mean(diff_ROM**2);
                                loss_rollout_FOM_i = loss_rollout_FOM_i + torch.mean(diff_FOM**2);
                            elif self.loss_types['rollout'] == "MAE":
                                loss_rollout_ROM_i = loss_rollout_ROM_i + torch.mean(torch.abs(diff_ROM));
                                loss_rollout_FOM_i = loss_rollout_FOM_i + torch.mean(torch.abs(diff_FOM));
                            else:
                                raise ValueError("Invalid rollout loss type: %s" % self.loss_types['rollout']);

                    # Average across sampled rollouts (and across components implicitly by summation).
                    loss_rollout_ROM_ith_param = loss_rollout_ROM_i / float(n_roll_i)
                    loss_rollout_FOM_ith_param = loss_rollout_FOM_i / float(n_roll_i)

                    # Accumulate totals.
                    loss_rollout_ROM += loss_rollout_ROM_ith_param
                    loss_rollout_FOM += loss_rollout_FOM_ith_param

                    # Log loss for this combination of parameters
                    param_tuple = tuple(self.param_space.train_space[i, :])
                    self._store_loss_by_param('rollout_ROM', param_tuple, iter + 1, loss_rollout_ROM_ith_param.item())
                    self._store_loss_by_param('rollout_FOM', param_tuple, iter + 1, loss_rollout_FOM_ith_param.item())

                # Log total rollout loss.
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
                    self.loss_weights['stab']       * loss_stab + 
                    self.loss_weights['coef']       * loss_coef);
            self._store_total_loss('total', iter + 1, loss.item());
            LOGGER.debug("Total loss (Autoencoder) computed: %f" % loss.item());



            # Convert coefs to numpy and find the maximum element.
            # Store a detached copy for reporting (needed after backprop), but keep original for gradient flow
            with torch.no_grad():
                train_coefs_detached    : numpy.ndarray = train_coefs.detach().cpu().numpy();                # Shape = (n_train, n_coefs).
                max_train_coef          : numpy.float32 = numpy.abs(train_coefs_detached).max();
                last_train_coefs_detached = train_coefs_detached;
                last_iter_idx             = int(iter);




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

                    # Save the full checkpoint (model state + train/test coefficients).
                    self._Save_Checkpoint(encoder_decoder = encoder_decoder_device,
                                          train_coefs     = train_coefs_detached,
                                          test_coefs      = self.test_coefs,
                                          iter            = int(iter));
                    checkpoint_saved        = True;

                    # Update the best set of parameters. 
                    self.best_train_coefs   = train_coefs_detached.copy();     # Shape = (n_train, n_coefs).
                    self.best_epoch         = int(iter);
                    best_loss               = loss.item();
                else:
                    LOGGER.debug("Skipping checkpoint during warmup period (epoch %d/%d in round, warmup ends at %d)" % 
                               (epochs_in_round, end_iter - start_iter, self.warmup_epochs));

            self.timer.end("Backwards Pass");
            


            # -------------------------------------------------------------------------------------
            # Report Results from this iteration 

            self.timer.start("Report");

            # Report the current iteration number and losses
            info_str : str = "Iter: %05d/%d, Total: %3.10f" % (iter + 1, self.max_iter, loss.item());
            if(self.loss_weights['recon'] > 0):         info_str += ", Recon: %3.6f"                            % loss_recon.item();
            if(self.loss_weights['rollout'] > 0):       info_str += ", Roll FOM: %3.6f, Roll ROM: %3.6f"        % (loss_rollout_FOM.item(),    loss_rollout_ROM.item());
            if(self.loss_weights['IC_rollout'] > 0):    info_str += ", IC Roll FOM: %3.6f, IC Roll ROM: %3.6f"  % (loss_IC_rollout_FOM.item(), loss_IC_rollout_ROM.item());
            if(self.loss_weights['LD'] > 0):            info_str += ", LD: %3.6f"                               % loss_LD.item();
            if(self.loss_weights['stab'] > 0):          info_str += ", Stab: %3.6f"                             % loss_stab.item();
            if(self.loss_weights['coef'] > 0):          info_str += ", Coef: %3.6f"                             % loss_coef.item();
            info_str += ", max|c|: %.3f" % max_train_coef;
            LOGGER.info(info_str);

            self.timer.end("Report");
            
            LOGGER.debug("Completed training iteration %d/%d" % (iter + 1, end_iter));
            self.timer.end("train_step");
        
        # Ensure we wrote a checkpoint for this round. If warmup prevented checkpointing, fall
        # back to saving the final epoch of this round.
        if checkpoint_saved == False:
            assert last_train_coefs_detached is not None and last_iter_idx is not None;
            LOGGER.warning("No checkpoint saved during this round (likely warmup-only). Saving final epoch checkpoint instead.");
            self._Save_Checkpoint(encoder_decoder = encoder_decoder_device,
                                  train_coefs     = last_train_coefs_detached,
                                  test_coefs      = self.test_coefs,
                                  iter            = int(last_iter_idx));
            self.best_train_coefs = last_train_coefs_detached.copy();
            self.best_epoch       = int(last_iter_idx);

        # All done!
        return;
