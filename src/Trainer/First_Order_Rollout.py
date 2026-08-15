# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

import  logging;

import  torch;
import  numpy;

from    EncoderDecoder              import  EncoderDecoder;
from    ParameterSpace              import  ParameterSpace;
from    Physics                     import  Physics;
from    LatentDynamics              import  LatentDynamics, InterpolatableLatentDynamics;
from    Utilities.Optimizer         import  Reset_Optimizer;
from    Trainer.Trainer             import  Trainer;
from    Schemas                     import  ExperimentConfig;

# Setup Logger
LOGGER : logging.Logger = logging.getLogger(__name__);



# -------------------------------------------------------------------------------------------------
# Trainer class
# -------------------------------------------------------------------------------------------------

class First_Order_Rollout(Trainer):
    def __init__(self, 
                 physics            : Physics, 
                 encoder_decoder    : EncoderDecoder, 
                 latent_dynamics    : LatentDynamics, 
                 param_space        : ParameterSpace, 
                 config             : ExperimentConfig):
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
          `max_greedy_iter`, `normalize`, `device`, and optional `noise_ratio`.
        - Trainer-specific hyperparameters live under `config['trainer'][config['trainer']['type']]`
          (learning rate, rollout curriculum settings, and loss weights/types). This lets weak
          subclasses reuse this initializer without duplicating rollout setup.

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

        assert isinstance(config, ExperimentConfig), "config must be an ExperimentConfig, got %s" % str(type(config));
        trainer_type : str = config.trainer.type;

        LOGGER.info("Initializing a %s object with First_Order_Rollout setup" % trainer_type); 

        # Fetch the trainer sub-dictionary.
        trainer_config                    = config.trainer;
        sub_config                        = getattr(trainer_config, trainer_type);

        # Call the super class initializer.
        super().__init__(   n_IC            = n_IC,
                            physics         = physics,
                            encoder_decoder = encoder_decoder,
                            latent_dynamics = latent_dynamics,
                            param_space     = param_space,
                            trainer_config  = trainer_config);


        # Fetch training hyperparameters 
        self.lr                     : float     = float(sub_config.lr);               # Learning rate for the optimizer.
        self.gradient_clip          : float     = float(sub_config.gradient_clip);     # Maximum allowable gradient magnitude; will rescale gradients if exceeded.
        self.warmup_epochs          : int       = int(sub_config.warmup_epochs);         # We warmup the learning rate for this many epochs after greedy sampling.


        # Fetch rollout hyperparameters
        self.p_rollout_init         : float     = float(sub_config.p_rollout_init);    # The proportion of the simulated we simulate forward when computing the rollout loss.
        self.rollout_update_freq    : int       = int(sub_config.rollout_update_freq);   # We increase p_rollout after this many iterations.
        self.dp_per_update          : float     = float(sub_config.dp_per_update);    # We increase p_rollout by this much each time we increase it.
        self.max_p_rollout          : float     = float(sub_config.max_p_rollout);     # Maximum value p_rollout is allowed to reach (curriculum ceiling for the frame rollout loss).


        # Rollout supervision (frame-rollout mode; safe for non-autonomous latent dynamics):
        #
        # Randomly select `n_rollouts` rollable start frames per training trajectory per epoch,
        # rollout each one using the *true* absolute-time grid slice t[k:j], and compare full
        # predicted trajectories against the true trajectory slice (no interpolation).
        self.n_rollouts             : int       = int(sub_config.n_rollouts);
        
        # Fetch IC rollout hyperparameters.
        self.p_IC_rollout_init      : float     = float(sub_config.p_IC_rollout_init);    # The proportion of the simulation we simulate forward when computing the IC rollout loss.
        self.IC_rollout_update_freq : int       = int(sub_config.IC_rollout_update_freq);   # We increase p_IC_rollout after this many iterations.
        self.IC_dp_per_update       : float     = float(sub_config.IC_dp_per_update);    # We increase p_IC_rollout by this much each time we increase it.
        self.max_p_IC_rollout       : float     = float(sub_config.max_p_IC_rollout);      # Maximum value p_IC_rollout is allowed to reach (curriculum ceiling for the IC rollout loss).

        # Fetch loss information.
        self.loss_weights           : dict      = sub_config.loss_weights.model_dump(mode = "python", by_alias = True);    # A dictionary housing the weights of the various parts of the loss function.
        self.loss_types             : dict      = sub_config.loss_types.model_dump(mode = "python", by_alias = True);      # A dictionary housing the type of loss function (MSE or MAE) for each part of the loss function.

        # Set up the loss functions.
        LOGGER.info("Setting up the optimizer with a learning rate of %f" % (self.lr));
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
            U_IC_Rollout_Targets.append([self.U_Train[i][0][:num_before_IC_rollout_final_i].to(device = t_i.device)]);

        # All done!
        return t_Grid_IC_rollout, n_IC_rollout_frames, U_IC_Rollout_Targets;



    # ---------------------------------------------------------------------------------------------
    # Iterate.
    # ---------------------------------------------------------------------------------------------


    def Iterate(self, 
                start_iter      : int, 
                end_iter        : int,
                profiler        : torch.profiler.profile | None = None) -> None:
        """
        Run one training round for a first-order system (`n_IC = 1`).

        This method performs gradient-based training over the epoch range
        `[start_iter, end_iter)`. Each epoch:

        1. Encodes the training trajectories to latent states `Z(t)` and decodes them back to
           reconstructed states `U_hat(t)` (reconstruction loss).
        2. Calls `latent_dynamics.compute_losses(...)` to evaluate the latent-dynamics loss using
           the current coefficient dictionaries stored by the LatentDynamics object.
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
        - the LatentDynamics state, including native training coefficient dictionaries

        At the end of the round, `Trainer.train()` loads that checkpoint so that the model and
        latent-dynamics coefficients reflect the best epoch of the round (not necessarily the last
        epoch).

        **Loss logging**

        This method records both per-parameter losses and totals using the base-class helpers
        `_cache_loss(...)`


        -------------------------------------------------------------------------------------------
        Arguments:
        -------------------------------------------------------------------------------------------

        start_iter : int
            The index of the first training iteration. Must have start_iter <= end_iter.

        end_iter : int 
            The index of the last training iteration. Must have start_iter <= end_iter.

        profiler : torch.profiler.profile | None
            An optional torch profiler that can be used to profile Iterate.
            
        -------------------------------------------------------------------------------------------
        Returns:
        -------------------------------------------------------------------------------------------

        None! 
        """
        
        # -------------------------------------------------------------------------------------
        # Setup. 

        # Map trainable state to self's device before constructing the optimizer.  This keeps
        # checkpoint-restored LD coefficients from staying on CPU during a GPU training round.
        device                  : str                       = self.device;
        encoder_decoder_device  : EncoderDecoder            = self.encoder_decoder.to(device);
        self.latent_dynamics.move_trainable_tensors_to_device(device);

        # Reset optimizer.
        optimizer_parameters_list   : list[torch.Tensor] = self._optimizer_parameters();
        self.optimizer = torch.optim.Adam(  optimizer_parameters_list, 
                                            lr              = self.lr, 
                                            weight_decay    = 1.0e-5, 
                                            foreach         = True);
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
        
        last_iter_idx             : int | None         = None;

        U_Train_device          : list[list[torch.Tensor]]  = [];
        t_Train_device          : list[torch.Tensor]        = [];
        for i in range(n_train):
            t_Train_device.append(self.t_Train[i].to(device));
            U_Train_device.append([self.U_Train[i][0].to(device)]);

        # Cache CPU/NumPy time grids once per training round.  These are used only for rollout
        # window selection, so keeping them on CPU avoids repeated GPU->CPU synchronization from
        # t_i.detach().cpu().numpy() inside the epoch loop.
        t_Train_np: list[numpy.ndarray] = [
            self.t_Train[i].detach().cpu().numpy()
            for i in range(n_train)
        ];

        # IC rollout setup
        if(self.loss_weights['IC_rollout'] > 0 and p_IC_rollout > 0):
            self.timer.start("IC Rollout Setup");

            t_Grid_IC_rollout, n_IC_rollout_frames, U_IC_Rollout_Targets = self._IC_rollout_setup(  t            = t_Train_device, 
                                                                                                    p_IC_rollout = p_IC_rollout);
            self.timer.end("IC Rollout Setup"); 

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
            
            self.optimizer.zero_grad(set_to_none=True);
            LOGGER.debug("Zeroed gradients for iteration %d" % (iter + 1));


            # -------------------------------------------------------------------------------------
            # Forward pass + Recon loss
            
            # Initialize losses. 
            loss_recon              : torch.Tensor = torch.zeros(1, dtype = torch.float32, device = device);
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
                    self._cache_loss('recon', recon_loss_ith_param.detach(), ith_param_tuple);
                    
                    LOGGER.debug("Reconstruction Loss (Autoencoder) - complete for parameter combination %d" % i);
                    self.timer.end("Reconstruction Loss");

            # Store total recon loss.
            self._cache_loss('recon', loss_recon.detach());


            # --------------------------------------------------------------------------------
            # Latent Dynamics losses

            self.timer.start("LD/Coefficient/Stability Losses");

            # Compute the latent dynamics losses; this is a dictionary with the same keys as 
            # self.latent_dynamics.loss_weights.
            raw_LD_loss_dict = self.latent_dynamics.compute_losses( 
                                                        Latent_States    = Latent_States, 
                                                        t_Grid           = t_Train_device,
                                                        params           = self.param_space.train_space);

            LD_loss_dict, loss_LD_weighted_sum = self._process_latent_dynamics_losses(
                                                        raw_loss_dict  = raw_LD_loss_dict,
                                                        params         = self.param_space.train_space,
                                                        device         = device);

            self.timer.end("LD/Coefficient/Stability Losses");


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
                    t_i_np  : numpy.ndarray = t_Train_np[i];
                    n_t_i   : int           = t_i.shape[0];

                    if n_t_i < 2:
                        continue;

                    # Rollout duration for this parameter combination.
                    t0      : float = float(t_i_np[0]);
                    tf      : float = float(t_i_np[-1]);
                    dur     : float = float(p_rollout * (tf - t0));
                    if dur <= 0.0:
                        continue;
                    
                    # Find the set of rollable frames.
                    rollable    = numpy.where(t_i_np + dur <= tf)[0];
                    if rollable.size == 0:
                        continue;
                    
                    # Pick out which frames we will roll out.
                    n_roll_i    = min(int(self.n_rollouts), int(rollable.size));
                    start_idx   = numpy.random.choice(rollable, size = n_roll_i, replace = False);

                    # Set up buffers to hold rolled out Z's (along with the associated latent and 
                    # FOM targets) and associated book-keeping (in lengths). Lengths specifies the 
                    # number of rolled out frames in each "window". We can use these buffers to 
                    # cut the number of decodes per parameter to 1, improving runtime.
                    Z_pred_windows      : list[torch.Tensor] = []
                    Z_tgt_windows       : list[torch.Tensor] = []
                    U_tgt_windows       : list[torch.Tensor] = []
                    lengths             : list[int]          = []

                    # Set up
                    param_i     = self.param_space.train_space[i, :].reshape(1, -1);

                    # Cycle through the frames we plan to rollout.
                    for k in start_idx:
                        k_int           : int   = int(k);
                        t_start         : float = float(t_i_np[k_int]);
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

                        # Pick out the times we will rollout over (we do this with the np 
                        # time grid since simulate uses CPU).
                        t_win_np : numpy.ndarray = t_i_np[k_int:(j_int + 1)];

                        # Fetch the latent and FOM states for this parameter value.
                        Z_0         : torch.Tensor          = Latent_States[i][0];          # (n_t_i, n_z)
                        U_0         : torch.Tensor          = U_Train_device[i][0];         # (n_t_i, ...)

                        # Fetch the latent/FOM targets for this parameter value.
                        Z_tgt_windows.append(Z_0[k_int:(j_int + 1), :]);
                        U_tgt_windows.append(U_0[k_int:(j_int + 1), ...]);

                        # Simulate latent dynamics using the absolute-time grid slice. 
                        # Z_pred_list_all[0][0] has shape (n_t_win, 1, n_z)
                        Z_pred_list_all : list[list[torch.Tensor]] = self.latent_dynamics.simulate(
                            IC     = [[Z_0[k_int:(k_int + 1), :]]],      # one param -> list[list[tensor]] of len n_IC
                            t_Grid = [t_win_np],
                            params = param_i);

                        # Prepare trajectory for decoding
                        Z_pred_i = Z_pred_list_all[0][0];
                        assert Z_pred_i.ndim == 3 and Z_pred_i.shape[1] == 1, f"Expected (n_t, 1, n_z), got {tuple(Z_pred_i.shape)}";
                        Z_pred_windows.append(Z_pred_i.squeeze(1)); # (n_t_win, n_z)
                        lengths.append(Z_pred_i.shape[0]);

                    # Now, concatenate the predicted Z solutions and decode!
                    assert len(Z_pred_windows) == len(Z_tgt_windows) == len(U_tgt_windows) == len(lengths) == n_roll_i;
                    Z_pred_cat          : torch.Tensor  = torch.cat(Z_pred_windows, dim = 0);
                    assert Z_pred_cat.shape[0] == sum(lengths);
                    U_pred_cat          : torch.Tensor  = encoder_decoder_device.Decode(Z_pred_cat)[0];
                    assert U_pred_cat.shape[0] == Z_pred_cat.shape[0];

                    # Finally, compute the losses for each rollout window.
                    loss_rollout_ROM_i : torch.Tensor   = torch.zeros(1, dtype = torch.float32, device = device);
                    loss_rollout_FOM_i : torch.Tensor   = torch.zeros(1, dtype = torch.float32, device = device);
                    offset             : int            = 0;
                    for len_i, Z_tgt_i, U_tgt_i, Z_pred_i in zip(lengths, Z_tgt_windows, U_tgt_windows, Z_pred_windows):
                        # Fetch the predictions for this rollout window.
                        U_pred_i = U_pred_cat[offset:offset + len_i]
                        offset  += len_i

                        assert Z_tgt_i.shape[0] == U_tgt_i.shape[0] == Z_pred_i.shape[0] == U_pred_i.shape[0];

                        # Accumulate losses over all IC components.
                        diff_ROM = Z_tgt_i - Z_pred_i;
                        diff_FOM = U_pred_i - U_tgt_i;

                        if self.loss_types['rollout'] == "MSE":
                            loss_rollout_ROM_i = loss_rollout_ROM_i + torch.mean(diff_ROM**2);
                            loss_rollout_FOM_i = loss_rollout_FOM_i + torch.mean(diff_FOM**2);
                        elif self.loss_types['rollout'] == "MAE":
                            loss_rollout_ROM_i = loss_rollout_ROM_i + torch.mean(torch.abs(diff_ROM));
                            loss_rollout_FOM_i = loss_rollout_FOM_i + torch.mean(torch.abs(diff_FOM));
                        else:
                            raise ValueError("Invalid rollout loss type: %s" % self.loss_types['rollout']);

                    assert offset == U_pred_cat.shape[0];

                    # Average across sampled rollouts (and across components implicitly by summation).
                    loss_rollout_ROM_ith_param = loss_rollout_ROM_i / float(n_roll_i)
                    loss_rollout_FOM_ith_param = loss_rollout_FOM_i / float(n_roll_i)

                    # Accumulate totals.
                    loss_rollout_ROM += loss_rollout_ROM_ith_param
                    loss_rollout_FOM += loss_rollout_FOM_ith_param

                    # Log loss for this combination of parameters
                    param_tuple = tuple(self.param_space.train_space[i, :])
                    self._cache_loss('rollout_ROM', loss_rollout_ROM_ith_param.detach(), param_tuple);
                    self._cache_loss('rollout_FOM', loss_rollout_FOM_ith_param.detach(), param_tuple);

                # Log total rollout loss.
                self._cache_loss('rollout_ROM', loss_rollout_ROM.detach());
                self._cache_loss('rollout_FOM', loss_rollout_FOM.detach());

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
                    
                    # Simulate the latent dynamics forward in time
                    Z_IC_Rollout_i    : list[list[torch.Tensor]]  = self.latent_dynamics.simulate(  IC      = [[Z_IC_i]], 
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
                    self._cache_loss('IC_rollout_ROM', loss_IC_rollout_ROM_ith_param.detach(), param_tuple);
                    self._cache_loss('IC_rollout_FOM', loss_IC_rollout_FOM_ith_param.detach(), param_tuple);

                # Store total IC rollout loss.
                self._cache_loss('IC_rollout_ROM', loss_IC_rollout_ROM.detach());
                self._cache_loss('IC_rollout_FOM', loss_IC_rollout_FOM.detach());

                LOGGER.debug("IC Rollout Loss (Autoencoder) - complete");
                self.timer.end("IC Rollout Loss");


            # --------------------------------------------------------------------------------
            # Total loss

            loss_rollout    : torch.Tensor  = loss_rollout_ROM    + loss_rollout_FOM;
            loss_IC_rollout : torch.Tensor  = loss_IC_rollout_ROM + loss_IC_rollout_FOM;


            # Compute the final loss.
            LOGGER.debug("Computing total loss (Autoencoder)");
            loss = (self.loss_weights['recon']      * loss_recon + 
                    self.loss_weights['rollout']    * loss_rollout + 
                    self.loss_weights['IC_rollout'] * loss_IC_rollout +
                    loss_LD_weighted_sum);
            self._cache_loss('total', loss.detach());
            LOGGER.debug("Total loss (Autoencoder) computed");



            # Record coefficient scale and the most recent epoch index for fallback checkpointing.
            if isinstance(self.latent_dynamics, InterpolatableLatentDynamics):
                with torch.no_grad():
                    coef_tensors_report = self.latent_dynamics.trainable_tensors();
                    train_coefs_flat_report = torch.cat([c.reshape(-1) for c in coef_tensors_report]);
                    max_train_coef = float(torch.abs(train_coefs_flat_report).max().item());
            last_iter_idx = int(iter);


            # -------------------------------------------------------------------------------------
            # Backward Pass

            self.timer.start("Backwards Pass");
            LOGGER.debug("Backward Pass - start (iteration %d)" % (iter + 1));

            #  Run back propagation and update the encoder_decoder parameters. 
            # Note: optimizer.zero_grad() is already called at the start of the iteration (line 373)
            loss.backward();
            
            # Clip gradients to prevent explosion during latent dynamics rollout.
            grad_norm = torch.nn.utils.clip_grad_norm_(
                optimizer_parameters_list,
                max_norm = self.gradient_clip,
                foreach  = True,
            )
            
            # Log if gradient clipping activates (indicates potential instability)
            if grad_norm > self.gradient_clip:
                LOGGER.warning("Gradient norm %.2f exceeded threshold, clipped to %f (iter %d)" % (grad_norm, self.gradient_clip, iter + 1));
            
            LOGGER.debug("Backward Pass - backward() complete, calling optimizer.step()");
            self.optimizer.step();
            LOGGER.debug("Backward Pass - complete (iteration %d)" % (iter + 1));

            # Flush all cached loss tensors after the optimizer update. This performs one batched
            # device-to-CPU scalar transfer for loss tracking, checkpoint decisions, and reporting.
            flushed_losses = self._flush_loss_cache(iter + 1);
            loss_value = flushed_losses[('total', 'total')];

            # Check if we hit a new minimum loss. If so, make a checkpoint, record the loss and 
            # the iteration number. 
            # NOTE: Skip checkpointing during warmup period to avoid saving "lucky" early epochs
            # that benefit from distribution shift before encoder_decoder has adapted.
            
            if loss_value < best_loss:
                if epochs_in_round >= self.warmup_epochs:
                    LOGGER.info("Got a new lowest loss (%f) on epoch %d" % (loss_value, iter + 1));

                    # Save the full checkpoint (model state + train/test coefficients).
                    self._Save_Checkpoint(encoder_decoder = encoder_decoder_device,
                                          iter            = int(iter));
                    checkpoint_saved        = True;

                    # Update the best set of parameters. 
                    self.best_epoch         = int(iter);
                    best_loss               = loss_value;
                else:
                    LOGGER.debug("Skipping checkpoint during warmup period (epoch %d/%d in round, warmup ends at %d)" % 
                               (epochs_in_round, end_iter - start_iter, self.warmup_epochs));

            self.timer.end("Backwards Pass");
        
            


            # -------------------------------------------------------------------------------------
            # Report Results from this iteration 

            self.timer.start("Report");

            # Report the current iteration number and losses
            info_str : str = "Iter: %05d/%d, Total: %3.10f" % (iter + 1, self.max_iter, loss_value);
            if(self.loss_weights['recon'] > 0):         info_str += ", Recon: %3.6f"                            % flushed_losses[('recon', 'total')];
            if(self.loss_weights['rollout'] > 0):       info_str += ", Roll FOM: %3.6f, Roll ROM: %3.6f"        % (flushed_losses.get(('rollout_FOM', 'total'), 0.0),    flushed_losses.get(('rollout_ROM', 'total'), 0.0));
            if(self.loss_weights['IC_rollout'] > 0):    info_str += ", IC Roll FOM: %3.6f, IC Roll ROM: %3.6f"  % (flushed_losses.get(('IC_rollout_FOM', 'total'), 0.0), flushed_losses.get(('IC_rollout_ROM', 'total'), 0.0));
            for key, value in LD_loss_dict.items():
                if self.latent_dynamics.loss_weights[key] > 0:
                    info_str += ", %s: %3.6f"   % flushed_losses[(key, 'total')];
            if isinstance(self.latent_dynamics, InterpolatableLatentDynamics): 
                info_str += ", max|c|: %.3f" % max_train_coef;
            LOGGER.info(info_str);

            self.timer.end("Report");
            
            LOGGER.debug("Completed training iteration %d/%d" % (iter + 1, end_iter));
            self.timer.end("train_step");

            # Step the profiler.
            if profiler is not None:
                profiler.step();

        
        # Ensure we wrote a checkpoint for this round. If warmup prevented checkpointing, fall
        # back to saving the final epoch of this round.
        if checkpoint_saved == False:
            assert last_iter_idx is not None;
            LOGGER.warning("No checkpoint saved during this round (likely warmup-only). Saving final epoch checkpoint instead.");
            self._Save_Checkpoint(encoder_decoder = encoder_decoder_device,
                                  iter            = int(last_iter_idx));
            self.best_epoch       = int(last_iter_idx);

        # All done!
        return;
