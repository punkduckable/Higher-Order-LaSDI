# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

import  logging;
import  time;

import  torch;
import  numpy;

from    HLaSDI.EncoderDecoder                  import  EncoderDecoder;
from    HLaSDI.ParameterSpace                  import  ParameterSpace;
from    HLaSDI.Physics                         import  Physics;
from    HLaSDI.LatentDynamics                  import  LatentDynamics, WeakLatentDynamics, InterpolatableLatentDynamics, LD_Loss_Container;
from    HLaSDI.Utilities.Optimizer             import  Reset_Optimizer;
from    HLaSDI.Trainer.Second_Order_Rollout    import  Second_Order_Rollout;
from    HLaSDI.Schemas                         import  ExperimentConfig;

# Setup Logger
LOGGER : logging.Logger = logging.getLogger(__name__);



# -------------------------------------------------------------------------------------------------
# Trainer class
# -------------------------------------------------------------------------------------------------

class Second_Order_Weak(Second_Order_Rollout):
    def __init__(self, 
                 physics            : Physics, 
                 encoder_decoder    : EncoderDecoder, 
                 latent_dynamics    : LatentDynamics, 
                 param_space        : ParameterSpace, 
                 config             : ExperimentConfig,
                 run_ID             : str | None = None):
        """
        This defines a Trainer class designed to train second-order dynamics using latent dynamics
        based on the weak formulation.
         
        It is a sub-class of Second_Order_Rollout that is specially designed to work with weak
        forms. Optional data noise is controlled by the base Trainer through top-level
        `trainer.noise_ratio`.
        

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
        
        assert isinstance(config, ExperimentConfig), "config must be an ExperimentConfig, got %s" % str(type(config));
        assert config.trainer.type == "Second_Order_Weak", "config.trainer.type = %s, should be Second_Order_Weak" % config.trainer.type;

        LOGGER.info("Initializing a Second_Order_Weak object"); 

        # Make sure we are set up to work with a weak-form latent dynamics object.
        assert isinstance(latent_dynamics, WeakLatentDynamics), "Second_Order_Weak requires a weak latent dynamics object.";
        assert hasattr(latent_dynamics, "add_weight_functions"), "latent dynamics must have an `add_weight_functions` method";
        assert hasattr(latent_dynamics, "get_test_functions"), "latent dynamics must have a `get_test_functions` method";

        # Call the Second_Order_Rollout initializer. It reads the trainer-specific settings from
        # config['trainer'][config['trainer']['type']], so the Second_Order_Weak config block can
        # be used directly.
        super().__init__(   physics         = physics,
                            encoder_decoder = encoder_decoder,
                            latent_dynamics = latent_dynamics,
                            param_space     = param_space,
                            config          = config,
                            run_ID          = run_ID);

        # All done!
        return;



    # ---------------------------------------------------------------------------------------------
    # Test function methods
    # ---------------------------------------------------------------------------------------------

    def _prepare_weak_form_data(self) -> None:
        r"""
        Build weak-form test functions for every testing parameter value.

        The latent-dynamics object owns the generated tensors. This trainer only supplies the
        parameter value and its time grid.
        """

        assert len(self.t_Test) == self.param_space.n_test(), "t_Test is not initialized or has wrong length";

        # Build weights for the *entire* test space once. Training parameters are a subset of the
        # test space, so this covers compute_losses and avoids needing sampler-specific logic.
        for i in range(self.param_space.n_test()):
            params_i = self.param_space.test_space[i, :];
            t_i : torch.Tensor = self.t_Test[i].to(self.device);
            self.latent_dynamics.add_weight_functions(params_i, t_i);

        LOGGER.info("Prepared weak-form test functions for %d test trajectories" % self.param_space.n_test());
        return;



    # ---------------------------------------------------------------------------------------------
    # Iterate
    # ---------------------------------------------------------------------------------------------

    def Iterate(self, 
                start_iter  : int, 
                end_iter    : int,
                profiler    : torch.profiler.profile | None = None) -> None:
        """
        Run one training round for a second-order system (`n_IC = 2`).

        This trainer is designed for higher-order physics where the state is represented via
        multiple time derivatives (e.g., displacement and velocity). Concretely, each training
        trajectory provides two streams `U_D(t)` and `U_V(t)` and the EncoderDecoder is expected
        to encode/decode these jointly (see `Autoencoder_Pair`).

        Each epoch in `[start_iter, end_iter)` typically performs:

        - Forward passes to obtain latent trajectories and reconstructions
        - Latent dynamics/coefficient/stability loss evaluation via `latent_dynamics.compute_losses(...)`
        - Higher-order consistency losses (e.g., chain-rule and consistency penalties)
        - Optional rollout and IC-rollout losses (curriculum-controlled)
        - Back-propagation + gradient clipping + optimizer step

        **Checkpointing (important)**

        When a new best epoch is found within the round, this method calls
        `Trainer._Save_Checkpoint(...)` to snapshot:

        - EncoderDecoder weights
        - the LatentDynamics state, including native training coefficient dictionaries

        This ensures `Trainer.train()` can restore the model and coefficients from the best epoch
        of the round, which is what greedy sampling should use.

        **Loss logging**

        This method records scalar loss totals and timings using the base-class helpers
        `_cache_metric(...)`.


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
        
        # Generate the weight functions and their derivatives.
        self._prepare_weak_form_data();

        # Map trainable state to self's device before constructing the optimizer.  This keeps
        # checkpoint-restored LD coefficients from staying on CPU during a GPU training round.
        device                  : str                       = self.device;
        encoder_decoder_device  : EncoderDecoder            = self.encoder_decoder.to(device);
        self.latent_dynamics.move_parameters_to_device(device);

        # Reset optimizer.
        optimizer_parameters_list    : list[torch.Tensor] = self._optimizer_parameters();
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
            
            ith_U_Train_device  : list[torch.Tensor] = [];
            for j in range(self.n_IC):
                ith_U_Train_device.append(self.U_Train[i][j].to(device));
            U_Train_device.append(ith_U_Train_device);

        # Cache CPU/NumPy time grids once per training round.  These are used only for rollout
        # window selection, so keeping them on CPU avoids repeated GPU->CPU synchronization from
        # t_i.detach().cpu().numpy() inside the epoch loop.
        t_Train_np: list[numpy.ndarray] = [
            self.t_Train[i].detach().cpu().numpy()
            for i in range(n_train)
        ];

        # IC rollout setup
        if(self.loss_weights['IC_rollout'] > 0 and p_IC_rollout > 0):
            timer : float = time.perf_counter();

            t_Grid_IC_rollout, n_IC_rollout_frames, U_IC_Rollout_Targets = self._IC_rollout_setup(  t            = t_Train_device, 
                                                                                                    p_IC_rollout = p_IC_rollout);
            self._cache_metric("time/IC_Rollout/Setup", time.perf_counter() - timer);

        # -----------------------------------------------------------------------------------------
        # Run the iterations!

        for iter in range(start_iter, end_iter):
            step_timer : float = time.perf_counter();
            LOGGER.debug("=" * 80);
            LOGGER.debug("Starting training iteration %d/%d" % (iter + 1, end_iter));


            # -------------------------------------------------------------------------------------
            # Warmup the learning rate for the first few epochs after greedy sampling.
            # NOTE: epochs_in_round will be recalculated later for rollout updates.

            epochs_in_round     : int = iter - start_iter;  # Progress within current training round
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
                timer : float = time.perf_counter();

                # Recalculate p_IC_rollout based on progress within current round
                p_IC_rollout   = min(self.max_p_IC_rollout, self.p_IC_rollout_init + self.IC_dp_per_update*(epochs_in_round//self.IC_rollout_update_freq));

                LOGGER.info("p_IC_rollout is now %f (epoch %d/%d in current round)" % (p_IC_rollout, epochs_in_round, end_iter - start_iter));

                # Setup IC rollout time grids and targets
                if(p_IC_rollout > 0):
                    t_Grid_IC_rollout, n_IC_rollout_frames, U_IC_Rollout_Targets = self._IC_rollout_setup(  t            = t_Train_device, 
                                                                                                            p_IC_rollout = p_IC_rollout);
                
                self._cache_metric("time/IC_Rollout/Setup", time.perf_counter() - timer);


            # -------------------------------------------------------------------------------------
            # Zero gradients.
            
            self.optimizer.zero_grad();
            LOGGER.debug("Zeroed gradients for iteration %d" % (iter + 1));


            # -------------------------------------------------------------------------------------
            # Main epoch loop + setup

            # Initialize losses. 
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
            Latent_States           : list[list[torch.Tensor]]  = [];       # len = n_train. i'th element is 2 element list of (n_t_i, n_z) arrays.
            forward_timer           : float                     = 0.0;
            recon_loss_timer        : float                     = 0.0;
            consistency_loss_timer  : float                     = 0.0;
            chain_rule_loss_timer   : float                     = 0.0;

            # Cycle through the combinations of parameter values.
            for i in range(n_train):
                # Setup. 
                D_i         : torch.Tensor  = U_Train_device[i][0];
                V_i         : torch.Tensor  = U_Train_device[i][1];

                D_i         = torch.squeeze(D_i);  # shape (n_t(i), physics.Frame_Shape)
                V_i         = torch.squeeze(V_i);  # shape (n_t(i), physics

                t_Grid_i    : torch.Tensor  = t_Train_device[i];
                n_t_i       : int           = t_Grid_i.shape[0];


                # -----------------------------------------------------------------------------
                # Forward pass

                timer : float = time.perf_counter();
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
                
                Latent_States.append(Z_i);

                U_Pred_i    : list[torch.Tensor]    = list(encoder_decoder_device.Decode(*Z_i));
                #D_Pred_i    : torch.Tensor          = U_Pred_i[0];  # shape = (n_t(i), physics.Frame_Shape)
                #V_Pred_i    : torch.Tensor          = U_Pred_i[1];  # shape = (n_t(i), physics.Frame_Shape)

                D_Pred_i    : torch.Tensor          = torch.squeeze(U_Pred_i[0]);  # shape = (n_t(i), physics.Frame_Shape)
                V_Pred_i    : torch.Tensor          = torch.squeeze(U_Pred_i[1]);  # shape = (n_t(i), physics.Frame_Shape)

                LOGGER.debug("Forward Pass (Autoencoder_Pair) - complete for parameter combination %d" % i);
                forward_timer += time.perf_counter() - timer;


                # ----------------------------------------------------------------------------
                # Reconstruction loss

                if(self.loss_weights['recon'] > 0):
                    timer : float = time.perf_counter();
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
                    
                    LOGGER.debug("Reconstruction Loss (Autoencoder_Pair) - complete for parameter combination %d" % i);
                    recon_loss_timer += time.perf_counter() - timer;


                # ---------------------------------------------------------------------------------
                # Weak-form consistency.
                #
                # The strong form enforces  dZ_D/dt = Z_V  via finite differences,
                # which amplifies noise.  Instead, integrate against test functions
                # and apply IBP (boundary terms vanish because φ_h is compactly
                # supported):
                #
                #   ∫ φ'_h(t) Z_D(t) dt  +  ∫ φ_h(t) Z_V(t) dt  =  0
                #
                # Matrix form:   dPhi @ Z_D  +  Phi @ Z_V  ≈  0
                # ---------------------------------------------------------------------------------

                if(self.loss_weights['consistency'] > 0):
                    timer : float = time.perf_counter();
                    LOGGER.debug("Consistency Loss (Autoencoder_Pair) - start for parameter combination %d" % i);

                    weight_function_derivatives = self.latent_dynamics.get_test_functions(self.param_space.train_space[i, :]);
                    Phi_i   : torch.Tensor = weight_function_derivatives[0].to(device = Z_D_i.device, dtype = Z_D_i.dtype);
                    dPhi_i  : torch.Tensor = weight_function_derivatives[1].to(device = Z_D_i.device, dtype = Z_D_i.dtype);

                    # Row-wise normalization (one scale per test function) so that
                    # test functions of different widths contribute equally.
                    scale   : torch.Tensor = torch.linalg.norm(dPhi_i, dim = 1, keepdim = True).clamp(min = 1e-10);

                    # Z-space:  dPhi @ Z_D + Phi @ Z_V ≈ 0
                    weak_lhs_Z  : torch.Tensor = (dPhi_i @ Z_D_i + Phi_i @ Z_V_i) / scale;     # (H, n_z)
                    consistency_Z_loss_ith_param = torch.mean(weak_lhs_Z**2) if self.loss_types['consistency'] == "MSE" else torch.mean(torch.abs(weak_lhs_Z));

                    # U-space:  dPhi @ D_pred + Phi @ V_pred ≈ 0
                    weak_lhs_U  : torch.Tensor = (dPhi_i @ D_Pred_i + Phi_i @ V_Pred_i) / scale;  # (H, n_x)
                    consistency_U_loss_ith_param = torch.mean(weak_lhs_U**2) if self.loss_types['consistency'] == "MSE" else torch.mean(torch.abs(weak_lhs_U));

                    # Accumulate and store.
                    loss_consistency_Z += consistency_Z_loss_ith_param;
                    loss_consistency_U += consistency_U_loss_ith_param;
                    
                    LOGGER.debug("Consistency Loss (Autoencoder_Pair) - complete for parameter combination %d" % i);
                    consistency_loss_timer      += time.perf_counter() - timer;


                # ---------------------------------------------------------------------------------
                # Weak-form chain rule.
                #
                # U-space chain rule enforces  V_FOM(t) = (d/dt) dec(Z_D(t)).
                # Multiply by φ_h, integrate, apply IBP:
                #
                #   ∫ φ_h(t) V_FOM(t) dt  =  -∫ φ'_h(t) dec(Z_D(t)) dt
                #
                # i.e.   Phi @ V_FOM + dPhi @ D_pred ≈ 0
                #
                # This smooths the noisy V_FOM and avoids JVP entirely.
                #
                # Z-space chain rule (Z_V = (d/dt)enc(D)) yields the same
                # weak equation as consistency Z:  dPhi @ Z_D + Phi @ Z_V ≈ 0
                # so it is *structurally identical* when the weak form is active.
                # We still compute and log it for monitoring.
                # ---------------------------------------------------------------------------------

                if(self.loss_weights['chain_rule'] > 0):
                    timer : float = time.perf_counter();
                    LOGGER.debug("Chain Rule Loss (Autoencoder_Pair) - start for parameter combination %d" % i);
                   
                    weight_function_derivatives = self.latent_dynamics.get_test_functions(self.param_space.train_space[i, :]);
                    Phi_i   : torch.Tensor = weight_function_derivatives[0].to(device = Z_D_i.device, dtype = Z_D_i.dtype);
                    dPhi_i  : torch.Tensor = weight_function_derivatives[1].to(device = Z_D_i.device, dtype = Z_D_i.dtype);
                    scale   : torch.Tensor = torch.linalg.norm(dPhi_i, dim = 1, keepdim = True).clamp(min = 1e-10);

                    # U-space:  Phi @ V_FOM + dPhi @ D_pred ≈ 0
                    weak_cr_U  : torch.Tensor = (Phi_i @ V_i + dPhi_i @ D_Pred_i) / scale;
                    chain_rule_U_loss_ith_param = torch.mean(weak_cr_U**2) if self.loss_types['chain_rule'] == "MSE" else torch.mean(torch.abs(weak_cr_U));

                    # Z-space:  dPhi @ Z_D + Phi @ Z_V ≈ 0  (same as weak consistency Z)
                    weak_cr_Z  : torch.Tensor = (dPhi_i @ Z_D_i + Phi_i @ Z_V_i) / scale;
                    chain_rule_Z_loss_ith_param = torch.mean(weak_cr_Z**2) if self.loss_types['chain_rule'] == "MSE" else torch.mean(torch.abs(weak_cr_Z));

                    # Accumulate and store.
                    loss_chain_rule_U += chain_rule_U_loss_ith_param;
                    loss_chain_rule_Z += chain_rule_Z_loss_ith_param;
                    
                    LOGGER.debug("Chain Rule Loss (Autoencoder_Pair) - complete for parameter combination %d" % i);
                    chain_rule_loss_timer += time.perf_counter() - timer;

            # Cache timing information
            self._cache_metric("time/Forward_Pass",         forward_timer);
            self._cache_metric("time/Recon_Loss",           recon_loss_timer);
            self._cache_metric("time/Consistency_Loss",     consistency_loss_timer);
            self._cache_metric("time/Chain_Rule_Loss",      chain_rule_loss_timer);

            # Cache the total recon, consistency, and chain rule losses.
            self._cache_metric('loss/recon/D/total', loss_recon_D.detach());
            self._cache_metric('loss/recon/V/total', loss_recon_V.detach());
            self._cache_metric('loss/consistency/Z/total', loss_consistency_Z.detach());
            self._cache_metric('loss/consistency/U/total', loss_consistency_U.detach());
            self._cache_metric('loss/chain_rule/U/total', loss_chain_rule_U.detach());
            self._cache_metric('loss/chain_rule/Z/total', loss_chain_rule_Z.detach());


            # --------------------------------------------------------------------------------
            # Latent Dynamics losses

            timer : float = time.perf_counter();

            # Compute the latent dynamics losses.
            LD_losses : LD_Loss_Container = self.latent_dynamics.compute_losses( 
                                                        Latent_States    = Latent_States, 
                                                        t_Grid           = t_Train_device,
                                                        step             = iter,
                                                        params           = self.param_space.train_space);

            # Cache metrics
            for key, value in LD_losses.metrics.items():
                self._cache_metric(key = key, value = value);

            # Compute weighted loss sum.
            loss_LD_weighted_sum  : torch.Tensor            = torch.zeros((), dtype = torch.float32, device = device);
            for key, value in LD_losses.losses.items():
                loss_LD_weighted_sum = loss_LD_weighted_sum + LD_losses.weights[key] * value;

            self._cache_metric("time/LD_Losses", time.perf_counter() - timer);


            # ---------------------------------------------------------------------------------
            # Rollout loss. Note that we need the coefficients before we can compute this.

            if(self.loss_weights['rollout'] > 0 and p_rollout > 0):
                timer : float = time.perf_counter();
                LOGGER.debug("Rollout Loss (Autoencoder_Pair) - start");

                # For each training parameter combination, randomly select a small number of
                # rollable start frames, rollout each one on the *true* absolute time grid,
                # and compare full trajectories (no interpolation / no random target points).
                for i in range(n_train):
                    t_i   : torch.Tensor = t_Train_device[i];
                    n_t_i : int          = t_i.shape[0];
                    if n_t_i < 2:
                        continue;

                    # Rollout duration for this parameter combination.
                    t_i_np : numpy.ndarray = t_Train_np[i];
                    t0  : float = float(t_i_np[0]);
                    tf  : float = float(t_i_np[-1]);
                    dur : float = float(p_rollout * (tf - t0));
                    if dur <= 0.0:
                        continue;

                    # Find the set of rollable frames.
                    rollable    = numpy.where(t_i_np + dur <= tf)[0];
                    if rollable.size == 0:
                        continue;
                    
                    # Pick out which frames we will roll out.
                    n_roll_i  = min(int(self.n_rollouts), int(rollable.size));
                    start_idx = numpy.random.choice(rollable, size = n_roll_i, replace = False);

                    param_i = self.param_space.train_space[i, :].reshape(1, -1);

                    Z_D_i : torch.Tensor = Latent_States[i][0];
                    Z_V_i : torch.Tensor = Latent_States[i][1];
                    D_i   : torch.Tensor = U_Train_device[i][0];
                    V_i   : torch.Tensor = U_Train_device[i][1];

                    # Set up buffers to hold rolled out Z's (along with the associated latent and
                    # FOM targets) and associated bookkeeping.  We decode once per parameter by
                    # concatenating all rollout windows, then split the decoded output back into
                    # windows before computing the per-window mean losses.
                    Z_D_pred_windows : list[torch.Tensor] = [];
                    Z_V_pred_windows : list[torch.Tensor] = [];
                    Z_D_tgt_windows  : list[torch.Tensor] = [];
                    Z_V_tgt_windows  : list[torch.Tensor] = [];
                    D_tgt_windows    : list[torch.Tensor] = [];
                    V_tgt_windows    : list[torch.Tensor] = [];
                    lengths          : list[int]          = [];
                    
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

                        # Fetch the targets
                        Z_D_tgt : torch.Tensor = Z_D_i[k_int:(j_int + 1), :];
                        Z_V_tgt : torch.Tensor = Z_V_i[k_int:(j_int + 1), :];
                        D_tgt   : torch.Tensor = D_i[k_int:(j_int + 1), ...];
                        V_tgt   : torch.Tensor = V_i[k_int:(j_int + 1), ...];
                        Z_D_tgt_windows.append(Z_D_tgt);
                        Z_V_tgt_windows.append(Z_V_tgt);
                        D_tgt_windows.append(D_tgt);
                        V_tgt_windows.append(V_tgt);

                        # Get the model's prediction (in the latent space)
                        Z_D0 : torch.Tensor = Z_D_i[k_int, :];
                        Z_V0 : torch.Tensor = Z_V_i[k_int, :];

                        Z_pred_all : list[list[torch.Tensor]] = self.latent_dynamics.simulate(
                            IC     = [[Z_D0, Z_V0]],
                            t_Grid = [t_win_np],
                            params = param_i);
                        Z_D_pred = Z_pred_all[0][0];
                        Z_V_pred = Z_pred_all[0][1];
                        assert Z_D_pred.ndim == 2 and Z_V_pred.ndim == 2;
                        assert Z_D_pred.shape[0] == Z_V_pred.shape[0] == t_win_np.shape[0];
                        Z_D_pred_windows.append(Z_D_pred);
                        Z_V_pred_windows.append(Z_V_pred);
                        lengths.append(Z_D_pred.shape[0]);

                    # Decode all rollout windows for this parameter in one batched call.
                    assert len(Z_D_pred_windows) == len(Z_V_pred_windows) == len(Z_D_tgt_windows) == len(Z_V_tgt_windows) == len(D_tgt_windows) == len(V_tgt_windows) == len(lengths) == n_roll_i;
                    Z_D_pred_cat : torch.Tensor = torch.cat(Z_D_pred_windows, dim = 0);
                    Z_V_pred_cat : torch.Tensor = torch.cat(Z_V_pred_windows, dim = 0);
                    assert Z_D_pred_cat.shape[0] == Z_V_pred_cat.shape[0] == sum(lengths);
                    D_pred_cat, V_pred_cat = encoder_decoder_device.Decode(Z_D_pred_cat, Z_V_pred_cat);
                    assert D_pred_cat.shape[0] == V_pred_cat.shape[0] == Z_D_pred_cat.shape[0];

                    # Compute losses window-by-window to preserve the old per-rollout weighting.
                    loss_ROM_i_D : torch.Tensor = torch.zeros(1, dtype = torch.float32, device = device);
                    loss_ROM_i_V : torch.Tensor = torch.zeros(1, dtype = torch.float32, device = device);
                    loss_FOM_i_D : torch.Tensor = torch.zeros(1, dtype = torch.float32, device = device);
                    loss_FOM_i_V : torch.Tensor = torch.zeros(1, dtype = torch.float32, device = device);
                    offset       : int          = 0;
                    for len_i, Z_D_tgt, Z_V_tgt, D_tgt, V_tgt, Z_D_pred, Z_V_pred in zip(lengths, Z_D_tgt_windows, Z_V_tgt_windows, D_tgt_windows, V_tgt_windows, Z_D_pred_windows, Z_V_pred_windows):
                        D_pred = D_pred_cat[offset:(offset + len_i), ...];
                        V_pred = V_pred_cat[offset:(offset + len_i), ...];
                        offset += len_i;

                        assert Z_D_tgt.shape[0] == Z_V_tgt.shape[0] == D_tgt.shape[0] == V_tgt.shape[0] == Z_D_pred.shape[0] == Z_V_pred.shape[0] == D_pred.shape[0] == V_pred.shape[0];

                        diff_Z_D = Z_D_tgt - Z_D_pred;
                        diff_Z_V = Z_V_tgt - Z_V_pred;
                        diff_D   = D_pred - D_tgt;
                        diff_V   = V_pred - V_tgt;

                        if self.loss_types['rollout'] == "MSE":
                            loss_ROM_i_D = loss_ROM_i_D + torch.mean(diff_Z_D**2);
                            loss_ROM_i_V = loss_ROM_i_V + torch.mean(diff_Z_V**2);
                            loss_FOM_i_D = loss_FOM_i_D + torch.mean(diff_D**2);
                            loss_FOM_i_V = loss_FOM_i_V + torch.mean(diff_V**2);
                        elif self.loss_types['rollout'] == "MAE":
                            loss_ROM_i_D = loss_ROM_i_D + torch.mean(torch.abs(diff_Z_D));
                            loss_ROM_i_V = loss_ROM_i_V + torch.mean(torch.abs(diff_Z_V));
                            loss_FOM_i_D = loss_FOM_i_D + torch.mean(torch.abs(diff_D));
                            loss_FOM_i_V = loss_FOM_i_V + torch.mean(torch.abs(diff_V));
                        else:
                            raise ValueError("Invalid rollout loss type: %s" % self.loss_types['rollout']);
                    assert offset == D_pred_cat.shape[0];

                    # Normalize losses based on number of rollouts.
                    rollout_ROM_D_loss_ith_param = loss_ROM_i_D / float(n_roll_i);
                    rollout_ROM_V_loss_ith_param = loss_ROM_i_V / float(n_roll_i);
                    rollout_FOM_D_loss_ith_param = loss_FOM_i_D / float(n_roll_i);
                    rollout_FOM_V_loss_ith_param = loss_FOM_i_V / float(n_roll_i);

                    # Update total.
                    loss_rollout_ROM_D += rollout_ROM_D_loss_ith_param;
                    loss_rollout_ROM_V += rollout_ROM_V_loss_ith_param;
                    loss_rollout_FOM_D += rollout_FOM_D_loss_ith_param;
                    loss_rollout_FOM_V += rollout_FOM_V_loss_ith_param;

                # Store total rollout loss.
                self._cache_metric('loss/rollout/ROM/D/total', loss_rollout_ROM_D.detach());
                self._cache_metric('loss/rollout/ROM/V/total', loss_rollout_ROM_V.detach());
                self._cache_metric('loss/rollout/FOM/D/total', loss_rollout_FOM_D.detach());
                self._cache_metric('loss/rollout/FOM/V/total', loss_rollout_FOM_V.detach());

                LOGGER.debug("Rollout Loss (Autoencoder_Pair) - complete");
                self._cache_metric("time/Rollout", time.perf_counter() - timer);


            # ---------------------------------------------------------------------------------
            # IC Rollout loss. This simulates forward from the FOM initial conditions.

            if(self.loss_weights['IC_rollout'] > 0 and p_IC_rollout > 0):
                timer : float = time.perf_counter();
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
                        D_IC_i = self.normalize(D_IC_i, 0);
                        V_IC_i = self.normalize(V_IC_i, 1);
                    
                    # Encode the FOM initial conditions
                    Z_D_IC_i, Z_V_IC_i = encoder_decoder_device.Encode(D_IC_i, V_IC_i);
                    Z_D_IC_i = Z_D_IC_i.reshape(-1);
                    Z_V_IC_i = Z_V_IC_i.reshape(-1);
                    
                    # Simulate the latent dynamics forward in time
                    Z_IC_Rollout_i    : list[list[torch.Tensor]]  = self.latent_dynamics.simulate(  IC      = [[Z_D_IC_i, Z_V_IC_i]], 
                                                                                                    t_Grid  = [t_Grid_IC_rollout[i]], 
                                                                                                    params  = param_i.reshape(1, -1));
                    
                    # Extract the predicted trajectory
                    Z_D_IC_Predict_i  : torch.Tensor              = Z_IC_Rollout_i[0][0];  # shape = (n_t_IC_rollout[i], n_z)
                    Z_V_IC_Predict_i  : torch.Tensor              = Z_IC_Rollout_i[0][1];  # shape = (n_t_IC_rollout[i], n_z)

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
                    
                # Store total IC rollout loss.
                self._cache_metric('loss/IC_rollout/Z_D/total', loss_IC_rollout_Z_D.detach());
                self._cache_metric('loss/IC_rollout/Z_V/total', loss_IC_rollout_Z_V.detach());
                self._cache_metric('loss/IC_rollout/D/total', loss_IC_rollout_D.detach());
                self._cache_metric('loss/IC_rollout/V/total', loss_IC_rollout_V.detach());

                LOGGER.debug("IC Rollout Loss (Autoencoder_Pair) - complete");
                self._cache_metric("time/IC_Rollout", time.perf_counter() - timer);


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
                    loss_LD_weighted_sum);
            self._cache_metric('loss/total', loss.detach());
            LOGGER.debug("Total loss (Autoencoder_Pair) computed");



            # Record coefficient scale and the most recent epoch index for fallback checkpointing.
            if isinstance(self.latent_dynamics, InterpolatableLatentDynamics):
                with torch.no_grad():
                    coef_tensors_report = self.latent_dynamics.parameters();
                    train_coefs_flat_report = torch.cat([c.reshape(-1) for c in coef_tensors_report]);
                    max_train_coef = float(torch.abs(train_coefs_flat_report).max().item());
            last_iter_idx = int(iter);




            # -------------------------------------------------------------------------------------
            # Backward Pass

            timer : float = time.perf_counter();
            LOGGER.debug("Backward Pass - start (iteration %d)" % (iter + 1));

            #  Run back propagation and update the encoder_decoder parameters. 
            # Note: optimizer.zero_grad() is already called at the start of the iteration (line 373)
            loss.backward();

            # Record the gradient in the LD and encoder_decoder
            grad_sq_encoder_decoder = torch.zeros((), device = device);
            for param in encoder_decoder_device.parameters():
                if param.grad is not None:
                    grad_sq_encoder_decoder = grad_sq_encoder_decoder + torch.sum(param.grad.detach()**2);
            grad_sq_latent_dynamics = torch.zeros((), device = device);
            for param in self.latent_dynamics.parameters():
                if param.grad is not None:
                    grad_sq_latent_dynamics = grad_sq_latent_dynamics + torch.sum(param.grad.detach()**2);
            self._cache_metric("grad_norm/encoder_decoder/raw", torch.sqrt(grad_sq_encoder_decoder).detach());
            self._cache_metric("grad_norm/latent_dynamics/raw", torch.sqrt(grad_sq_latent_dynamics).detach());

            # Clip gradients to prevent explosion during latent dynamics rollout.
            grad_norm = torch.nn.utils.clip_grad_norm_(
                optimizer_parameters_list,
                max_norm = self.gradient_clip,
                foreach  = True,
            )
            detached_grad_norm = grad_norm.detach();
            self._cache_metric("grad_norm/raw", detached_grad_norm);
            clip_value = detached_grad_norm.new_tensor(self.gradient_clip);
            self._cache_metric("grad_norm/actual", torch.minimum(detached_grad_norm, clip_value));
                                    
            # Log if gradient clipping activates (indicates potential instability)
            if grad_norm > self.gradient_clip:
                LOGGER.warning("Gradient norm %.2f exceeded threshold, clipped to %f (iter %d)" % (grad_norm, self.gradient_clip, iter + 1));
            
            LOGGER.debug("Backward Pass - backward() complete, calling optimizer.step()");
            self.optimizer.step();
            LOGGER.debug("Backward Pass - complete (iteration %d)" % (iter + 1));
            self._cache_metric("time/backwards", time.perf_counter() - timer);
            self._cache_metric("time/step", time.perf_counter() - step_timer);

            # Flush cached tensors after the optimizer update. This performs one batched
            # device-to-CPU scalar transfer for loss tracking, checkpoint decisions, and reporting,
            flushed_metrics = self._flush_metrics_cache(iter + 1);
            loss_value = flushed_metrics["loss/total"];

            # Check if we hit a new minimum loss. If so, make a checkpoint, record the loss and 
            # the iteration number. 
            # NOTE: Skip checkpointing during warmup period to avoid saving "lucky" early epochs
            # that benefit from distribution shift before encoder_decoder has adapted.
            if loss_value < best_loss:
                if epochs_in_round >= self.warmup_epochs:
                    LOGGER.info("Got a new lowest loss (%f) on epoch %d" % (loss_value, iter + 1));
                    self._Save_Checkpoint(encoder_decoder = encoder_decoder_device,
                                          iter            = int(iter));
                    checkpoint_saved      = True;

                    self.best_epoch       = int(iter);
                    best_loss             = loss_value;
                else:
                    LOGGER.debug("Skipping checkpoint during warmup period (epoch %d/%d in round, warmup ends at %d)" % 
                               (epochs_in_round, end_iter - start_iter, self.warmup_epochs));
            # -------------------------------------------------------------------------------------
            # Report Results from this iteration 

            # Report the current iteration number and losses
            info_str : str = "Iter: %05d/%d, Total: %3.6f" % (iter + 1, self.max_iter, loss_value);
            if(self.loss_weights['recon'] > 0):         info_str += ", Recon D: %3.6f, Recon V: %3.6f"                                              % (flushed_metrics['loss/recon/D/total'],                   flushed_metrics['loss/recon/V/total']);
            if(self.loss_weights['consistency'] > 0):   info_str += ", Consistency Z: %3.6f, Consistency U: %3.6f"                                  % (flushed_metrics['loss/consistency/Z/total'],             flushed_metrics['loss/consistency/U/total']);
            if(self.loss_weights['chain_rule'] > 0):    info_str += ", CR U: %3.6f, CR Z: %3.6f"                                                    % (flushed_metrics['loss/chain_rule/U/total'],              flushed_metrics['loss/chain_rule/Z/total']);
            if(self.loss_weights['rollout'] > 0):       info_str += ", Roll FOM D: %3.6f, Roll FOM V: %3.6f, Roll ROM D: %3.6f, Roll ROM V: %3.6f"  % (flushed_metrics.get('loss/rollout/FOM/D/total', 0.0),    flushed_metrics.get('loss/rollout/FOM/V/total', 0.0),  flushed_metrics.get('loss/rollout/ROM/D/total', 0.0),  flushed_metrics.get('loss/rollout/ROM/V/total', 0.0));
            if(self.loss_weights['IC_rollout'] > 0):    info_str += ", IC Roll D: %3.6f, IC Roll V: %3.6f, IC Roll ZD: %3.6f, IC Roll ZV: %3.6f"    % (flushed_metrics.get('loss/IC_rollout/D/total', 0.0),    flushed_metrics.get('loss/IC_rollout/V/total', 0.0),    flushed_metrics.get('loss/IC_rollout/Z_D/total', 0.0), flushed_metrics.get('loss/IC_rollout/Z_V/total', 0.0));
            for key in LD_losses.losses.keys():
                info_str += ", %s: %3.6f"   % (key, flushed_metrics.get(f"loss/{key}/total", 0.0));
            if isinstance(self.latent_dynamics, InterpolatableLatentDynamics): 
                info_str += ", max|c|: %.3f" % max_train_coef;
            LOGGER.info(info_str);
            
            LOGGER.debug("Completed training iteration %d/%d" % (iter + 1, end_iter));
        
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
