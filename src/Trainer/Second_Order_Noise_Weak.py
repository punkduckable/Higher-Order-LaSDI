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
from    copy                        import  deepcopy;

import  torch;
import  numpy;

from    EncoderDecoder              import  EncoderDecoder;
from    ParameterSpace              import  ParameterSpace;
from    Physics                     import  Physics;
from    LatentDynamics              import  LatentDynamics;
from    Optimizer                   import  Reset_Optimizer;
from    Second_Order_Noise          import  Second_Order_Noise;

# Setup Logger
LOGGER : logging.Logger = logging.getLogger(__name__);



# -------------------------------------------------------------------------------------------------
# Trainer class
# -------------------------------------------------------------------------------------------------

class Second_Order_Noise_Weak(Second_Order_Noise):
    # A dictionary that sends each parameter to a torch.Tensor of shape (N, n_t), whose i,j element 
    # holds the value of the i'th test function at the j'th time value.
    Phis : dict[tuple, torch.Tensor];   # dict[param_tuple -> torch.Tensor(H, n_t)]

    # A dictionary that sends each parameter to a torch.Tensor of shape (N, n_t), whose i,j element 
    # holds the value of the time derivative of the i'th test function at the j'th time value.    
    dPhis : dict[tuple, torch.Tensor];   # dict[param_tuple -> torch.Tensor(H, n_t)]

    # A dictionary that sends each parameter to a torch.Tensor of shape (N, n_t), whose i,j element 
    # holds the value of the second time derivative of the i'th test function at the j'th time 
    # value.      
    d2Phis : dict[tuple, torch.Tensor];   # dict[param_tuple -> torch.Tensor(H, n_t)]

    def __init__(self, 
                 physics            : Physics, 
                 encoder_decoder    : EncoderDecoder, 
                 latent_dynamics    : LatentDynamics, 
                 param_space        : ParameterSpace, 
                 config             : dict):
        """
        This defines a Trainer class designed to train second order dynamics from noisy data using
        latent dynamics based on the weak formulation. 
         
        It is a sub-class of Second_Order_Noise that is specially designed to work with weak forms.
        Thus, it has most of the same dependencies/attributes as the base "Second_Order_Rollout" 
        and "Second_Order_Noise" classes. See those classes for details. 
        

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
        
        # Make sure this config is for a "Second_Order_Noise_Weak" tainer.
        assert 'trainer' in config,                                     "config must contain a 'trainer' sub-dictionary";
        assert 'type' in config['trainer'],                             "trainer dictionary must contain a 'type' attribute";
        assert config['trainer']['type'] == "Second_Order_Noise_Weak",  "config['trainer']['type'] = %s, should be Second_Order_Noise_Weak" % config['trainer']['type'];
        assert "Second_Order_Noise_Weak" in config['trainer'],          "Second_Order_Noise_Weak must be in config['trainer']";

        LOGGER.info("Initializing a Second_Order_Noise_Weak object"); 

        # Make sure we are set up to work with the "spring_w" latent dynamics type.
        assert config['latent_dynamics'].get('type', None) == 'spring_w', "Currently, Second_Order_Noise_Weak can only work with the spring_w latent dynamics type.";
        assert 'spring_w' in config['latent_dynamics'], "config['latent_dynamics'] must contain a 'spring_w' sub-dictionary when type == 'spring_w'";
        assert hasattr(latent_dynamics, "get_test_functions"), "latent dynamics must have a `get_test_functions` method";

        # Next, we need to reconfigure the config to read like it is for a "Second_Order_Noise" 
        # object. This will allow us to hijack the "Second_Order_Noise" initializer to do most 
        # of the actual setup.
        noise_config : dict                             = deepcopy(config);
        noise_config['trainer']['type']                 = "Second_Order_Noise";
        del noise_config['trainer']['Second_Order_Noise_Weak'];
        noise_config['trainer']['Second_Order_Noise']   = config['trainer']['Second_Order_Noise_Weak'];        

        # Initialize the weight functions.
        self.Phis   = {};
        self.dPhis  = {};
        self.d2Phis = {};

        # Call the Second_Order_Noise initializer.
        super().__init__(   physics         = physics,
                            encoder_decoder = encoder_decoder,
                            latent_dynamics = latent_dynamics,
                            param_space     = param_space,
                            config          = noise_config);

        # All done!
        return;



    # ---------------------------------------------------------------------------------------------
    # Test function methods
    # ---------------------------------------------------------------------------------------------

    def _prepare_weak_form_data(self) -> None:
        """
        Build and cache weak-form test functions keyed by parameter tuple.

        Design choice: we key by parameter values (not by list index) so that downstream
        components (including greedy sampling) can look up weight functions without relying on
        list ordering.
        """

        assert len(self.t_Test) == self.param_space.n_test(), "t_Test is not initialized or has wrong length";

        self.Phis   = {};
        self.dPhis  = {};
        self.d2Phis = {};

        # Build weights for the *entire* test space once. Training parameters are a subset of the
        # test space, so this covers all calibrations and avoids needing sampler-specific logic.
        for i in range(self.param_space.n_test()):
            # Use plain Python floats for stable hashing / consistent lookup in LD.
            key = tuple(float(x) for x in self.param_space.test_space[i, :]);
            t_i : torch.Tensor = self.t_Test[i].to(self.device);
            T_i : float        = float(t_i[-1].detach().cpu().item());
            Phi_i, dPhi_i, d2Phi_i = self.latent_dynamics.get_test_functions(
                T               = T_i,
                n_t             = int(t_i.shape[0]),
                timesteps       = t_i);
            self.Phis[key]   = Phi_i;
            self.dPhis[key]  = dPhi_i;
            self.d2Phis[key] = d2Phi_i;

        LOGGER.info("Prepared weak-form test functions for %d test trajectories" % len(self.Phis));
        return;




    # ---------------------------------------------------------------------------------------------
    # Iterate
    # ---------------------------------------------------------------------------------------------

    def Iterate(self, 
                start_iter      : int, 
                end_iter        : int) -> None:
        """
        Run one training round for a second-order system (`n_IC = 2`).

        This trainer is designed for higher-order physics where the state is represented via
        multiple time derivatives (e.g., displacement and velocity). Concretely, each training
        trajectory provides two streams `U_D(t)` and `U_V(t)` and the EncoderDecoder is expected
        to encode/decode these jointly (see `Autoencoder_Pair`).

        Each epoch in `[start_iter, end_iter)` typically performs:

        - Forward passes to obtain latent trajectories and reconstructions
        - Latent dynamics calibration/loss evaluation via `latent_dynamics.calibrate(...)`
        - Higher-order consistency losses (e.g., chain-rule and consistency penalties)
        - Optional rollout and IC-rollout losses (curriculum-controlled)
        - Backpropagation + gradient clipping + optimizer step

        **Checkpointing (important)**

        When a new best epoch is found within the round, this method calls
        `Trainer._Save_Checkpoint(...)` to snapshot:

        - EncoderDecoder weights
        - the per-training-point coefficient matrix (`train_coefs`)
        - the full test-space coefficient matrix (`self.test_coefs`)

        This ensures `Trainer.train()` can restore the model and coefficients from the best epoch
        of the round, which is what greedy sampling should use.

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
        
        # Generate the weight functions and their derivatives.
        self._prepare_weak_form_data();
        assert hasattr(self.latent_dynamics, "set_weight_functions"), "To use weak forms, the latent dynamics class must have a 'set_weight_functions' method."
        self.latent_dynamics.set_weight_functions(  Phis_by_param   = self.Phis,
                                                    dPhis_by_param  = self.dPhis,
                                                    d2Phis_by_param = self.d2Phis);

        # Reset optimizer.
        Reset_Optimizer(self.optimizer);

        # Add noise
        if(self.noise_ratio > 0):
            self.apply_noise_to_U_Train();

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
        # Noise-related warnings.

        if self.noise_ratio > 0.0:
            LOGGER.info("noise_ratio = %f with weak form active. Consistency and chain-rule losses "
                        "will use noise-tolerant weak-form variants (IBP, no finite differences)." % self.noise_ratio);


        # -----------------------------------------------------------------------------------------
        # Run the iterations!

        for iter in range(start_iter, end_iter):
            self.timer.start("train_step");
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
            # Setup losses

            # Initialize losses. 
            loss_LD                 : torch.Tensor = torch.zeros(1, dtype = torch.float32, device = device);
            loss_stab               : torch.Tensor = torch.zeros(1, dtype = torch.float32, device = device);
            loss_coef              : torch.Tensor = torch.zeros(1, dtype = torch.float32, device = device);
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
                if iter % 100 == 0 or iter == start_iter:  # Log every 100 iters and first iter
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
                #D_Pred_i    : torch.Tensor          = U_Pred_i[0];  # shape = (n_t(i), physics.Frame_Shape)
                #V_Pred_i    : torch.Tensor          = U_Pred_i[1];  # shape = (n_t(i), physics.Frame_Shape)

                D_Pred_i    : torch.Tensor          = torch.squeeze(U_Pred_i[0]);  # shape = (n_t(i), physics.Frame_Shape)
                V_Pred_i    : torch.Tensor          = torch.squeeze(U_Pred_i[1]);  # shape = (n_t(i), physics.Frame_Shape)


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
                    self.timer.start("Consistency Loss");
                    LOGGER.debug("Consistency Loss (Autoencoder_Pair) - start for parameter combination %d" % i);

                    key = tuple(self.param_space.train_space[i, :]);
                    Phi_i   : torch.Tensor = self.Phis[key].to(device = Z_D_i.device, dtype = Z_D_i.dtype);
                    dPhi_i  : torch.Tensor = self.dPhis[key].to(device = Z_D_i.device, dtype = Z_D_i.dtype);

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
                    
                    param_tuple = tuple(self.param_space.train_space[i, :]);
                    self._store_loss_by_param('consistency_Z', param_tuple, iter + 1, consistency_Z_loss_ith_param.item());
                    self._store_loss_by_param('consistency_U', param_tuple, iter + 1, consistency_U_loss_ith_param.item());

                    LOGGER.debug("Consistency Loss (Autoencoder_Pair) - complete for parameter combination %d" % i);
                    self.timer.end("Consistency Loss");


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
                    self.timer.start("Chain Rule Loss");
                    LOGGER.debug("Chain Rule Loss (Autoencoder_Pair) - start for parameter combination %d" % i);
                   
                    key = tuple(self.param_space.train_space[i, :]);
                    Phi_i   : torch.Tensor = self.Phis[key].to(device = Z_D_i.device, dtype = Z_D_i.dtype);
                    dPhi_i  : torch.Tensor = self.dPhis[key].to(device = Z_D_i.device, dtype = Z_D_i.dtype);
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
                    
                    param_tuple = tuple(self.param_space.train_space[i, :]);
                    self._store_loss_by_param('chain_rule_U', param_tuple, iter + 1, chain_rule_U_loss_ith_param.item());
                    self._store_loss_by_param('chain_rule_Z', param_tuple, iter + 1, chain_rule_Z_loss_ith_param.item());

                    LOGGER.debug("Chain Rule Loss (Autoencoder_Pair) - complete for parameter combination %d" % i);
                    self.timer.end("Chain Rule Loss");

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
                self._store_loss_by_param('LD', param_tuple, iter + 1, loss_LD_list[i].item());
                self._store_loss_by_param('stab', param_tuple, iter + 1, loss_stab_list[i].item());
                self._store_loss_by_param('coef', param_tuple, iter + 1, loss_coef_list[i].item());


            # Compute the total loss.
            loss_LD     = torch.sum(torch.stack(loss_LD_list));
            loss_stab   = torch.sum(torch.stack(loss_stab_list));
            loss_coef   = torch.sum(torch.stack(loss_coef_list));

            # Append the total loss to loss_by_param.
            self._store_total_loss('LD', iter + 1, loss_LD.item());
            self._store_total_loss('stab', iter + 1, loss_stab.item());
            self._store_total_loss('coef', iter + 1, loss_coef.item());

            LOGGER.debug("Calibration (Autoencoder_Pair) - complete");
            self.timer.end("Calibration");


            # ---------------------------------------------------------------------------------
            # Rollout loss. Note that we need the coefficients before we can compute this.

            if(self.loss_weights['rollout'] > 0 and p_rollout > 0):
                self.timer.start("Rollout Loss");
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
                    t0  : float = float(t_i[0].item());
                    tf  : float = float(t_i[-1].item());
                    dur : float = float(p_rollout * (tf - t0));
                    if dur <= 0.0:
                        continue;

                    # Find the set of rollable frames.
                    t_i_np      = t_i.detach().cpu().numpy();
                    rollable    = numpy.where(t_i_np + dur <= tf)[0];
                    if rollable.size == 0:
                        continue;
                    
                    # Pick out which frames we will roll out.
                    n_roll_i  = min(int(self.n_rollouts), int(rollable.size));
                    start_idx = numpy.random.choice(rollable, size = n_roll_i, replace = False);

                    # Set up
                    loss_ROM_i_D : torch.Tensor = torch.zeros(1, dtype = torch.float32, device = device);
                    loss_ROM_i_V : torch.Tensor = torch.zeros(1, dtype = torch.float32, device = device);
                    loss_FOM_i_D : torch.Tensor = torch.zeros(1, dtype = torch.float32, device = device);
                    loss_FOM_i_V : torch.Tensor = torch.zeros(1, dtype = torch.float32, device = device);

                    param_i = self.param_space.train_space[i, :].reshape(1, -1);
                    coef_i  = train_coefs[i, :].reshape(1, -1);

                    Z_D_i : torch.Tensor = Latent_States[i][0];
                    Z_V_i : torch.Tensor = Latent_States[i][1];
                    D_i   : torch.Tensor = U_Train_device[i][0];
                    V_i   : torch.Tensor = U_Train_device[i][1];
                    
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

                        # Fetch the targets
                        Z_D_tgt : torch.Tensor = Z_D_i[k_int:(j_int + 1), :];
                        Z_V_tgt : torch.Tensor = Z_V_i[k_int:(j_int + 1), :];
                        D_tgt   : torch.Tensor = D_i[k_int:(j_int + 1), ...];
                        V_tgt   : torch.Tensor = V_i[k_int:(j_int + 1), ...];

                        # Get the model's prediction (in the latent space)
                        Z_D0 : torch.Tensor = Z_D_i[k_int:(k_int + 1), :];
                        Z_V0 : torch.Tensor = Z_V_i[k_int:(k_int + 1), :];

                        Z_pred_all : list[list[torch.Tensor]] = self.latent_dynamics.simulate(
                            coefs  = coef_i,
                            IC     = [[Z_D0, Z_V0]],
                            t_Grid = [t_win],
                            params = param_i);
                        Z_D_pred = Z_pred_all[0][0].squeeze(1);
                        Z_V_pred = Z_pred_all[0][1].squeeze(1);

                        # Decode the predictions.
                        D_pred, V_pred = encoder_decoder_device.Decode(Z_D_pred, Z_V_pred);

                        # Compute differences between true and predicted value.
                        diff_Z_D = Z_D_tgt - Z_D_pred;
                        diff_Z_V = Z_V_tgt - Z_V_pred;
                        diff_D   = D_pred - D_tgt;
                        diff_V   = V_pred - V_tgt;

                        # Update loss.
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

                    # Store results for this combination of parameters
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
                    self.loss_weights['stab']           * loss_stab + 
                    self.loss_weights['coef']           * loss_coef);
            self._store_total_loss('total', iter + 1, loss.item());
            LOGGER.debug("Total loss (Autoencoder_Pair) computed: %f" % loss.item());



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
                    self._Save_Checkpoint(encoder_decoder = encoder_decoder_device,
                                          train_coefs     = train_coefs_detached,
                                          test_coefs      = self.test_coefs,
                                          iter            = int(iter));
                    checkpoint_saved      = True;

                    self.best_train_coefs = train_coefs_detached.copy();
                    self.best_epoch       = int(iter);
                    best_loss             = loss.item();
                else:
                    LOGGER.debug("Skipping checkpoint during warmup period (epoch %d/%d in round, warmup ends at %d)" % 
                               (epochs_in_round, end_iter - start_iter, self.warmup_epochs));

            self.timer.end("Backwards Pass");
            


            # -------------------------------------------------------------------------------------
            # Report Results from this iteration 

            self.timer.start("Report");

            # Report the current iteration number and losses
            info_str : str = "Iter: %05d/%d, Total: %3.6f" % (iter + 1, self.max_iter, loss.item());
            if(self.loss_weights['recon'] > 0):         info_str += ", Recon D: %3.6f, Recon V: %3.6f"                                              % (loss_recon_D.item(),       loss_recon_V.item());
            if(self.loss_weights['consistency'] > 0):   info_str += ", Consistency Z: %3.6f, Consistency U: %3.6f"                                  % (loss_consistency_Z.item(), loss_consistency_U.item());
            if(self.loss_weights['chain_rule'] > 0):    info_str += ", CR U: %3.6f, CR Z: %3.6f"                                                    % (loss_chain_rule_U.item(),  loss_chain_rule_Z.item());
            if(self.loss_weights['rollout'] > 0):       info_str += ", Roll FOM D: %3.6f, Roll FOM V: %3.6f, Roll ROM D: %3.6f, Roll ROM V: %3.6f"  % (loss_rollout_FOM_D.item(), loss_rollout_FOM_V.item(),  loss_rollout_ROM_D.item(),  loss_rollout_ROM_V.item());
            if(self.loss_weights['IC_rollout'] > 0):    info_str += ", IC Roll D: %3.6f, IC Roll V: %3.6f, IC Roll ZD: %3.6f, IC Roll ZV: %3.6f"    % (loss_IC_rollout_D.item(),  loss_IC_rollout_V.item(),   loss_IC_rollout_Z_D.item(), loss_IC_rollout_Z_V.item());
            if(self.loss_weights['LD'] > 0):            info_str += ", LD: %3.6f"                                                                   % loss_LD.item();
            if(self.loss_weights['stab'] > 0):          info_str += ", Stab: %3.6f"                                                                 % loss_stab.item();
            if(self.loss_weights['coef'] > 0):          info_str += ", Coef: %3.6f"                                                                 % loss_coef.item();
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
