# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

import  logging;
from    typing                                  import  Callable;

import  numpy;
import  torch;

from    HLaSDI.LatentDynamics.LatentDynamics    import  LatentDynamics, LD_Loss_Container;
from    HLaSDI.Schemas                          import  CABSOLELatentDynamicsConfig, WeakCABSOLELatentDynamicsConfig, CABLELatentDynamicsSettings;
from    HLaSDI.EncoderDecoder                   import  MultiLayerPerceptron;
from    HLaSDI.Utilities.FiniteDifference       import  Derivative1_Order4, Derivative1_Order2_NonUniform;
from    HLaSDI.Utilities.SecondOrderSolvers     import  RK4;
from    HLaSDI.Utilities.Statistics             import  tensor_statistics;


# Setup Logger.
LOGGER : logging.Logger = logging.getLogger(__name__);



# -------------------------------------------------------------------------------------------------
# CABSOLE class
# -------------------------------------------------------------------------------------------------

class CABSOLE(LatentDynamics):
    def __init__(   self, 
                    n_z             :   int, 
                    Uniform_t_Grid  :   bool,
                    n_p             :   int, 
                    config          :   CABSOLELatentDynamicsConfig | WeakCABSOLELatentDynamicsConfig) -> None:
        r"""
        Initializes a Second Order Convex Affine Blend of Second Order Latent Experts 
        latent-dynamics object.

        This class models second-order latent dynamics in native form as

            z''(t) = \sum_{m = 1}^{N} w_m(t, \theta) [ K_m z(t) + C_m z'(t) + b_m ].

        Here, z is the latent state. For each m \in {1, 2, ... , N}, K_m \in \mathbb{R}^{n x n} 
        and C_m \in \mathbb{R}^{n x n} represent the m'th "stiffness" and "damping" coefficient 
        matrices, while b_m is an offset/constant forcing vector. 


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        n_z : int
            Number of latent dimensions.

        Uniform_t_Grid : bool
            Selects uniform-grid or nonuniform-grid finite differences when estimating
            accelerations from latent trajectories.

        n_p : int 
            The number of (scalar) parameters in the parameter space.

        config : CABSOLELatentDynamicsConfig
            The latent-dynamics configuration schema. The `cabsole` settings specify the number of
            experts, the target number of active experts, epsilon, and the gate-network 
            architecture.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Nothing!
        """

        assert isinstance(config, (CABSOLELatentDynamicsConfig, WeakCABSOLELatentDynamicsConfig)), "config must be a CABSOLELatentDynamicsConfig, got %s" % str(type(config));

        # Run the base class initializer. 
        LatentDynamics.__init__(
            self,
            n_z            = n_z, 
            n_IC           = 2, 
            n_p            = n_p,
            Uniform_t_Grid = Uniform_t_Grid,
            trainable      = config.trainable,
            stochastic     = False,
            config         = config);

        # Extract sub-class specific attributes.
        sub : CABLELatentDynamicsSettings       = config.cabsole;
        self.n_experts          : int           = sub.n_experts;
        self.n_active           : int           = sub.n_active;
        self.hidden_widths      : list[int]     = sub.hidden_widths;
        self.activations        : list[str]     = [sub.activations]*len(self.hidden_widths) if isinstance(sub.activations, str) else sub.activations;
        self.coef_norm          : str           = sub.coef_norm
        self.use_biases         : bool          = sub.use_biases;
        self.use_z_in_gate      : bool          = sub.use_z_in_gate;
        self.eps_engaged        : float         = sub.eps_engaged;
        self.use_mask           : bool          = sub.use_mask;
        self.mask_threshold     : float | None  = sub.mask_threshold;
        self.first_mask_step    : int   | None  = sub.first_mask_step;
        self.mask_update_freq   : int   | None  = sub.mask_update_freq;

        # Initialize the gate network.
        input_dim   : int       = (n_p + 1 + 2*n_z) if self.use_z_in_gate else (n_p + 1);
        widths      : list[int] = [input_dim] + self.hidden_widths + [self.n_experts];
        self.w                  = MultiLayerPerceptron(widths = widths, activations = self.activations);
        with torch.no_grad():
            for param in self.w.parameters():
                param.mul_(0.01);

        # Randomly initialize the experts. These are leaf tensors because the Trainer passes them
        # directly to the optimizer through parameters().
        self.unmasked_K : torch.Tensor = (0.01*torch.rand((self.n_experts, self.n_z, self.n_z), dtype = torch.float32)).requires_grad_(self.trainable);
        self.unmasked_C : torch.Tensor = (0.01*torch.rand((self.n_experts, self.n_z, self.n_z), dtype = torch.float32)).requires_grad_(self.trainable);
        self.unmasked_b : torch.Tensor | None;
        if self.use_biases:
            self.unmasked_b = torch.zeros((self.n_experts, 1, self.n_z), dtype = torch.float32).requires_grad_(self.trainable);
        else:
            self.unmasked_b = None;

        # Hard coefficient masks. A value of one means active and zero means permanently removed
        # from the effective latent dynamics.
        self.K_mask : torch.Tensor = torch.ones_like(self.unmasked_K);
        self.C_mask : torch.Tensor = torch.ones_like(self.unmasked_C);
        self.b_mask : torch.Tensor | None;
        if self.use_biases:
            assert self.unmasked_b is not None;
            self.b_mask = torch.ones_like(self.unmasked_b);
        else:
            self.b_mask = None;
        for param in self.w.parameters():
            param.requires_grad_(self.trainable);

        # Setup the loss functions used by compute_losses.
        self.MSE = torch.nn.MSELoss(reduction = 'mean');
        self.MAE = torch.nn.L1Loss(reduction = 'mean');

        self.last_tail_mass_loss : torch.Tensor | None = None;
        self.last_tail_mass_loss_list : list[torch.Tensor] | None = None;
        return;



    # ---------------------------------------------------------------------------------------------
    # parameters, move_parameters_to_device, and initialize_coefficients
    # ---------------------------------------------------------------------------------------------


    def parameters(self) -> list[torch.Tensor]:
        r"""
        Return CABSOLE-owned tensors that should be passed to torch optimizers.

        These are the expert matrices, optional expert biases, and gate-network parameters. The list 
        is empty when the latent dynamics are frozen.
        """

        if self.trainable == False:
            return [];

        # Append un-masked K, C, b
        tensors : list[torch.Tensor] = [self.unmasked_K, self.unmasked_C];
        if self.unmasked_b is not None:
            tensors.append(self.unmasked_b);

        # Append gate network parameters.
        for param in self.w.parameters():
            tensors.append(param);
        return tensors;


    def move_parameters_to_device(self, device : torch.device | str) -> None:
        r"""
        Move CABSOLE-owned parameters to a device.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        device : torch.device or str
            The destination device for the experts and gate network.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Nothing!
        """

        # Keep K, C, and b as leaf tensors because the Trainer optimizes them directly.
        self.unmasked_K = self.unmasked_K.detach().to(device = device).requires_grad_(self.trainable);
        self.unmasked_C = self.unmasked_C.detach().to(device = device).requires_grad_(self.trainable);
        self.K_mask     = self.K_mask.to(device = device, dtype = self.unmasked_K.dtype);
        self.C_mask     = self.C_mask.to(device = device, dtype = self.unmasked_C.dtype);
        if self.unmasked_b is not None:
            self.unmasked_b = self.unmasked_b.detach().to(device = device).requires_grad_(self.trainable);
        if self.b_mask is not None:
            assert self.unmasked_b is not None;
            self.b_mask = self.b_mask.to(device = device, dtype = self.unmasked_b.dtype);

        # Now move the gate matrix.
        self.w = self.w.to(device = device);
        for param in self.w.parameters():
            param.requires_grad_(self.trainable);

        # All done :)
        return;


    def initialize_coefficients(
            self,
            Latent_States   : list[list[torch.Tensor]],
            t_Grid          : list[torch.Tensor],
            device          : torch.device,
            params          : numpy.ndarray) -> None:
        r"""
        Move the globally initialized CABSOLE parameters to the requested device.

        CABSOLE does not fit one coefficient dictionary per training parameter. Its experts and gate
        are initialized when the object is constructed and then trained directly. This method keeps
        the standard latent-dynamics initialization hook but only validates the incoming training
        data and moves CABSOLE-owned tensors to `device`.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        Latent_States : list[list[torch.Tensor]], len = n_param
            The i'th list element is a two-element list. The first tensor holds the latent
            displacement trajectory Z_D with shape (n_t(i), n_z), and the second tensor holds the
            latent velocity trajectory Z_V with shape (n_t(i), n_z).

        t_Grid : list[torch.Tensor], len = n_param
            Time grid for each latent trajectory.

        device : torch.device
            The device where CABSOLE's experts and gate network should live.
            
        params : numpy.ndarray, shape = (n_param, n_p)
            The parameters currently represented in the training set.
            
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        None.
        """

        # Checks.
        assert params is not None, "CABSOLE.initialize_coefficients requires params!";
        assert isinstance(params, numpy.ndarray) and len(params.shape) == 2;
        assert params.shape[1] == self.n_p;
        assert isinstance(t_Grid, list);
        assert isinstance(Latent_States, list);
        assert len(Latent_States) == len(t_Grid) == params.shape[0];

        # Move K, C, optional b, masks, and w to specified device.
        self.move_parameters_to_device(device);
        return None;


    # ---------------------------------------------------------------------------------------------
    # Compute Losses, RHS, and Simulate
    # ---------------------------------------------------------------------------------------------


    def compute_losses(
        self, 
        Latent_States : list[list[torch.Tensor]],
        t_Grid        : list[torch.Tensor],
        step          : int,
        params        : numpy.ndarray | None = None,
    ) -> LD_Loss_Container:
        r"""
        Compute latent-dynamics, coefficient, and gate-diversity losses for training parameters.

        For each parameter row, this method evaluates the global CABSOLE mixture-of-experts model

            z''(t) = \sum_{m = 1}^{N} w_m(t, \theta) [ K_m z(t) + C_m z'(t) + b_m ].

        The coefficient loss penalizes the global expert matrices and optional biases. The
        diversity loss is a squared coefficient-of-variation penalty on total dense expert load,
        and the tail loss penalizes softmax mass outside the top `n_active` experts.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        Latent_States : list[list[torch.Tensor]], len = n_param
            The i'th list element is a two-element list whose entries are latent displacement and
            velocity tensors with shape (n_t(i), n_z).

        t_Grid : list[torch.Tensor], len = n_param
            The i'th element is a 1D tensor of shape (n_t(i)) holding the time grid for the i'th
            parameter combination.

        step : int
            The optimizer step number. This is used for periodic coefficient-mask updates when
            masking is enabled.

        params : numpy.ndarray, shape = (n_param, n_p)
            Parameter rows used as inputs to the gate network.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        losses : LD_Loss_Container
            A LD_Loss_Container object housing the losses and their weights. It houses the 
            following losses:

            - `LD`: finite-difference residual losses.
            - `coef`: scalar global expert-size penalty.
            - `diversity`: scalar global squared-CV expert-load penalty.
            - `tail`: soft top-`n_active` tail-mass penalties.
        """
    
        # Checks.
        assert params is not None, "CABSOLE.compute_losses requires params for the gate network";
        assert isinstance(params, numpy.ndarray) and len(params.shape) == 2;
        assert params.shape[1] == self.n_p;
        assert isinstance(t_Grid, list);
        assert isinstance(Latent_States, list);
        assert len(Latent_States) == len(t_Grid) == params.shape[0];
        assert len(t_Grid) > 0;

        # Map params to a tensor
        w_param         : torch.Tensor  = next(self.w.parameters());
        gate_device                     = w_param.device
        gate_dtype                      = w_param.dtype;
        params_tensor   : torch.Tensor  = torch.tensor(params, dtype = gate_dtype, device = gate_device);

        # Setup
        loss_LD_list        : list[torch.Tensor]        = [];
        summed_weights      : torch.Tensor              = torch.zeros((self.n_experts), dtype = self.unmasked_K.dtype, device = self.unmasked_K.device);
        times_engaged       : torch.Tensor              = torch.zeros((self.n_experts), dtype = torch.int64, device = self.unmasked_K.device);
        n_engaged_list      : list[torch.Tensor]        = [];
        loss_tail_list      : list[torch.Tensor]        = [];
        weights_list        : list[torch.Tensor]        = [];
        tail_mass_list      : list[torch.Tensor]        = [];
        metrics             : dict[str, torch.Tensor]   = {};

        # Periodically update the hard coefficient masks. Masked entries are multiplied out in all
        # RHS, simulation, and coefficient-loss evaluations.
        if self.use_mask:
            assert self.first_mask_step is not None;
            assert self.mask_update_freq is not None;
            if step >= self.first_mask_step and (step - self.first_mask_step) % self.mask_update_freq == 0:
                self._update_mask();

            # Record metrics
            metrics["n_active/K"] = self.K_mask.sum().to(device = self.unmasked_K.device, dtype = self.unmasked_K.dtype).detach();
            metrics["n_active/C"] = self.C_mask.sum().to(device = self.unmasked_K.device, dtype = self.unmasked_K.dtype).detach();
            if self.use_biases:
                metrics["n_active/b"] = self.b_mask.sum().to(device = self.unmasked_K.device, dtype = self.unmasked_K.dtype).detach();

        n_param     : int           = len(t_Grid);
        for i in range(n_param):
            # Fetch this parameter's latent trajectory and time grid.
            ith_params      : torch.Tensor  = params_tensor[i, :]
            ith_t_Grid      : torch.Tensor  = t_Grid[i];
            ith_Z           : torch.Tensor  = Latent_States[i][0]; # [n_t_i, n_z]
            ith_dZ_dt       : torch.Tensor  = Latent_States[i][1]; # [n_t_i, n_z]
            n_t_i           : int           = len(ith_t_Grid);
            
            assert isinstance(ith_Z,        torch.Tensor);
            assert isinstance(ith_dZ_dt,    torch.Tensor);  
            assert len(ith_Z.shape)         == 2 and ith_Z.shape[1]         == self.n_z;
            assert len(ith_dZ_dt.shape)     == 2 and ith_dZ_dt.shape[1]     == self.n_z;
            assert len(ith_t_Grid.shape)    == 1 and ith_t_Grid.shape[0]    == n_t_i;
            assert ith_Z.shape[0]           == n_t_i;
            assert ith_dZ_dt.shape[0]       == n_t_i;

            # Compute d2Z/dt2. Uniform grids use the higher-order stencil; nonuniform grids use the
            # nonuniform finite-difference helper.
            if(self.Uniform_t_Grid  == True):
                h : float = (ith_t_Grid[1] - ith_t_Grid[0]).item();
                ith_d2Z_dt2 : torch.Tensor = Derivative1_Order4(U = ith_dZ_dt, h = h);
            else:
                ith_d2Z_dt2 = Derivative1_Order2_NonUniform(U = ith_dZ_dt, t_Grid = ith_t_Grid);

            # Evaluate expert weights.
            ith_RHS, ith_weights = self._evaluate_rhs(Z = ith_Z, dZ_dt = ith_dZ_dt, t_Grid = ith_t_Grid, params = ith_params, t0 = ith_t_Grid[0], t_span = ith_t_Grid[-1] - ith_t_Grid[0]);
            weights_list.append(ith_weights.to(device = self.unmasked_K.device, dtype = self.unmasked_K.dtype));

            # Record which experts are engaged during each step for this parameter.
            ith_engaged : torch.Tensor = (ith_weights > self.eps_engaged).to(dtype = torch.bool, device = self.unmasked_K.device)
            times_engaged += torch.sum(ith_engaged, dim = 0);
            n_engaged_list.append(torch.sum(ith_engaged, dim = 1).to(device = self.unmasked_K.device, dtype = self.unmasked_K.dtype));

            # Compute the LD loss for the i'th combination of parameters.
            ith_loss_LD = self.MSE(ith_d2Z_dt2, ith_RHS);
            loss_LD_list.append(ith_loss_LD);

            # Accumulate expert loads across all parameter values and times. This is a 
            # deterministic analogue of MoE importance/load diversity: it encourages all
            # experts to be useful somewhere without forcing all experts to be active at 
            # every step.
            summed_weights = summed_weights + torch.sum(ith_weights.to(device = self.unmasked_K.device, dtype = self.unmasked_K.dtype), dim = 0);

            # Tail-mass penalty: compute how much softmax mass lies outside the top-n_active 
            # logits. If this is small, most of the probability mass is in the top n_active 
            # experts, as intended. 
            if self.n_active >= self.n_experts:
                ith_tail_mass : torch.Tensor = torch.zeros((n_t_i), dtype = ith_weights.dtype, device = ith_weights.device);
            else:
                ith_topk_idx             : torch.Tensor = torch.topk(ith_weights, self.n_active, dim = 1, sorted = False).indices;
                ith_topk_dense_mass      : torch.Tensor = torch.sum(ith_weights.gather(1, ith_topk_idx), dim = 1);
                ith_tail_mass            : torch.Tensor = 1.0 - ith_topk_dense_mass;
            tail_mass_list.append(ith_tail_mass.to(device = self.unmasked_K.device, dtype = self.unmasked_K.dtype));
            ith_tail_loss : torch.Tensor = torch.mean(torch.pow(ith_tail_mass.to(device = self.unmasked_K.device, dtype = self.unmasked_K.dtype), 2))
            loss_tail_list.append(ith_tail_loss);


        # Evaluate loss statistics (computed across times and parameters).
        weights         : torch.Tensor = torch.cat(weights_list, dim = 0);
        tail_masses     : torch.Tensor = torch.cat(tail_mass_list, dim = 0);
        n_engaged       : torch.Tensor = torch.cat(n_engaged_list, dim = 0);
        metrics.update(tensor_statistics(prefix = "expert/weights",         values = weights));
        metrics.update(tensor_statistics(prefix = "mass/tail",              values = tail_masses));
        metrics.update(tensor_statistics(prefix = "expert/num_engaged",    values = n_engaged));
        metrics.update(tensor_statistics(prefix = "expert/times_engaged",  values = times_engaged));
        metrics["expert/num_ever_engaged"] = torch.sum(times_engaged > 0).to(device = self.unmasked_K.device, dtype = self.unmasked_K.dtype).detach();

        # Coefficient loss is the sum of the selected norms of the matrix portions of each expert,
        # plus the selected norm of each enabled bias. This is a scalar global loss, so the
        # Trainer will not multiply it by n_param.
        K_coef  : torch.Tensor          = self.K;
        C_coef  : torch.Tensor          = self.C;
        b_coef  : torch.Tensor | None   = self.b;
        ord     : int                   = 1 if self.coef_norm == 'l1' else 2;
        K_norms         : torch.Tensor          = torch.linalg.vector_norm(K_coef.reshape(self.n_experts, -1), ord = ord, dim = 1).sum();
        C_norms         : torch.Tensor          = torch.linalg.vector_norm(C_coef.reshape(self.n_experts, -1), ord = ord, dim = 1).sum();
        if b_coef is None:
            b_norms     : torch.Tensor          = torch.zeros((), dtype = K_coef.dtype, device = K_coef.device);
        else:
            b_norms     : torch.Tensor          = torch.linalg.vector_norm(b_coef.reshape(self.n_experts, -1), ord = ord, dim = 1).sum();
        loss_coef       : torch.Tensor          = K_norms + C_norms + b_norms;

        # diversity loss is the squared coefficient of variation of expert loads. Use 
        # the population standard deviation so n_experts = 1 produces zero instead of NaN.
        metrics.update(tensor_statistics(prefix = "expert/summed_weight", values = summed_weights));
        eps             : float                 = torch.finfo(summed_weights.dtype).eps;
        mean_load       : torch.Tensor          = torch.mean(summed_weights);
        std_load        : torch.Tensor          = torch.std(summed_weights, unbiased = False);
        loss_diversity  : torch.Tensor          = torch.pow(std_load/(mean_load + eps), 2);

        # Store the average tail-mass loss for diagnostics/plotting; the weighted objective and
        # logged loss metric use the summed scalar returned under the `tail` key.
        loss_tail : torch.Tensor = torch.mean(torch.stack(loss_tail_list));
        self.last_tail_mass_loss = loss_tail.detach();
        self.last_tail_mass_loss_list = [loss.detach() for loss in loss_tail_list];

        # All done :)
        loss_LD     : torch.Tensor      = torch.sum(torch.stack(loss_LD_list));
        loss_tail   : torch.Tensor      = torch.sum(torch.stack(loss_tail_list));
        metrics["loss/LD/total"]        = loss_LD.detach();
        metrics["loss/coef/total"]      = loss_coef.detach();
        metrics["loss/coef/K"]          = K_norms.detach();
        metrics["loss/coef/C"]          = C_norms.detach();
        metrics["loss/coef/b"]          = b_norms.detach();
        metrics["loss/diversity/total"] = loss_diversity.detach();
        metrics["loss/tail/total"]      = loss_tail.detach();

        losses_dict = {'LD' : loss_LD, 'coef' : loss_coef, 'diversity' : loss_diversity, 'tail' : loss_tail};

        return LD_Loss_Container(losses = losses_dict, weights = self.loss_weights, params = params, metrics = metrics);


    def RHS(    self,
                Z       : list[list[torch.Tensor | numpy.ndarray]],
                t_Grid  : list[numpy.ndarray | torch.Tensor],
                params  : numpy.ndarray,
                sample  : bool = False) -> list[torch.Tensor | numpy.ndarray]:
        r"""
        Evaluate the CABSOLE mixture-of-second-order-experts right-hand side.

        For each parameter value, \theta, we evaluate

            z''(t) = \sum_{m = 1}^{N} w_m(t, \theta) [ K_m z(t) + C_m z'(t) + b_m ].
        
        at each latent displacement/velocity pair in `Z[i]`. CABSOLE is deterministic and owns one
        global expert set, so `sample` is accepted only for interface compatibility and ignored.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        Z : list[list[torch.Tensor | numpy.ndarray]], len = n_param
            The i'th element is a two-element list. `Z[i][0]` stores latent displacements and
            `Z[i][1]` stores latent velocities. Both entries must have shape (n_t(i), n_z).

        t_Grid : list[numpy.ndarray | torch.Tensor], len = n_param
            The i'th entry is a one-dimensional time grid with length n_t(i). CABSOLE is
            non-autonomous through its gate, so these times are used when computing expert weights.

        params : numpy.ndarray, shape = (n_param, n_p)
            Parameter rows corresponding to the latent states stored in Z.

        sample : bool
            Ignored. Present only to match the LatentDynamics interface.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        RH_Sides : list[numpy.ndarray | torch.Tensor], len = n_param
            The i'th entry has the same backend and leading dimensions as `Z[i][0]` and last
            dimension n_z. It stores the CABSOLE RHS evaluated at the supplied states/times.
        """

        # Checks.
        assert isinstance(params, numpy.ndarray), "params must be a 2d numpy.ndarray, not %s" % str(type(params));
        assert len(params.shape) == 2, "params must be a 2d numpy.ndarray of shape (n_param, n_p). Got shape %s" % str(params.shape);
        assert params.shape[1] == self.n_p;
        n_param : int = params.shape[0];
        assert isinstance(Z, list) and len(Z) == n_param;
        assert isinstance(t_Grid, list) and len(t_Grid) == n_param;

        # Compute right-hand sides.
        RH_Sides : list[numpy.ndarray | torch.Tensor] = [];
        LOGGER.debug("Computing CABSOLE RHS with %d parameter combinations" % n_param);
        for i in range(n_param):
            ith_Z       : numpy.ndarray | torch.Tensor  = Z[i][0];
            ith_dZ_dt   : numpy.ndarray | torch.Tensor  = Z[i][1];
            ith_t_Grid  : numpy.ndarray | torch.Tensor  = t_Grid[i];

            assert isinstance(ith_Z,        (numpy.ndarray, torch.Tensor));
            assert isinstance(ith_dZ_dt,    (numpy.ndarray, torch.Tensor));

            assert len(ith_Z.shape)         == 2;
            assert len(ith_dZ_dt.shape)     == 2;

            assert ith_Z.shape[-1] == self.n_z;
            assert ith_dZ_dt.shape[-1] == self.n_z;

            assert len(ith_t_Grid.shape) == 1;
            assert ith_Z.shape[0] == ith_t_Grid.shape[0];
            assert ith_dZ_dt.shape[0] == ith_t_Grid.shape[0];

            ith_RHS, _ = self._evaluate_rhs(
                                Z       = ith_Z,
                                dZ_dt   = ith_dZ_dt,
                                t_Grid  = ith_t_Grid,
                                params  = params[i, :],
                                t0      = ith_t_Grid[0],
                                t_span  = ith_t_Grid[-1] - ith_t_Grid[0]);
            RH_Sides.append(ith_RHS);
        
        # All done!
        return RH_Sides;



    def simulate(   self,
                    IC      : list[list[numpy.ndarray   | torch.Tensor]],
                    t_Grid  : list[numpy.ndarray        | torch.Tensor],
                    params  : numpy.ndarray,
                    sample  : bool = False) -> list[list[numpy.ndarray | torch.Tensor]]:
        r"""
        Time-integrate the deterministic CABSOLE latent dynamics.

        The gate is evaluated at the RK stage time and parameter value, so the integrated system is
        generally non-autonomous even though each expert is affine in z. The model is
        deterministic, so `sample` is accepted for interface compatibility but ignored.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        IC : list[list[numpy.ndarray]] or list[list[torch.Tensor]], len = n_param
            i'th element is a two-element list `[D0, V0]`. Both `D0` and `V0` must be
            one-dimensional arrays/tensors of shape (n_z), holding the initial displacement and
            velocity for the i'th parameter value.

        t_Grid : list[numpy.ndarray] or list[torch.Tensor], len = n_param
            i'th entry is a one-dimensional time grid of shape (n_t(i)).

        params: numpy.ndarray, shape = (n_param, n_p)
            The i'th row holds the i'th combination of parameter values.

        sample : bool 
            Ignored. Present only to match the LatentDynamics interface.

        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Z : list[list[numpy.ndarray]] or list[list[torch.Tensor]], len = n_parm
            i'th element is `[D, V]`. Both `D` and `V` have shape (n_t(i), n_z).
        """

        # Checks.
        assert isinstance(params, numpy.ndarray);
        assert len(params.shape) == 2;
        assert params.shape[1] == self.n_p;
        n_param : int = params.shape[0];
        assert isinstance(t_Grid, list) and isinstance(IC, list);
        assert len(IC) == n_param and len(t_Grid) == n_param;

        # Loop through parameter combinations.
        Z : list[list[numpy.ndarray | torch.Tensor]] = [];
        LOGGER.debug("Simulating CABSOLE with %d parameter combinations" % n_param);
        for i in range(n_param):
            ith_IC     : list[numpy.ndarray | torch.Tensor]  = IC[i];
            ith_t_Grid : numpy.ndarray | torch.Tensor        = t_Grid[i];
            ith_params : numpy.ndarray                       = params[i, :];

            assert isinstance(ith_IC, list) and len(ith_IC) == 2;
            if(isinstance(ith_t_Grid, torch.Tensor)):
                ith_t_Grid = ith_t_Grid.detach().cpu().numpy();
            assert len(ith_t_Grid.shape) == 1;
            t0          : float = ith_t_Grid[0];
            t_span      : float = ith_t_Grid[-1] - ith_t_Grid[0];
            ith_Z0      : numpy.ndarray | torch.Tensor = ith_IC[0];
            ith_dZ_dt0  : numpy.ndarray | torch.Tensor = ith_IC[1];
            assert(isinstance(ith_Z0, (numpy.ndarray, torch.Tensor)))
            assert(type(ith_dZ_dt0) == type(ith_Z0))
            assert len(ith_Z0.shape) == 1       and ith_Z0.shape[0]     == self.n_z;
            assert len(ith_dZ_dt0.shape) == 1   and ith_dZ_dt0.shape[0] == self.n_z;

            # Define the right-hand side in either NumPy or PyTorch. The solver backend follows the
            # initial-condition backend; this preserves differentiability for tensor rollouts. When
            # the gate is time/parameter-only, precompute its stage-time effective coefficients
            # once and let the generic RK4 solver use a cheap closure.
            if isinstance(ith_Z0, numpy.ndarray):
                def f(t : float, z : numpy.ndarray, dz_dt : numpy.ndarray) -> numpy.ndarray:
                    t_eval  : numpy.ndarray = numpy.asarray([t], dtype = ith_t_Grid.dtype);
                    RHS, _ = self._evaluate_rhs(Z = z.reshape(1, -1), dZ_dt = dz_dt.reshape(1, -1), t_Grid = t_eval, params = ith_params, t0 = t0, t_span = t_span);
                    return RHS.reshape(-1);
            else:
                if self.use_z_in_gate == False:
                    f = self._make_time_only_torch_rhs(
                        t_Grid = ith_t_Grid,
                        params = ith_params,
                        device = ith_Z0.device,
                        dtype  = ith_Z0.dtype);
                else:
                    def f(t : float, z : torch.Tensor, dz_dt : torch.Tensor) -> torch.Tensor:
                        param : torch.Tensor = next(self.w.parameters());
                        gate_device = param.device
                        gate_dtype  = param.dtype;

                        t_eval  : torch.Tensor = torch.tensor([t], dtype = gate_dtype, device = gate_device);
                        RHS, _  = self._evaluate_rhs(Z = z.reshape(1, -1), dZ_dt = dz_dt.reshape(1, -1), t_Grid = t_eval, params = ith_params, t0 = t0, t_span = t_span);
                        return RHS.reshape(-1);

            # Solve the ODE for this single latent initial state.
            ith_Z, ith_dZ_dt = RK4(f = f, y0 = ith_Z0, Dy0 = ith_dZ_dt0, t_Grid = ith_t_Grid);

            # Add this parameter's trajectory to the output list.
            Z.append([ith_Z, ith_dZ_dt]);

        # All done!
        return Z;


    # ---------------------------------------------------------------------------------------------
    # Serialization
    # ---------------------------------------------------------------------------------------------


    def export(self) -> dict:
        r"""Export CABSOLE metadata, expert tensors, and gate-network parameters."""

        param_dict = {'n_z'             : self.n_z,
                      'n_IC'            : self.n_IC,
                      'n_p'             : self.n_p,
                      'config'          : self.config.model_dump(mode = "python", by_alias = True),
                      'Uniform_t_Grid'  : self.Uniform_t_Grid,
                      'unmasked_K'      : self.unmasked_K.detach().cpu().clone(),
                      'unmasked_C'      : self.unmasked_C.detach().cpu().clone(),
                      'unmasked_b'      : None if self.unmasked_b is None else self.unmasked_b.detach().cpu().clone(),
                      'K_mask'          : self.K_mask.detach().cpu().clone(),
                      'C_mask'          : self.C_mask.detach().cpu().clone(),
                      'b_mask'          : None if self.b_mask is None else self.b_mask.detach().cpu().clone(),
                      'w_state_dict'    : {key: value.detach().cpu().clone() for key, value in self.w.state_dict().items()}};
        return param_dict;


    def load(self, dict_ : dict) -> None:
        r"""Load CABSOLE metadata, expert tensors, and gate-network parameters."""

        assert(self.n_z             == dict_['n_z']);
        assert(self.n_IC            == dict_['n_IC']);
        assert(self.n_p             == dict_['n_p']);
        assert(self.Uniform_t_Grid  == dict_['Uniform_t_Grid']);

        # Fetch the unmasked K, C tensors.
        unmasked_K : torch.Tensor = dict_['unmasked_K'];
        unmasked_C : torch.Tensor = dict_['unmasked_C'];
        assert isinstance(unmasked_K, torch.Tensor) and unmasked_K.shape == (self.n_experts, self.n_z, self.n_z);
        assert isinstance(unmasked_C, torch.Tensor) and unmasked_C.shape == (self.n_experts, self.n_z, self.n_z);
        self.unmasked_K = unmasked_K.detach().clone().requires_grad_(self.trainable);
        self.unmasked_C = unmasked_C.detach().clone().requires_grad_(self.trainable);

        # Do the same for unmasked b.
        if self.use_biases:
            unmasked_b = dict_['unmasked_b'];
            assert isinstance(unmasked_b, torch.Tensor) and unmasked_b.shape == (self.n_experts, 1, self.n_z);
            self.unmasked_b = unmasked_b.detach().clone().requires_grad_(self.trainable);
        else:
            self.unmasked_b = None;

        # Now fetch the K, C mask
        K_mask = dict_.get('K_mask', torch.ones_like(self.unmasked_K));
        C_mask = dict_.get('C_mask', torch.ones_like(self.unmasked_K));
        assert isinstance(K_mask, torch.Tensor) and K_mask.shape == (self.n_experts, self.n_z, self.n_z);
        assert isinstance(C_mask, torch.Tensor) and C_mask.shape == (self.n_experts, self.n_z, self.n_z);
        self.K_mask = K_mask.detach().clone().to(device = self.unmasked_K.device, dtype = self.unmasked_K.dtype);
        self.C_mask = C_mask.detach().clone().to(device = self.unmasked_K.device, dtype = self.unmasked_K.dtype);

        # and the b mask
        if self.use_biases:
            assert self.unmasked_b is not None;
            b_mask = dict_.get('b_mask', torch.ones_like(self.unmasked_b));
            assert isinstance(b_mask, torch.Tensor) and b_mask.shape == (self.n_experts, 1, self.n_z);
            self.b_mask = b_mask.detach().clone().to(device = self.unmasked_b.device, dtype = self.unmasked_b.dtype);
        else:
            self.b_mask = None;

        # Finally, load the gate network
        self.w.load_state_dict(dict_['w_state_dict']);
        for param in self.w.parameters():
            param.requires_grad_(self.trainable);

        # All done :) 
        return;


    # ---------------------------------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------------------------------

    def _weights(
            self,
            t_Grid  : numpy.ndarray | torch.Tensor,
            Z       : numpy.ndarray | torch.Tensor,
            dZ_dt    : numpy.ndarray | torch.Tensor,
            params  : numpy.ndarray | torch.Tensor,
            *,
            t0      : float, 
            t_span  : float) -> torch.Tensor:
        r"""
        Evaluate gate weights on one time/state trajectory and one parameter value.

        The returned tensor has shape (n_t, n_experts) and lives on the same device/dtype as the
        gate network. Callers can cast it to the latent-state backend before evaluating experts.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        t_Grid : numpy.ndarray or torch.Tensor, shape = (n_t)
            One-dimensional time grid for a single parameter value. NumPy inputs may have any
            floating dtype. Torch inputs may live on any device. The values are cast to the gate
            network's dtype/device before forming gate inputs. 

        Z : numpy.ndarray or torch.Tensor, shape = (n_t, n_z) or (n_z)
            Latent states corresponding to `t_Grid`. One-dimensional inputs are accepted only for
            one time sample. These values are concatenated to the gate input only when
            `self.use_z_in_gate` is enabled.

        dZ_dt : numpy.ndarray or torch.Tensor, shape = (n_t, n_z) or (n_z)
            Time derivatives of the latent states corresponding to `t_Grid`. One-dimensional 
            inputs are accepted only for one time sample. These values are concatenated to the 
            gate input only when `self.use_z_in_gate` is enabled.
        
        params : numpy.ndarray | torch.Tensor, shape = (n_p)
            Parameter vector for the same trajectory. The values are cast to the gate network's
            dtype/device and broadcast to shape (n_t, n_p).

        t0 : float 
            The starting time for this parameter value; used to scale time inputs to gate network.
        
        t_span : float
            The difference between the minimum and maximum time for this parameter value; used to 
            scale time inputs to gate network.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        weights : torch.Tensor, shape = (n_t, n_experts)
           expert weights evaluated at all (t, params) pairs. The dtype and device match the 
           gate-network parameters. Each row sums to one.
        """

        # Setup 
        w_param : torch.Tensor = next(self.w.parameters());
        gate_device = w_param.device
        gate_dtype  = w_param.dtype;

        # -----------------------------------------------------------------------------------------
        # Build gate network inputs

        # Normalize the times. 
        tau_Grid = (t_Grid - t0)/t_span;

        # Map tau_Grid to a tensor.
        # The gate is a torch.nn.Module; its inputs must be tensors.
        if isinstance(tau_Grid, numpy.ndarray):
            tau_tensor : torch.Tensor = torch.tensor(tau_Grid, dtype = gate_dtype, device = gate_device);
        else:
            tau_tensor = tau_Grid.to(device = gate_device, dtype = gate_dtype);
        assert len(tau_tensor.shape) == 1;

        # Check that the latent states correspond to the same number of time samples.
        if len(Z.shape) == 1:
            assert Z.shape[0] == self.n_z;
            assert tau_tensor.shape[0] == 1;
        else:
            assert len(Z.shape) == 2 and Z.shape[1] == self.n_z;
            assert Z.shape[0] == tau_tensor.shape[0];
        
        if len(dZ_dt.shape) == 1:
            assert dZ_dt.shape[0] == self.n_z;
            assert tau_tensor.shape[0] == 1;
        else:
            assert len(dZ_dt.shape) == 2 and dZ_dt.shape[1] == self.n_z;
            assert dZ_dt.shape[0] == tau_tensor.shape[0];

        # Broadcast n_t copies of the parameter tensor to build inputs for the gate network.
        if isinstance(params, numpy.ndarray):
            param_tensor : torch.Tensor = torch.tensor(params, dtype = gate_dtype, device = gate_device);
        else:
            param_tensor = params.to(device = gate_device, dtype = gate_dtype);
        param_tensor = param_tensor.reshape(1, self.n_p).expand(tau_tensor.shape[0], self.n_p);

        # Build the gate network inputs
        w_inputs : torch.Tensor = torch.cat([tau_tensor.reshape(-1, 1), param_tensor], dim = 1);
        if self.use_z_in_gate:
            if isinstance(Z, numpy.ndarray):
                z_tensor : torch.Tensor = torch.tensor(Z, dtype = gate_dtype, device = gate_device);
            else:
                z_tensor = Z.to(device = gate_device, dtype = gate_dtype);
            if isinstance(dZ_dt, numpy.ndarray):
                dzdt_tensor : torch.Tensor = torch.tensor(dZ_dt, dtype = gate_dtype, device = gate_device);
            else:
                dzdt_tensor = dZ_dt.to(device = gate_device, dtype = gate_dtype);
            
            z_tensor    = z_tensor.reshape(tau_tensor.shape[0], self.n_z);
            dzdt_tensor = dzdt_tensor.reshape(tau_tensor.shape[0], self.n_z);

            w_inputs = torch.cat([w_inputs, z_tensor, dzdt_tensor], dim = 1);

        # -----------------------------------------------------------------------------------------
        # Evaluate weights

        # Compute logits
        logits : torch.Tensor = self.w(w_inputs);

        # Compute weights by applying a soft max to the logits.
        weights : torch.Tensor = torch.softmax(logits, dim = 1);

        # All done :) 
        return weights;


    def _effective_coefficients(
            self,
            weights : torch.Tensor,
            device  : torch.device,
            dtype   : torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Collapse expert coefficients into one affine system per time sample.

        Given weights w[t, m], this forms

            K_bar[t] = sum_m w[t, m] K[m],
            C_bar[t] = sum_m w[t, m] C[m],
            b_bar[t] = sum_m w[t, m] b[m].
            

        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        weights : torch.Tensor, shape = (n_t, n_experts)
            Expert weights for one trajectory. This tensor may live on a different device/dtype
            than the requested output; it is cast to `device` and `dtype` internally.

        device : torch.device
            Destination device for the returned effective coefficients.

        dtype : torch.dtype
            Destination floating-point dtype for the returned effective coefficients.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        K_bar : torch.Tensor, shape = (n_t, n_z, n_z)
            Time-dependent effective stiffness operator, on `device` with dtype `dtype`.

        C_bar : torch.Tensor, shape = (n_t, n_z, n_z)
            Time-dependent effective damping operators, on `device` with dtype `dtype`.
            
        b_bar : torch.Tensor, shape = (n_t, n_z)
            Time-dependent effective affine shifts, on `device` with dtype `dtype`.
        """

        weights = weights.to(device = device, dtype = dtype);

        # Compute effective K.
        K       : torch.Tensor = self.K.to(device = device, dtype = dtype);
        K_flat  : torch.Tensor = K.reshape(self.n_experts, self.n_z*self.n_z);
        K_bar   : torch.Tensor = (weights @ K_flat).reshape(weights.shape[0], self.n_z, self.n_z);

        # Do the same for C.
        C       : torch.Tensor = self.C.to(device = device, dtype = dtype);
        C_flat  : torch.Tensor = C.reshape(self.n_experts, self.n_z*self.n_z);
        C_bar   : torch.Tensor = (weights @ C_flat).reshape(weights.shape[0], self.n_z, self.n_z);

        # Compute effective b, if we need to.
        if self.b is None:
            b_bar                  = torch.zeros((weights.shape[0], self.n_z), dtype = dtype, device = device);
        else:
            b                      = self.b.to(device = device, dtype = dtype).reshape(self.n_experts, self.n_z);
            b_bar   : torch.Tensor = weights @ b;

        # All done :) 
        return K_bar, C_bar, b_bar;


    @property
    def K(self) -> torch.Tensor:
        r"""
        Return the effective stiffness matrices after applying the hard mask.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        K : torch.Tensor, shape = (n_experts, n_z, n_z)
            Expert matrices with masked entries set to zero.
        """

        if self.use_mask:
            return self.unmasked_K * self.K_mask.to(device = self.unmasked_K.device, dtype = self.unmasked_K.dtype);
        return self.unmasked_K;


    @property
    def C(self) -> torch.Tensor:
        r"""
        Return the effective damping matrices after applying the hard mask.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        C : torch.Tensor, shape = (n_experts, n_z, n_z)
            Expert matrices with masked entries set to zero.
        """

        if self.use_mask:
            return self.unmasked_C * self.C_mask.to(device = self.unmasked_C.device, dtype = self.unmasked_C.dtype);
        return self.unmasked_C;


    @property
    def b(self) -> torch.Tensor | None:
        r"""
        Return the effective expert biases after applying the hard mask.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        b : torch.Tensor or None, shape = (n_experts, 1, n_z)
            Expert biases with masked entries set to zero. Returns None when biases are disabled.
        """

        if self.unmasked_b is None:
            return None;
        if self.use_mask:
            assert self.b_mask is not None;
            return self.unmasked_b * self.b_mask.to(device = self.unmasked_b.device, dtype = self.unmasked_b.dtype);
        return self.unmasked_b;


    @torch.no_grad()
    def _update_mask(self) -> tuple[int, int]:
        r"""
        Permanently mask small expert coefficients.

        Any active matrix or bias entry whose current effective absolute value is below
        `self.mask_threshold` is set to zero in the hard mask. Previously masked entries remain
        masked because the update multiplies the old mask by the new keep-mask.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Nothing!
        """

        assert self.mask_threshold is not None;

        # Fetch the current mask.
        self.K_mask = self.K_mask.to(device = self.unmasked_K.device, dtype = self.unmasked_K.dtype);
        self.C_mask = self.C_mask.to(device = self.unmasked_C.device, dtype = self.unmasked_C.dtype);

        # Determine which components of K, C are bigger than the threshold.
        K_keep      : torch.Tensor = (self.K.abs() >= self.mask_threshold).to(dtype = self.unmasked_K.dtype);
        C_keep      : torch.Tensor = (self.C.abs() >= self.mask_threshold).to(dtype = self.unmasked_C.dtype);

        # Update the mask; note that any previously masked components remain masked.
        self.K_mask = (self.K_mask * K_keep).contiguous();
        self.C_mask = (self.C_mask * C_keep).contiguous();

        # Update K, C.
        self.unmasked_K.data.mul_(self.K_mask);
        self.unmasked_C.data.mul_(self.C_mask);
        n_active : int = int(self.K_mask.sum().item()) + int(self.C_mask.sum().item());
        n_total  : int = int(self.K_mask.numel()) + int(self.C_mask.numel());

        # Update b, if it exists
        if self.unmasked_b is not None:
            assert self.b_mask is not None;
            self.b_mask = self.b_mask.to(device = self.unmasked_b.device, dtype = self.unmasked_b.dtype);
            b           : torch.Tensor = self.b;
            assert b is not None;
            b_keep      : torch.Tensor = (b.abs() >= self.mask_threshold).to(dtype = self.unmasked_b.dtype);
            self.b_mask = (self.b_mask * b_keep).contiguous();
            self.unmasked_b.data.mul_(self.b_mask);
            n_active   += int(self.b_mask.sum().item());
            n_total    += int(self.b_mask.numel());

        # Report masking information 
        LOGGER.info("%d/%d coefficients are still active across %d experts" % (n_active, n_total, self.n_experts));
        return n_active, n_total;


    def _make_time_only_torch_rhs(
            self,
            t_Grid  : numpy.ndarray,
            params  : numpy.ndarray,
            device  : torch.device,
            dtype   : torch.dtype) -> Callable[[float, torch.Tensor, torch.Tensor], torch.Tensor]:
        r"""
        Build a cached torch RHS closure for time/parameter-only gates.

        When `self.use_z_in_gate` is False, the gate weights depend only on the RK stage time and
        the parameter value. In that case, all gate weights and effective affine coefficients used
        by RK4 can be evaluated once before time stepping. The returned closure still has the
        generic solver signature `f(t, z, z')`, but each call only performs the affine map 
        associated with that stage time:

            f(t, z, z') = K_bar(t) z + C_bar(t) dz_dt + b_bar(t).

        This method intentionally does not modify or specialize `RK4`; it only constructs a faster
        CABSOLE-specific right-hand-side function for the time-only gate case.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        t_Grid : numpy.ndarray, shape = (n_t)
            One-dimensional time grid passed to RK4. The cached stage times are generated using
            the same arithmetic as the generic RK4 implementation.

        params : numpy.ndarray, shape = (n_p)
            Parameter vector associated with this rollout.

        device : torch.device
            Device on which the returned RHS should evaluate latent-state operations.

        dtype : torch.dtype
            Floating-point dtype for the returned RHS values.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        f : Callable
            Function with signature `f(t, z, dz_dt)` returning a torch.Tensor with the same shape
            as `z`.
        """

        assert self.use_z_in_gate == False;
        assert isinstance(t_Grid, numpy.ndarray);
        assert len(t_Grid.shape) == 1;
        assert isinstance(params, numpy.ndarray) and params.shape == (self.n_p,);

        # Build exactly the stage times requested by the generic RK4 implementation. We key the
        # cache by Python floats because RK4 passes scalar NumPy/Python times to the closure.
        stage_times : list[float]       = [];
        time_to_idx : dict[float, int]  = {};

        def add_time(t : float) -> None:
            key : float = float(t);
            if key not in time_to_idx:
                time_to_idx[key] = len(stage_times);
                stage_times.append(key);
            return;

        for n in range(t_Grid.size - 1):
            tn = t_Grid[n];
            hn = t_Grid[n + 1] - t_Grid[n];
            add_time(tn);
            add_time(tn + hn/2);
            add_time(tn + hn);

        stage_t_Grid : numpy.ndarray = numpy.asarray(stage_times, dtype = t_Grid.dtype);

        # `_weights` ignores Z and dZ_dt when `use_z_in_gate` is False, but it still validates
        # compatible shapes. Use empty tensors on the gate device to avoid needless host/device
        # copies.
        w_param : torch.Tensor = next(self.w.parameters());
        dummy_Z : torch.Tensor = torch.empty(
            (stage_t_Grid.shape[0], self.n_z),
            dtype  = w_param.dtype,
            device = w_param.device);
        dummy_dZ_dt : torch.Tensor = torch.empty_like(dummy_Z);

        # Evaluate all gate weights and collapse all effective affine systems in one batched call.
        # Do not use no_grad: training may differentiate through simulated trajectories.
        weights : torch.Tensor = self._weights(
            stage_t_Grid,
            dummy_Z,
            dummy_dZ_dt,
            params,
            t0      = t_Grid[0],
            t_span  = t_Grid[-1] - t_Grid[0]);
        K_bar, C_bar, b_bar = self._effective_coefficients(weights, device, dtype);

        # Split K, C, b into lists (speeds up indexing)
        K_slices = list(torch.unbind(K_bar, dim=0))
        C_slices = list(torch.unbind(C_bar, dim=0))
        b_slices = list(torch.unbind(b_bar, dim=0))

        # Make the final method.
        def f(t: float, z: torch.Tensor, dz_dt: torch.Tensor) -> torch.Tensor:
            idx = time_to_idx[float(t)]
            return torch.matmul(z, K_slices[idx].T) + torch.matmul(dz_dt, C_slices[idx].T) + b_slices[idx]

        # All done :)
        return f;


    def _evaluate_rhs(
            self,
            *,
            Z       : numpy.ndarray | torch.Tensor,
            dZ_dt   : numpy.ndarray | torch.Tensor,
            t_Grid  : numpy.ndarray | torch.Tensor,
            params  : numpy.ndarray | torch.Tensor,
            t0      : float | torch.Tensor,
            t_span  : float | torch.Tensor) -> tuple[numpy.ndarray | torch.Tensor, torch.Tensor]:
        r"""
        Evaluate CABSOLE's right-hand side and return the corresponding gate weights.

        This is the single pointwise CABSOLE RHS helper used by loss evaluation, public RHS calls,
        and the generic simulation path. The backend of `Z` determines the backend of the returned
        RHS. NumPy inputs are evaluated under `torch.no_grad()` and converted back to NumPy; torch
        inputs preserve autograd through the gate, expert coefficients, and affine evaluation.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        Z : numpy.ndarray | torch.Tensor, shape = (n_t, n_z)
            Latent states at which to evaluate the RHS. The returned RHS has the same backend, 
            dtype, and shape as `Z`.

        dZ_dt : numpy.ndarray | torch.Tensor, shape = (n_t, n_z)
            Time derivative of the latent states at which to evaluate the RHS. 
        
        t_Grid : numpy.ndarray or torch.Tensor, shape = (n_t)
            One-dimensional time grid corresponding to the first dimension of `Z` or `dZ_dt`. 
            Values are used by the gate network.

        params : numpy.ndarray | torch.Tensor, shape = (n_p)
            Parameter vector associated with `Z`. Values are used by the gate network.

        t0 : float or torch.Tensor
            Time origin used to normalize gate inputs. 

        t_span : float or torch.Tensor
            Time span used to normalize gate inputs. 


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        RHS, weights 

        RHS : numpy.ndarray | torch.Tensor, shape = Z.shape
            CABSOLE right-hand-side values; will have the same type as Z. 

        weights : torch.Tensor, shape = (n_t, n_experts)
            The expert weights at each time step.
        """

        # Check and normalize the latent-state shape. 
        assert isinstance(Z, (numpy.ndarray, torch.Tensor));
        assert isinstance(dZ_dt, (numpy.ndarray, torch.Tensor));
        assert type(Z) == type(dZ_dt);
        assert len(Z.shape)         == 2    and Z.shape[1]      == self.n_z;
        assert len(dZ_dt.shape)     == 2    and dZ_dt.shape[1]  == self.n_z;
        assert len(t_Grid.shape)    == 1;
        assert t_Grid.shape[0]      == dZ_dt.shape[0];

        if isinstance(Z, numpy.ndarray):
            with torch.no_grad():
                weights         : torch.Tensor = self._weights(t_Grid = t_Grid, Z = Z, dZ_dt = dZ_dt, params = params, t0 = t0, t_span = t_span);
                if Z.dtype == numpy.dtype(numpy.float64):
                    dtype = torch.float64;
                else:
                    dtype = torch.float32;
                K_bar, C_bar, b_bar     = self._effective_coefficients(weights, torch.device("cpu"), dtype);
                K_np : numpy.ndarray    = K_bar.detach().cpu().numpy().astype(Z.dtype, copy = False);
                C_np : numpy.ndarray    = C_bar.detach().cpu().numpy().astype(Z.dtype, copy = False);
                b_np : numpy.ndarray    = b_bar.detach().cpu().numpy().astype(Z.dtype, copy = False);

            # For each time t, compute K_np[t] @ Z[t] + C_np[t] @ dZ_dt[t] + b_np. 
            # Shapes:
            #   K_np         : (n_t, n_z, n_z)
            #   C_np         : (n_t, n_z, n_z)
            #   Z[..., None] : (n_t, n_z, 1)
            # numpy.matmul returns (n_t, n_z, 1), then squeeze gives (n_t, n_z).
            RHS : numpy.ndarray = numpy.matmul(K_np, Z[..., None]).squeeze(-1) + numpy.matmul(C_np, dZ_dt[..., None]).squeeze(-1) + b_np;
            return RHS, weights;

        else: 
            assert isinstance(Z, torch.Tensor);
            weights         : torch.Tensor  = self._weights(t_Grid = t_Grid, Z = Z, dZ_dt = dZ_dt, params = params, t0 = t0, t_span = t_span);
            K_bar, C_bar, b_bar             = self._effective_coefficients(weights, Z.device, Z.dtype);

            # For each time t, compute K_bar[t] @ Z[t] + C_bar[t] @ dZ_dt[t] + b. 
            # Shapes:
            #   K_np         : (n_t, n_z, n_z)
            #   C_np         : (n_t, n_z, n_z)
            #   Z[..., None] : (n_t, n_z, 1)
            # torch.bmm returns (n_t, n_z, 1), then squeeze gives (n_t, n_z).
            RHS : torch.Tensor = torch.bmm(K_bar, Z.unsqueeze(-1)).squeeze(-1) + torch.bmm(C_bar, dZ_dt.unsqueeze(-1)).squeeze(-1) + b_bar;
            return RHS, weights;
