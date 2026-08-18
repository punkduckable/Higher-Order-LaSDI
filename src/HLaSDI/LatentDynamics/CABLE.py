# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

import  logging;

import  numpy;
import  torch;

from    HLaSDI.LatentDynamics.LatentDynamics    import  LatentDynamics, LD_Loss_Container;
from    HLaSDI.Schemas                          import  CABLELatentDynamicsConfig, CABLELatentDynamicsSettings, WeakCABLELatentDynamicsConfig;
from    HLaSDI.EncoderDecoder                   import  MultiLayerPerceptron;
from    HLaSDI.Utilities.FiniteDifference       import  Derivative1_Order4, Derivative1_Order2_NonUniform;
from    HLaSDI.Utilities.FirstOrderSolvers      import  RK4;
from    HLaSDI.Utilities.Statistics             import  tensor_statistics;

LOGGER  : logging.Logger    = logging.getLogger(__name__);



# -------------------------------------------------------------------------------------------------
# CABLE class
# -------------------------------------------------------------------------------------------------

class CABLE(LatentDynamics):
    def __init__(   self, 
                    n_z             : int,
                    Uniform_t_Grid  : bool,
                    n_p             : int,
                    config          : CABLELatentDynamicsConfig) -> None:
        r"""
        Initialize a CABLE latent-dynamics model.

        CABLE is a deterministic mixture-of-linear/affine-experts latent ODE. For a parameter value
        \theta and time t, it evolves the latent state according to

            z'(t) = \sum_{m = 1}^{N} w_m(t, \theta) [ A_m z(t) + b_m ],

        when biases are enabled, and according to

            z'(t) = \sum_{m = 1}^{N} w_m(t, \theta) A_m z(t),

        otherwise. Here N is the number of experts, each A_m is an n_z x n_z matrix, each enabled
        b_m is an n_z-vector, and w_m(t, \theta) is the gate weight assigned to the m'th expert. The gate is
        a neural network

            w : \mathbb{R}^{1 + n_p} \to \mathbb{R}^{N}

        We do not hard-threshold the gate during RHS evaluation. Instead, `n_active` is a soft
        target used by the tail-mass loss: training can encourage most softmax mass to live on a
        small number of experts without introducing a discontinuous top-k cutoff.

        The trainable latent-dynamics state consists of the expert matrices, optional expert
        biases, and gate-network parameters. If coefficient masking is enabled, matrix and bias
        entries whose absolute values fall below the mask threshold are permanently removed from
        the effective latent dynamics. Unlike interpolatable SINDy-type models, CABLE owns one
        global set of parameters rather than one coefficient dictionary per training parameter.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        n_z : int
            The number of latent dimensions.

        Uniform_t_Grid : bool
            If True, each trajectory has uniform time spacing and we can use the higher-order
            finite-difference stencil. Otherwise, nonuniform-grid finite differences are used.

        n_p : int 
            The number of (scalar) parameters in the parameter space.

        config : CABLELatentDynamicsConfig
            CABLE latent-dynamics configuration. The `cable` settings specify the number of
            experts, the target number of active experts, epsilon, and the gate-network 
            architecture.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Nothing!
        """

        assert isinstance(config, (CABLELatentDynamicsConfig, WeakCABLELatentDynamicsConfig)), "config must be a CABLELatentDynamicsConfig, got %s" % str(type(config));

        # Run the base class initializer. 
        LatentDynamics.__init__(
            self,
            n_z            = n_z, 
            n_IC           = 1, 
            n_p            = n_p,
            Uniform_t_Grid = Uniform_t_Grid,
            trainable      = config.trainable,
            stochastic     = False,
            config         = config);

        # Extract sub-class specific attributes.
        sub : CABLELatentDynamicsSettings       = config.cable;
        self.n_experts          : int           = sub.n_experts;
        self.n_active           : int           = sub.n_active;
        self.hidden_widths      : list[int]     = sub.hidden_widths;
        self.activations        : list[str]     = sub.activations;
        self.coef_norm          : str           = sub.coef_norm
        self.use_biases         : bool          = sub.use_biases;
        self.eps_engaged        : float         = sub.eps_engaged;
        self.use_mask           : bool          = sub.use_mask;
        self.mask_threshold     : float | None  = sub.mask_threshold;
        self.first_mask_step    : int   | None  = sub.first_mask_step;
        self.mask_update_freq   : int   | None  = sub.mask_update_freq;

        # Initialize the gate network.
        widths      : list[int] = [n_p + 1] + self.hidden_widths + [self.n_experts];
        self.w                  = MultiLayerPerceptron(widths = widths, activations = self.activations);
        with torch.no_grad():
            for param in self.w.parameters():
                param.mul_(0.01);

        # Randomly initialize the experts. These are leaf tensors because the Trainer passes them
        # directly to the optimizer through parameters().
        self.unmasked_A : torch.Tensor = (0.01*torch.rand((self.n_experts, self.n_z, self.n_z), dtype = torch.float32)).requires_grad_(self.trainable);
        self.unmasked_b : torch.Tensor | None;
        if self.use_biases:
            self.unmasked_b = torch.zeros((self.n_experts, 1, self.n_z), dtype = torch.float32).requires_grad_(self.trainable);
        else:
            self.unmasked_b = None;

        # Hard coefficient masks. A value of one means active and zero means permanently removed
        # from the effective latent dynamics.
        self.A_mask : torch.Tensor = torch.ones_like(self.unmasked_A);
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
        Return CABLE-owned tensors that should be passed to torch optimizers.

        These are the expert matrices, optional expert biases, and gate-network parameters. The list is
        empty when the latent dynamics are frozen.
        """

        if self.trainable == False:
            return [];

        # Append un-masked A, b
        tensors : list[torch.Tensor] = [self.unmasked_A];
        if self.unmasked_b is not None:
            tensors.append(self.unmasked_b);

        # Append gate network parameters.
        for param in self.w.parameters():
            tensors.append(param);
        return tensors;


    def move_parameters_to_device(self, device : torch.device | str) -> None:
        r"""
        Move CABLE-owned parameters to a device.


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

        # Keep A and b as leaf tensors because the Trainer optimizes them directly.
        self.unmasked_A = self.unmasked_A.detach().to(device = device).requires_grad_(self.trainable);
        self.A_mask     = self.A_mask.to(device = device, dtype = self.unmasked_A.dtype);
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
        Move the globally initialized CABLE parameters to the requested device.

        CABLE does not fit one coefficient dictionary per training parameter. Its experts and gate
        are initialized when the object is constructed and then trained directly. This method keeps
        the standard latent-dynamics initialization hook but only validates the incoming training
        data and moves CABLE-owned tensors to `device`.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        Latent_States : list[list[torch.Tensor]], len = n_param
            The i'th entry contains one tensor with shape (n_t(i), n_z), holding the encoded latent
            trajectory for the i'th parameter.

        t_Grid : list[torch.Tensor], len = n_param
            Time grid for each latent trajectory.

        device : torch.device
            The device where CABLE's experts and gate network should live.
            
        params : numpy.ndarray, shape = (n_param, n_p)
            The parameters currently represented in the training set.
            
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        None.
        """

        # Checks.
        assert params is not None, "CABLE.initialize_coefficients requires params!";
        assert isinstance(params, numpy.ndarray) and len(params.shape) == 2;
        assert params.shape[1] == self.n_p;
        assert isinstance(t_Grid, list);
        assert isinstance(Latent_States, list);
        assert len(Latent_States) == len(t_Grid) == params.shape[0];

        # Move A, optional b, masks, and w to specified device.
        self.move_parameters_to_device(device);
        return None;

    
    # ---------------------------------------------------------------------------------------------
    # Compute Losses, RHS, and Simulate
    # ---------------------------------------------------------------------------------------------


    def compute_losses(  
        self,  
        Latent_States   : list[list[torch.Tensor]], 
        t_Grid          : list[torch.Tensor], 
        step            : int,
        params          : numpy.ndarray | None = None
    ) -> LD_Loss_Container:
        r"""
        Compute CABLE latent-dynamics, coefficient, and gate-diversity losses.

        For each parameter value, this method computes a finite-difference approximation to
        dZ/dt, evaluates the deterministic mixture-of-experts right-hand side at each time sample,
        and compares the two. 

        The coefficient loss penalizes the size of the affine experts. The stability loss is a
        deterministic load-balancing/diversity surrogate computed from the softmax
        weights: it accumulates each expert's total dense gate weight across all training
        parameters and times, then returns the squared coefficient of variation of those totals.
        This is zero when all experts receive the same total dense load and positive when the dense
        gate collapses onto a subset of experts.

        This method also computes a per-parameter tail-mass penalty,

            mean_t (1 - sum_{m in top-n_active(t,theta_i)} q_m(t,theta_i))^2,

        where q is the dense softmax over all experts, and k = self.n_active. This quantifies how 
        much probability mass is outside of the top n_active experts. 


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        Latent_States : list[list[torch.Tensor]], len = n_param
            Encoded latent trajectories. The i'th entry contains one tensor of shape (n_t(i), n_z).

        t_Grid : list[torch.Tensor], len = n_param
            Time grids corresponding to the latent trajectories.
        
        step : int
            The optimizer step number.

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
        assert params is not None, "CABLE.compute_losses requires params for the gate network";
        assert isinstance(params, numpy.ndarray) and len(params.shape) == 2;
        assert params.shape[1] == self.n_p;
        assert isinstance(t_Grid, list);
        assert isinstance(Latent_States, list);
        assert len(Latent_States) == len(t_Grid) == params.shape[0];
        assert len(t_Grid) > 0;

        # Setup
        loss_LD_list        : list[torch.Tensor]        = [];
        summed_weights      : torch.Tensor              = torch.zeros((self.n_experts), dtype = self.unmasked_A.dtype, device = self.unmasked_A.device);
        times_engaged       : torch.Tensor              = torch.zeros((self.n_experts), dtype = torch.int64, device = self.unmasked_A.device);
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
            metrics["n_active/A"] = self.A_mask.sum().to(device = self.unmasked_A.device, dtype = self.unmasked_A.dtype).detach();
            if self.use_biases:
                metrics["n_active/b"] = self.b_mask.sum().to(device = self.unmasked_A.device, dtype = self.unmasked_A.dtype).detach();

        n_param     : int           = len(t_Grid);
        for i in range(n_param):
            # Fetch this parameter's latent trajectory and time grid.
            ith_params  : numpy.ndarray = params[i, :]
            ith_t_Grid  : torch.Tensor  = t_Grid[i];
            ith_Z       : torch.Tensor  = Latent_States[i][0]; # [n_t_i, n_z]
            n_t_i       : int           = len(ith_t_Grid);
            assert isinstance(ith_Z, torch.Tensor);
            assert len(ith_Z.shape) == 2 and ith_Z.shape[1] == self.n_z;
            assert len(ith_t_Grid.shape) == 1 and ith_t_Grid.shape[0] == n_t_i;
            assert ith_Z.shape[0] == n_t_i;

            # Compute dZ/dt. Uniform grids use the higher-order stencil; nonuniform grids use the
            # nonuniform finite-difference helper.
            if(self.Uniform_t_Grid == True):
                h       : float         = (ith_t_Grid[1] - ith_t_Grid[0]).item();
                dZdt    : torch.Tensor  = Derivative1_Order4(ith_Z, h);
            else:
                dZdt                    = Derivative1_Order2_NonUniform(ith_Z, t_Grid = ith_t_Grid);

            # Evaluate expert weights.
            ith_weights       : torch.Tensor = self._weights_for_t_grid(ith_t_Grid, ith_params, t0 = ith_t_Grid[0], t_span = ith_t_Grid[-1] - ith_t_Grid[0]);
            weights_list.append(ith_weights.to(device = self.unmasked_A.device, dtype = self.unmasked_A.dtype));
            ith_RHS           : torch.Tensor = self._evaluate_torch_rhs_from_weights(ith_Z, ith_weights);

            # Record which experts are engaged during each step for this parameter.
            ith_engaged : torch.Tensor = (ith_weights > self.eps_engaged).to(dtype = torch.bool, device = self.unmasked_A.device)
            times_engaged += torch.sum(ith_engaged, dim = 0);
            n_engaged_list.append(torch.sum(ith_engaged, dim = 1).to(device = self.unmasked_A.device, dtype = self.unmasked_A.dtype));

            # Compute the LD loss for the i'th combination of parameters.
            ith_loss_LD = self.MSE(dZdt, ith_RHS);
            loss_LD_list.append(ith_loss_LD);
            metrics[f"loss/LD/{str(ith_params)}"] = ith_loss_LD.detach();

            # Accumulate expert loads across all parameter values and times. This is a 
            # deterministic analogue of MoE importance/load diversity: it encourages all
            # experts to be useful somewhere without forcing all experts to be active at 
            # every step.
            summed_weights = summed_weights + torch.sum(ith_weights.to(device = self.unmasked_A.device, dtype = self.unmasked_A.dtype), dim = 0);

            # Tail-mass penalty: compute how much softmax mass lies outside the top-n_active 
            # logits. If this is small, most of the probability mass is in the top n_active 
            # experts, as intended. 
            if self.n_active >= self.n_experts:
                ith_tail_mass : torch.Tensor = torch.zeros((n_t_i), dtype = ith_weights.dtype, device = ith_weights.device);
            else:
                ith_topk_idx             : torch.Tensor = torch.topk(ith_weights, self.n_active, dim = 1, sorted = False).indices;
                ith_topk_dense_mass      : torch.Tensor = torch.sum(ith_weights.gather(1, ith_topk_idx), dim = 1);
                ith_tail_mass            : torch.Tensor = 1.0 - ith_topk_dense_mass;
            tail_mass_list.append(ith_tail_mass.to(device = self.unmasked_A.device, dtype = self.unmasked_A.dtype));
            ith_tail_loss : torch.Tensor = torch.mean(torch.pow(ith_tail_mass.to(device = self.unmasked_A.device, dtype = self.unmasked_A.dtype), 2))
            loss_tail_list.append(ith_tail_loss);
            metrics[f"loss/tail/{str(ith_params)}"] = ith_tail_loss.detach();


        # Evaluate loss statistics (computed across times and parameters).
        weights         : torch.Tensor = torch.cat(weights_list, dim = 0);
        tail_masses     : torch.Tensor = torch.cat(tail_mass_list, dim = 0);
        n_engaged       : torch.Tensor = torch.cat(n_engaged_list, dim = 0);
        metrics.update(tensor_statistics(prefix = "expert/weights",         values = weights));
        metrics.update(tensor_statistics(prefix = "mass/tail",              values = tail_masses));
        metrics.update(tensor_statistics(prefix = "experts/num_engaged",    values = n_engaged));
        metrics.update(tensor_statistics(prefix = "experts/times_engaged",  values = times_engaged));
        metrics["experts/num_ever_engaged"] = torch.sum(times_engaged > 0).to(device = self.unmasked_A.device, dtype = self.unmasked_A.dtype).detach();

        # Coefficient loss is the sum of the selected norms of the matrix portions of each expert,
        # plus the selected norm of each enabled bias. This is a scalar global loss, so the
        # Trainer will not multiply it by n_param.
        A_coef  : torch.Tensor          = self.A;
        b_coef  : torch.Tensor | None   = self.b;
        ord     : int                   = 1 if self.coef_norm == 'l1' else 2;
        A_norms         : torch.Tensor          = torch.linalg.vector_norm(A_coef.reshape(self.n_experts, -1), ord = ord, dim = 1).sum();
        if b_coef is None:
            b_norms     : torch.Tensor          = torch.zeros((), dtype = A_coef.dtype, device = A_coef.device);
        else:
            b_norms     : torch.Tensor          = torch.linalg.vector_norm(b_coef.reshape(self.n_experts, -1), ord = ord, dim = 1).sum();
        loss_coef       : torch.Tensor          = A_norms + b_norms;

        # diversity loss is the squared coefficient of variation of expert loads. Use 
        # the population standard deviation so n_experts = 1 produces zero instead of NaN.
        eps             : float                 = torch.finfo(summed_weights.dtype).eps;
        mean_load       : torch.Tensor          = torch.mean(summed_weights);
        std_load        : torch.Tensor          = torch.std(summed_weights, unbiased = False);
        loss_diversity  : torch.Tensor          = torch.pow(std_load/(mean_load + eps), 2);

        # Store the average tail-mass loss for diagnostics/plotting; the weighted objective uses
        # the summed scalar returned under the `tail` key, while per-parameter values live in
        # metrics.
        loss_tail : torch.Tensor = torch.mean(torch.stack(loss_tail_list));
        self.last_tail_mass_loss = loss_tail.detach();
        self.last_tail_mass_loss_list = [loss.detach() for loss in loss_tail_list];

        # All done :)
        metrics["loss/diversity/total"] = loss_diversity.detach();
        metrics["loss/coef/total"]      = loss_coef.detach();
        metrics["loss/coef/A"]          = A_norms.detach();
        metrics["loss/coef/b"]          = b_norms.detach();
        loss_LD     : torch.Tensor      = torch.sum(torch.stack(loss_LD_list));
        loss_tail   : torch.Tensor      = torch.sum(torch.stack(loss_tail_list));
        metrics["loss/LD/total"]        = loss_LD.detach();
        metrics["loss/tail/total"]      = loss_tail.detach();

        losses_dict = {'LD' : loss_LD, 'coef' : loss_coef, 'diversity' : loss_diversity, 'tail' : loss_tail};

        return LD_Loss_Container(losses = losses_dict, weights = self.loss_weights, params = params, metrics = metrics);


    def RHS(    self,
                Z       : list[list[torch.Tensor | numpy.ndarray]],
                t_Grid  : list[numpy.ndarray | torch.Tensor],
                params  : numpy.ndarray,
                sample  : bool = False) -> list[torch.Tensor | numpy.ndarray]:
        r"""
        Evaluate the CABLE mixture-of-experts right-hand side.

        For each parameter value, theta, and time t, this evaluates

            z'(t) = \sum_m w_m(t, theta) [ A_m z(t) + b_m ]

        at the supplied latent states. The model is deterministic, so `sample` is accepted for
        interface compatibility but ignored.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        Z : list[list[torch.Tensor | numpy.ndarray]], len = n_param
            The i'th element is a one-element list whose first entry is a tensor/array of shape
            (n_t(i), n_z). 

        t_Grid : list[numpy.ndarray | torch.Tensor], len = n_param
            The i'th entry is a one-dimensional time grid with length n_t(i). CABLE is
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
            dimension n_z. It stores the CABLE RHS evaluated at the supplied states/times.
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
        LOGGER.debug("Computing CABLE RHS with %d parameter combinations" % n_param);
        for i in range(n_param):
            ith_Z      : list[numpy.ndarray | torch.Tensor]  = Z[i];
            ith_t_Grid : numpy.ndarray | torch.Tensor        = t_Grid[i];

            assert isinstance(ith_Z, list) and len(ith_Z) == 1;
            ith_Z0 : numpy.ndarray | torch.Tensor = ith_Z[0];
            assert isinstance(ith_Z0, (numpy.ndarray, torch.Tensor));
            assert len(ith_Z0.shape) == 2;
            assert ith_Z0.shape[-1] == self.n_z;
            assert len(ith_t_Grid.shape) == 1;
            assert ith_Z0.shape[0] == ith_t_Grid.shape[0];

            if isinstance(ith_Z0, numpy.ndarray):
                RH_Sides.append(self._evaluate_numpy_rhs(ith_Z0, ith_t_Grid, params[i, :]));
            else:
                RH_Sides.append(self._evaluate_torch_rhs(ith_Z0, ith_t_Grid, params[i, :]));

        # All done!
        return RH_Sides;


    def simulate(   self,
                    IC      : list[list[numpy.ndarray | torch.Tensor]],
                    t_Grid  : list[numpy.ndarray      | torch.Tensor],
                    params  : numpy.ndarray,
                    sample  : bool = False) -> list[list[numpy.ndarray | torch.Tensor]]:
        r"""
        Time-integrate the deterministic CABLE latent dynamics.

        The gate is evaluated at the RK stage time and parameter value, so the integrated system is
        generally non-autonomous even though each expert is affine in z. The model is
        deterministic, so `sample` is accepted for interface compatibility but ignored.

        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        IC : list[list[numpy.ndarray | torch.Tensor]], len = n_param
            Initial latent states for each parameter value. CABLE has one IC component, so
            `IC[i][0]` must have shape (n_z).

        t_Grid : list[numpy.ndarray | torch.Tensor], len = n_param
            One-dimensional time grids for simulation.

        params : numpy.ndarray, shape = (n_param, n_p)
            The i'th row holds the i'th combination of parameter values.
        
        sample : bool 
            Ignored. Present only to match the LatentDynamics interface.
            

        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Z : list[list[numpy.ndarray | torch.Tensor]]
            Simulated latent trajectories. Z[i][0] has shape (n_t(i), n_z).
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
        LOGGER.debug("Simulating CABLE with %d parameter combinations" % n_param);
        for i in range(n_param):
            ith_IC     : list[numpy.ndarray | torch.Tensor]  = IC[i];
            ith_t_Grid : numpy.ndarray | torch.Tensor        = t_Grid[i];
            ith_params : numpy.ndarray                       = params[i, :];

            assert isinstance(ith_IC, list) and len(ith_IC) == 1;
            if(isinstance(ith_t_Grid, torch.Tensor)):
                ith_t_Grid = ith_t_Grid.detach().cpu().numpy();
            assert len(ith_t_Grid.shape) == 1;
            t0      : float = ith_t_Grid[0];
            t_span  : float = ith_t_Grid[-1] - ith_t_Grid[0];
            ith_Z0 : numpy.ndarray | torch.Tensor = ith_IC[0];
            assert len(ith_Z0.shape) == 1 and ith_Z0.shape[0] == self.n_z;

            # Define the right-hand side in either NumPy or PyTorch. The solver backend follows the
            # initial-condition backend; this preserves differentiability for tensor rollouts.
            if isinstance(ith_Z0, numpy.ndarray):
                def f(t : float, z : numpy.ndarray) -> numpy.ndarray:
                    t_eval  : numpy.ndarray = numpy.asarray([t], dtype = ith_t_Grid.dtype);
                    with torch.no_grad():
                        weights         : torch.Tensor = self._weights_for_t_grid(t_eval, ith_params, t0 = t0, t_span = t_span);
                        if z.dtype == numpy.dtype(numpy.float64):
                            dtype = torch.float64;
                        else:
                            dtype = torch.float32;
                        A_bar, b_bar                  = self._effective_coefficients(weights, torch.device("cpu"), dtype);
                        A_np : numpy.ndarray          = A_bar[0].detach().cpu().numpy().astype(z.dtype, copy = False);
                        b_np : numpy.ndarray          = b_bar[0].detach().cpu().numpy().astype(z.dtype, copy = False);
                    return b_np + numpy.matmul(z, A_np.T);
            else:
                def f(t : float, z : torch.Tensor) -> torch.Tensor:
                    param : torch.Tensor = next(self.w.parameters());
                    gate_device = param.device
                    gate_dtype  = param.dtype;

                    t_eval  : torch.Tensor = torch.tensor([t], dtype = gate_dtype, device = gate_device);
                    weights : torch.Tensor = self._weights_for_t_grid(t_eval, ith_params, t0 = t0, t_span = t_span);
                    A_bar, b_bar           = self._effective_coefficients(weights, z.device, z.dtype);
                    return b_bar[0] + torch.matmul(z, A_bar[0].T);

            # Solve the ODE for this single latent initial state.
            ith_Z = RK4(f = f, y0 = ith_Z0, t_Grid = ith_t_Grid);

            # Add this parameter's trajectory to the output list.
            Z.append([ith_Z]);

        # All done!
        return Z;

 
    # ---------------------------------------------------------------------------------------------
    # Serialization
    # ---------------------------------------------------------------------------------------------


    def export(self) -> dict:
        r"""Export CABLE metadata, expert tensors, and gate-network parameters."""

        param_dict = {'n_z'             : self.n_z,
                      'n_IC'            : self.n_IC,
                      'n_p'             : self.n_p,
                      'config'          : self.config.model_dump(mode = "python", by_alias = True),
                      'Uniform_t_Grid'  : self.Uniform_t_Grid,
                      'unmasked_A'      : self.unmasked_A.detach().cpu().clone(),
                      'unmasked_b'      : None if self.unmasked_b is None else self.unmasked_b.detach().cpu().clone(),
                      'A_mask'          : self.A_mask.detach().cpu().clone(),
                      'b_mask'          : None if self.b_mask is None else self.b_mask.detach().cpu().clone(),
                      'w_state_dict'    : {key: value.detach().cpu().clone() for key, value in self.w.state_dict().items()}};
        return param_dict;


    def load(self, dict_ : dict) -> None:
        r"""Load CABLE metadata, expert tensors, and gate-network parameters."""

        assert(self.n_z             == dict_['n_z']);
        assert(self.n_IC            == dict_['n_IC']);
        assert(self.n_p             == dict_['n_p']);
        assert(self.Uniform_t_Grid  == dict_['Uniform_t_Grid']);

        # Fetch the unmasked A tensor.
        unmasked_A : torch.Tensor = dict_['unmasked_A'];
        assert isinstance(unmasked_A, torch.Tensor) and unmasked_A.shape == (self.n_experts, self.n_z, self.n_z);
        self.unmasked_A = unmasked_A.detach().clone().requires_grad_(self.trainable);

        # Do the same for unmasked b.
        if self.use_biases:
            unmasked_b = dict_['unmasked_b'];
            assert isinstance(unmasked_b, torch.Tensor) and unmasked_b.shape == (self.n_experts, 1, self.n_z);
            self.unmasked_b = unmasked_b.detach().clone().requires_grad_(self.trainable);
        else:
            self.unmasked_b = None;

        # Now fetch the A mask
        A_mask = dict_.get('A_mask', torch.ones_like(self.unmasked_A));
        assert isinstance(A_mask, torch.Tensor) and A_mask.shape == (self.n_experts, self.n_z, self.n_z);
        self.A_mask = A_mask.detach().clone().to(device = self.unmasked_A.device, dtype = self.unmasked_A.dtype);

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

    def _weights_for_t_grid(
            self,
            t_Grid  : numpy.ndarray | torch.Tensor,
            params  : numpy.ndarray,
            *,
            t0      : float, 
            t_span  : float) -> torch.Tensor:
        r"""
        Evaluate gate weights on one time grid and one parameter value.

        The returned tensor has shape (n_t, n_experts) and lives on the same device/dtype as the
        gate network. Callers can cast it to the latent-state backend before evaluating experts.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        t_Grid : numpy.ndarray or torch.Tensor, shape = (n_t)
            One-dimensional time grid for a single parameter value. NumPy inputs may have any
            floating dtype. Torch inputs may live on any device. The values are cast to the gate
            network's dtype/device before forming gate inputs. 

        params : numpy.ndarray, shape = (n_p)
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

        assert t_span > 0, "t_Grid has no range; t_Grid[0] = %f, t_Grid[-1] = %f" % (t_Grid[0], t_Grid[-1]);

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


        # Broadcast n_t copies of param_tensor to build inputs for the gate network.
        param_tensor : torch.Tensor = torch.tensor(params, dtype = gate_dtype, device = gate_device).reshape(1, self.n_p);
        param_tensor = param_tensor.expand(tau_tensor.shape[0], self.n_p);

        # Build the gate network inputs
        w_inputs : torch.Tensor = torch.cat([tau_tensor.reshape(-1, 1), param_tensor], dim = 1);

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
            dtype   : torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Collapse expert coefficients into one affine system per time sample.

        Given weights w[t, m], this forms

            A_bar[t] = sum_m w[t, m] A[m],
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

        A_bar : torch.Tensor, shape = (n_t, n_z, n_z)
            Time-dependent effective linear operators, on `device` with dtype `dtype`.

        b_bar : torch.Tensor, shape = (n_t, n_z)
            Time-dependent effective affine shifts, on `device` with dtype `dtype`.
        """

        weights = weights.to(device = device, dtype = dtype);
        A   : torch.Tensor        = self.A.to(device = device, dtype = dtype);
        b   : torch.Tensor | None = self.b;
        if b is None:
            b   = torch.zeros((self.n_experts, self.n_z), dtype = dtype, device = device);
        else:
            b   = b.to(device = device, dtype = dtype).reshape(self.n_experts, self.n_z);

        A_flat  : torch.Tensor = A.reshape(self.n_experts, self.n_z*self.n_z);
        A_bar   : torch.Tensor = (weights @ A_flat).reshape(weights.shape[0], self.n_z, self.n_z);
        b_bar   : torch.Tensor = weights @ b;
        return A_bar, b_bar;


    @property
    def A(self) -> torch.Tensor:
        r"""
        Return the effective expert matrices after applying the hard mask.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        A : torch.Tensor, shape = (n_experts, n_z, n_z)
            Expert matrices with masked entries set to zero.
        """

        if self.use_mask:
            return self.unmasked_A * self.A_mask.to(device = self.unmasked_A.device, dtype = self.unmasked_A.dtype);
        return self.unmasked_A;


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
    def _update_mask(self) -> None:
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
        self.A_mask = self.A_mask.to(device = self.unmasked_A.device, dtype = self.unmasked_A.dtype);

        # Determine which components of A are bigger than the threshold.
        A_keep      : torch.Tensor = (self.A.abs() >= self.mask_threshold).to(dtype = self.unmasked_A.dtype);

        # Update the mask; note that any previously masked components remain masked.
        self.A_mask = (self.A_mask * A_keep).contiguous();

        # Update A.
        self.unmasked_A.data.mul_(self.A_mask);
        n_active : int = int(self.A_mask.sum().item());
        n_total  : int = int(self.A_mask.numel());

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


    def _evaluate_torch_rhs(
            self,
            Z       : torch.Tensor,
            t_Grid  : numpy.ndarray | torch.Tensor,
            params  : numpy.ndarray) -> torch.Tensor:
        r"""
        Evaluate CABLE's right-hand side with torch tensors.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        Z : torch.Tensor, shape = (n_t, n_z)
            Latent states at which to evaluate the RHS. The output preserves this tensor's dtype
            and device.

        t_Grid : numpy.ndarray or torch.Tensor, shape = (n_t)
            One-dimensional time grid corresponding to the first dimension of `Z`. Values are used
            by the gate network and are cast to the gate-network dtype/device internally.

        params : numpy.ndarray, shape = (n_p)
            Parameter vector associated with `Z`. Values are used by the gate network.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        RHS : torch.Tensor, shape = (n_t, n_z)
            CABLE right-hand-side values, with dtype/device matching `Z`.
        """

        weights         : torch.Tensor = self._weights_for_t_grid(t_Grid, params, t0 = t_Grid[0], t_span = t_Grid[-1] - t_Grid[0]);
        return self._evaluate_torch_rhs_from_weights(Z, weights);


    def _evaluate_torch_rhs_from_weights(
            self,
            Z       : torch.Tensor,
            weights : torch.Tensor) -> torch.Tensor:
        r"""
        Evaluate CABLE's right-hand side with precomputed torch gate weights.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        Z : torch.Tensor, shape = (n_t, n_z)
            Latent states at which to evaluate the RHS. The first dimension must match
            `weights.shape[0]`. The returned tensor matches this dtype/device.

        weights : torch.Tensor, shape = (n_t, n_experts)
            Precomputed expert weights. This tensor may have the gate-network dtype/device; it is
            cast to `Z.device` and `Z.dtype` before combining experts.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        RHS : torch.Tensor, shape = (n_t, n_z)
            CABLE right-hand-side values, RHS[t] = A_bar[t] Z[t] + b_bar[t].
        """

        A_bar, b_bar                    = self._effective_coefficients(weights, Z.device, Z.dtype);

        assert len(Z.shape) == 2 and Z.shape[1] == self.n_z;
        assert Z.shape[0] == weights.shape[0];
        # For each time t, compute A_bar[t] @ Z[t]. Shapes:
        #   A_bar       : (n_t, n_z, n_z)
        #   Z[..., None]: (n_t, n_z, 1)
        # torch.bmm returns (n_t, n_z, 1), then squeeze gives (n_t, n_z).
        return torch.bmm(A_bar, Z.unsqueeze(-1)).squeeze(-1) + b_bar;


    def _evaluate_numpy_rhs(
            self,
            Z       : numpy.ndarray,
            t_Grid  : numpy.ndarray | torch.Tensor,
            params  : numpy.ndarray) -> numpy.ndarray:
        r"""
        Evaluate CABLE's right-hand side with NumPy arrays.

        This helper evaluates the torch gate and expert tensors without tracking gradients, moves
        the effective coefficients to CPU, and returns a NumPy array.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        Z : numpy.ndarray, shape = (n_t, n_z)
            Latent states at which to evaluate the RHS. The returned array preserves this dtype and
            shape.

        t_Grid : numpy.ndarray or torch.Tensor, shape = (n_t)
            One-dimensional time grid corresponding to the first dimension of `Z`. Values are used
            by the gate network.

        params : numpy.ndarray, shape = (n_p)
            Parameter vector associated with `Z`. Values are used by the gate network.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        RHS : numpy.ndarray, shape = (n_t, n_z)
            CABLE right-hand-side values as a NumPy array with dtype matching `Z.dtype`.
        """

        with torch.no_grad():
            weights         : torch.Tensor = self._weights_for_t_grid(t_Grid, params, t0 = t_Grid[0], t_span = t_Grid[-1] - t_Grid[0]);
            if Z.dtype == numpy.dtype(numpy.float64):
                dtype = torch.float64;
            else:
                dtype = torch.float32;
            A_bar, b_bar                   = self._effective_coefficients(weights, torch.device("cpu"), dtype);
            A_np : numpy.ndarray           = A_bar.detach().cpu().numpy().astype(Z.dtype, copy = False);
            b_np : numpy.ndarray           = b_bar.detach().cpu().numpy().astype(Z.dtype, copy = False);

        assert len(Z.shape) == 2 and Z.shape[1] == self.n_z;
        # For each time t, compute A_np[t] @ Z[t]. Shapes:
        #   A_np       : (n_t, n_z, n_z)
        #   Z[...,None]: (n_t, n_z, 1)
        # numpy.matmul returns (n_t, n_z, 1), then squeeze gives (n_t, n_z).
        Az : numpy.ndarray = numpy.matmul(A_np, Z[..., None]).squeeze(-1);
        return Az + b_np;
