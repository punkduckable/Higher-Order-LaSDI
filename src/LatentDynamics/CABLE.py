# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

import  logging;

import  numpy;
import  torch;

from    LatentDynamics.LatentDynamics   import  LatentDynamics;
from    Schemas                         import  CABLELatentDynamicsConfig, CABLELatentDynamicsSettings;
from    EncoderDecoder                  import  MultiLayerPerceptron;
from    Utilities.FiniteDifference      import  Derivative1_Order4, Derivative1_Order2_NonUniform;
from    Utilities.FirstOrderSolvers     import  RK4;

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

        CABLE is a deterministic mixture-of-affine-experts latent ODE. For a parameter value
        \theta and time t, it evolves the latent state according to

            z'(t) = \sum_{m = 1}^{N} w_m(t, \theta) [ A_m z(t) + b_m ],

        where N is the number of experts, each A_m is an n_z x n_z matrix, each b_m is an
        n_z-vector, and w_m(t, \theta) is the gate weight assigned to the m'th expert. The gate is
        a neural network

            w : \mathbb{R}^{1 + n_p} \to \mathbb{R}^{N}

        whose components below epsilon are set to zero. Further, we train the network such that 
        for any individual step, almost all of the mass is concentrated in <= n_active experts, 
        giving loosely sparse weights.

        The trainable latent-dynamics state consists of the expert matrices, expert biases, and
        gate-network parameters. Unlike interpolatable SINDy-type models, CABLE owns one global set
        of parameters rather than one coefficient dictionary per training parameter.


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

        assert isinstance(config, CABLELatentDynamicsConfig), "config must be a CABLELatentDynamicsConfig, got %s" % str(type(config));

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
        sub : CABLELatentDynamicsSettings = config.cable;
        self.n_experts      : int       = sub.n_experts;
        self.n_active       : int       = sub.n_active;
        self.hidden_widths  : list[int] = sub.hidden_widths;
        self.activations    : list[str] = sub.activations;

        # Initialize the gate network.
        widths      : list[int] = [n_p + 1] + self.hidden_widths + [self.n_experts];
        self.w                  = MultiLayerPerceptron(widths = widths, activations = self.activations);

        # Randomly initialize the experts. These are leaf tensors because the Trainer passes them
        # directly to the optimizer through trainable_tensors().
        self.A : torch.Tensor = torch.rand((self.n_experts, self.n_z, self.n_z), dtype = torch.float32).requires_grad_(self.trainable);
        self.b : torch.Tensor = torch.zeros((self.n_experts, 1, self.n_z), dtype = torch.float32).requires_grad_(self.trainable);
        for param in self.w.parameters():
            param.requires_grad_(self.trainable);

        # Setup the loss functions used by compute_losses.
        self.MSE = torch.nn.MSELoss(reduction = 'mean');
        self.MAE = torch.nn.L1Loss(reduction = 'mean');

        # Diagnostic-only until the Trainer supports arbitrary named LD losses.
        self.last_tail_mass_loss : torch.Tensor | None = None;
        self.last_tail_mass_loss_list : list[torch.Tensor] | None = None;
        return;


    # ---------------------------------------------------------------------------------------------
    # trainable_tensors, move_trainable_tensors_to_device, and initialize_coefficients
    # ---------------------------------------------------------------------------------------------


    def trainable_tensors(self) -> list[torch.Tensor]:
        r"""
        Return CABLE-owned tensors that should be passed to torch optimizers.

        These are the expert matrices, expert biases, and gate-network parameters. The list is
        empty when the latent dynamics are frozen.
        """

        if self.trainable == False:
            return [];

        tensors : list[torch.Tensor] = [self.A, self.b];
        for param in self.w.parameters():
            tensors.append(param);
        return tensors;


    def move_trainable_tensors_to_device(self, device : torch.device | str) -> None:
        r"""
        Move CABLE-owned trainable tensor state to a device.


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
        self.A = self.A.detach().to(device = device).requires_grad_(self.trainable);
        self.b = self.b.detach().to(device = device).requires_grad_(self.trainable);
        self.w = self.w.to(device = device);
        for param in self.w.parameters():
            param.requires_grad_(self.trainable);
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

        # Move A, b, and w to specified device.
        self.move_trainable_tensors_to_device(device);
        return None;
    
    # ---------------------------------------------------------------------------------------------
    # Compute Losses, RHS, and Simulate
    # ---------------------------------------------------------------------------------------------


    def compute_losses(  
        self,  
        Latent_States   : list[list[torch.Tensor]], 
        loss_type       : str,
        t_Grid          : list[torch.Tensor], 
        params          : numpy.ndarray | None = None
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
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

            mean_t (1 - sum_{m in top_k(t,theta_i)} q_m(t,theta_i))^2,

        where q is the dense softmax over all experts, and k = self.n_active. This quantifies how 
        much probability mass is outside of the top n_active experts. 


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        Latent_States : list[list[torch.Tensor]], len = n_param
            Encoded latent trajectories. The i'th entry contains one tensor of shape (n_t(i), n_z).

        loss_type : str
            The latent-dynamics residual loss. Must be either "MSE" or "MAE".

        t_Grid : list[torch.Tensor], len = n_param
            Time grids corresponding to the latent trajectories.

        params : numpy.ndarray, shape = (n_param, n_p)
            Parameter rows used as inputs to the gate network.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        loss_LD_list : list[torch.Tensor], len = n_param
            Per-parameter residual losses between finite-difference derivatives and CABLE RHS
            values.

        loss_coef_list : list[torch.Tensor], len = n_param
            Per-parameter copies of the scaled global expert-size penalty. The scaling prevents the
            Trainer from multiplying this global penalty by n_param when it sums the list.

        loss_stab_list : list[torch.Tensor], len = n_param
            Per-parameter copies of the scaled squared coefficient of variation of aggregate
            expert loads. The scaling has the same purpose as for `loss_coef_list`.
        """

        # Checks.
        assert params is not None, "CABLE.compute_losses requires params for the gate network";
        assert isinstance(params, numpy.ndarray) and len(params.shape) == 2;
        assert params.shape[1] == self.n_p;
        assert isinstance(t_Grid, list);
        assert isinstance(Latent_States, list);
        assert len(Latent_States) == len(t_Grid) == params.shape[0];
        assert len(t_Grid) > 0;
        assert loss_type in ["MSE", "MAE"];

        # Prepare lists for per-parameter losses. The Trainer is responsible for applying weights
        # and summing these scalar losses into the total objective.
        loss_LD_list          : list[torch.Tensor] = [];
        summed_weights        : torch.Tensor       = torch.zeros((self.n_experts), dtype = self.A.dtype, device = self.A.device);
        tail_mass_loss_list   : list[torch.Tensor] = [];

        n_param : int = len(t_Grid);
        for i in range(n_param):
            # Fetch this parameter's latent trajectory and time grid.
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
            ith_weights       : torch.Tensor = self._weights_for_t_grid(ith_t_Grid, params[i, :]);
            ith_RHS           : torch.Tensor = self._evaluate_torch_rhs_from_weights(ith_Z, ith_weights);

            # Compute the LD loss.
            if(loss_type == "MSE"):
                loss_LD = self.MSE(dZdt, ith_RHS);
            else:
                loss_LD = self.MAE(dZdt, ith_RHS);
            loss_LD_list.append(loss_LD);

            # Accumulate expert loads across all parameter values and times. This is a 
            # deterministic analogue of MoE importance/load diversity: it encourages all
            # experts to be useful somewhere without forcing all experts to be active at 
            # every step.
            summed_weights = summed_weights + torch.sum(ith_weights.to(device = self.A.device, dtype = self.A.dtype), dim = 0);

            # Tail-mass penalty: compute how much softmax mass lies outside the top-n_active 
            # logits. If this is small, most of the probability mass is in the top n_active 
            # experts, as intended. 
            if self.n_active >= self.n_experts:
                ith_tail_mass : torch.Tensor = torch.zeros((n_t_i), dtype = ith_weights.dtype, device = ith_weights.device);
            else:
                ith_topk_idx             : torch.Tensor = torch.topk(ith_weights, self.n_active, dim = 1, sorted = False).indices;
                ith_topk_dense_mass      : torch.Tensor = torch.sum(ith_weights.gather(1, ith_topk_idx), dim = 1);
                ith_tail_mass            : torch.Tensor = 1.0 - ith_topk_dense_mass;
            tail_mass_loss_list.append(torch.mean(torch.pow(ith_tail_mass.to(device = self.A.device, dtype = self.A.dtype), 2)));

        # Coefficient loss is the sum of the Frobenius norms of the matrix portions of each expert,
        # plus the L2 norm of each bias. The loss is global, so we divide each list entry by n_param.
        A_norms         : torch.Tensor          = torch.linalg.vector_norm(self.A.reshape(self.n_experts, -1), dim = 1).sum();
        b_norms         : torch.Tensor          = torch.linalg.vector_norm(self.b.reshape(self.n_experts, -1), dim = 1).sum();
        loss_coef       : torch.Tensor          = (A_norms + b_norms) / float(n_param);
        loss_coef_list  : list[torch.Tensor]    = [loss_coef]*n_param;

        # Stability/diversity loss is the squared coefficient of variation of expert loads. Use 
        # the population standard deviation so n_experts = 1 produces zero instead of NaN.
        eps             : float                 = torch.finfo(summed_weights.dtype).eps;
        mean_load       : torch.Tensor          = torch.mean(summed_weights);
        std_load        : torch.Tensor          = torch.std(summed_weights, unbiased = False);
        loss_stab       : torch.Tensor          = torch.pow(std_load/(mean_load + eps), 2) / float(n_param);
        loss_stab_list  : list[torch.Tensor]    = [loss_stab]*n_param;

        # Compute and store the tail-mass loss for diagnostics. This is deliberately not
        # returned yet; the next loss-API change can expose it as its own named loss.
        loss_tail : torch.Tensor = torch.mean(torch.stack(tail_mass_loss_list));
        self.last_tail_mass_loss = loss_tail.detach();
        self.last_tail_mass_loss_list = [loss.detach() for loss in tail_mass_loss_list];

        # All done :) 
        return loss_LD_list, loss_coef_list, loss_stab_list;


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
            (n_t(i), n_z). Batched RHS inputs are intentionally not supported here; `simulate`
            handles batched initial conditions separately.

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
            Initial latent states for each parameter value. CABLE has one IC component, so each
            entry is a one-element list whose tensor/array has shape (n_initial_conditions, n_z).

        t_Grid : list[numpy.ndarray | torch.Tensor], len = n_param
            Time grids for simulation. A one-dimensional grid is shared by all initial conditions
            for that parameter; a two-dimensional grid supplies one row per initial condition.

        params : numpy.ndarray, shape = (n_param, n_p)
            The i'th row holds the i'th combination of parameter values.
        
        sample : bool 
            Ignored. Present only to match the LatentDynamics interface.
            

        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Z : list[list[numpy.ndarray | torch.Tensor]]
            Simulated latent trajectories. Z[i][0] has shape (n_t(i), n_initial_conditions, n_z).
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
            assert len(ith_t_Grid.shape) == 1 or len(ith_t_Grid.shape) == 2;
            if(isinstance(ith_t_Grid, torch.Tensor)):
                ith_t_Grid = ith_t_Grid.detach().cpu().numpy();
            Same_t_Grid : bool = (len(ith_t_Grid.shape) == 1);
            ith_Z0 : numpy.ndarray | torch.Tensor = ith_IC[0];
            n_i    : int                          = ith_Z0.shape[0];
            assert len(ith_Z0.shape) == 2 and ith_Z0.shape[1] == self.n_z;
            if(Same_t_Grid == False):
                assert ith_t_Grid.shape[0] == n_i;

            # Define the right-hand side in either NumPy or PyTorch. The solver backend follows the
            # initial-condition backend; this preserves differentiability for tensor rollouts.
            if isinstance(ith_Z0, numpy.ndarray):
                def f(t : float, z : numpy.ndarray) -> numpy.ndarray:
                    t_eval  : numpy.ndarray = numpy.asarray([t], dtype = ith_t_Grid.dtype);
                    with torch.no_grad():
                        weights         : torch.Tensor = self._weights_for_t_grid(t_eval, ith_params);
                        if z.dtype == numpy.dtype(numpy.float64):
                            dtype = torch.float64;
                        else:
                            dtype = torch.float32;
                        A_bar, b_bar                  = self._effective_coefficients(weights, torch.device("cpu"), dtype);
                        A_np : numpy.ndarray          = A_bar[0].detach().cpu().numpy().astype(z.dtype, copy = False);
                        b_np : numpy.ndarray          = b_bar[0].detach().cpu().numpy().astype(z.dtype, copy = False).reshape(1, -1);
                    return b_np + numpy.matmul(z, A_np.T);
            else:
                def f(t : float, z : torch.Tensor) -> torch.Tensor:
                    param : torch.Tensor = next(self.w.parameters());
                    gate_device = param.device
                    gate_dtype  = param.dtype;

                    t_eval  : torch.Tensor = torch.tensor([t], dtype = gate_dtype, device = gate_device);
                    weights : torch.Tensor = self._weights_for_t_grid(t_eval, ith_params);
                    A_bar, b_bar           = self._effective_coefficients(weights, z.device, z.dtype);
                    return b_bar[0].reshape(1, -1) + torch.matmul(z, A_bar[0].T);

            # Solve the ODE. If all ICs share the same time grid we integrate them as a batch;
            # otherwise, integrate each initial condition separately and concatenate the results.
            if(Same_t_Grid == True):
                ith_Z = RK4(f = f, y0 = ith_Z0, t_Grid = ith_t_Grid);
            else:
                Z_list : list[torch.Tensor | numpy.ndarray] = [];
                for j in range(n_i):
                    Z_j = RK4(f = f, y0 = ith_Z0[j, :].reshape(1, -1), t_Grid = ith_t_Grid[j, :]);
                    Z_list.append(Z_j);
                if(isinstance(ith_Z0, numpy.ndarray)):
                    ith_Z = numpy.concatenate(Z_list, axis = 1);
                else:
                    ith_Z = torch.cat(Z_list, dim = 1);

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
                      'A'               : self.A.detach().cpu().clone(),
                      'b'               : self.b.detach().cpu().clone(),
                      'w_state_dict'    : {key: value.detach().cpu().clone() for key, value in self.w.state_dict().items()}};
        return param_dict;


    def load(self, dict_ : dict) -> None:
        r"""Load CABLE metadata, expert tensors, and gate-network parameters."""

        assert(self.n_z             == dict_['n_z']);
        assert(self.n_IC            == dict_['n_IC']);
        assert(self.n_p             == dict_['n_p']);
        assert(self.Uniform_t_Grid  == dict_['Uniform_t_Grid']);

        A = dict_['A'];
        b = dict_['b'];
        assert isinstance(A, torch.Tensor) and A.shape == (self.n_experts, self.n_z, self.n_z);
        assert isinstance(b, torch.Tensor) and b.shape == (self.n_experts, 1, self.n_z);
        self.A = A.detach().clone().requires_grad_(self.trainable);
        self.b = b.detach().clone().requires_grad_(self.trainable);
        self.w.load_state_dict(dict_['w_state_dict']);
        for param in self.w.parameters():
            param.requires_grad_(self.trainable);
        return;


    # ---------------------------------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------------------------------

    def _weights_for_t_grid(
            self,
            t_Grid  : numpy.ndarray | torch.Tensor,
            params  : numpy.ndarray) -> torch.Tensor:
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


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        weights : torch.Tensor, shape = (n_t, n_experts)
           expert weights evaluated at all (t, params) pairs. The dtype and device match the 
           gate-network parameters. Each row sums to one.
        """

        # Setup 
        param : torch.Tensor = next(self.w.parameters());
        gate_device = param.device
        gate_dtype  = param.dtype;

        # -----------------------------------------------------------------------------------------
        # Build gate network inputs

        # Map t_Grid to a tensor.
        # The gate is a torch.nn.Module; its inputs must be tensors.
        if isinstance(t_Grid, numpy.ndarray):
            t_tensor : torch.Tensor = torch.tensor(t_Grid, dtype = gate_dtype, device = gate_device);
        else:
            t_tensor = t_Grid.to(device = gate_device, dtype = gate_dtype);
        assert len(t_tensor.shape) == 1;

        # Broadcast n_t copies of param_tensor to build inputs for the gate network.
        param_tensor : torch.Tensor = torch.tensor(params, dtype = gate_dtype, device = gate_device).reshape(1, self.n_p);
        param_tensor = param_tensor.expand(t_tensor.shape[0], self.n_p);

        # Build the gate network inputs
        w_inputs : torch.Tensor = torch.cat([t_tensor.reshape(-1, 1), param_tensor], dim = 1);

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
        A       = self.A.to(device = device, dtype = dtype);
        b       = self.b.to(device = device, dtype = dtype).reshape(self.n_experts, self.n_z);

        A_flat  : torch.Tensor = A.reshape(self.n_experts, self.n_z*self.n_z);
        A_bar   : torch.Tensor = (weights @ A_flat).reshape(weights.shape[0], self.n_z, self.n_z);
        b_bar   : torch.Tensor = weights @ b;
        return A_bar, b_bar;


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

        weights         : torch.Tensor = self._weights_for_t_grid(t_Grid, params);
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
            weights         : torch.Tensor = self._weights_for_t_grid(t_Grid, params);
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
