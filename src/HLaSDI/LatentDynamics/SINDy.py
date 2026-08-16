# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

import  logging;

import  numpy;
import  torch;

from    HLaSDI.LatentDynamics.Interpolatable   import  InterpolatableLatentDynamics;
from    HLaSDI.LatentDynamics.LatentDynamics   import  LD_Loss_Container;
from    HLaSDI.Schemas                         import  SINDyLatentDynamicsConfig;
from    HLaSDI.Utilities.FiniteDifference      import  Derivative1_Order4, Derivative1_Order2_NonUniform;
from    HLaSDI.Utilities.FirstOrderSolvers     import  RK4;

LOGGER  : logging.Logger    = logging.getLogger(__name__);



# -------------------------------------------------------------------------------------------------
# SINDy class
# -------------------------------------------------------------------------------------------------

class SINDy(InterpolatableLatentDynamics):
    def __init__(   self, 
                    n_z             : int,
                    Uniform_t_Grid  : bool,
                    n_p             : int,
                    config          : SINDyLatentDynamicsConfig) -> None:
        r"""
        Initializes a SINDy latent-dynamics object.

        This model assumes a first-order autonomous latent ODE with an affine right-hand side

            z'(t) = A z(t) + b,

        where A is an n_z x n_z matrix and b is an n_z-vector. Historically this class stored the
        same information as one flattened coefficient vector whose underlying matrix was
        [b; A^T]. The new coefficient ownership model stores coefficients in their native form
        under each training parameter in `self.train_coefs`:

            self.train_coefs[param_key] = {"A": A, "b": b}.


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

        config : dict
            The latent-dynamics configuration dictionary. It must three keys: `type`, `trainable`,
            and `sindy`. It must have `config["type"] == "sindy"` and `config["sindy"]` should be a 
            dictionary housing sub-class specific settings. The required `lstsq_reg` entry controls
            ridge regularization used by `initialize_coefficients(...)` when initializing 
            coefficients from encoded trajectories.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Nothing!
        """

        assert isinstance(config, SINDyLatentDynamicsConfig), "config must be a SINDyLatentDynamicsConfig, got %s" % str(type(config));

        # Run the base class initializer. Note that this initializes self.train_coefs.
        InterpolatableLatentDynamics.__init__(
            self,
            n_z            = n_z, 
            n_coefs        = n_z*(n_z + 1), 
            n_IC           = 1, 
            n_p            = n_p,
            Uniform_t_Grid = Uniform_t_Grid,
            trainable      = config.trainable,
            config         = config);

        # Set up class-specific variables.
        self.lstsq_reg : float = config.sindy.lstsq_reg;
        LOGGER.info("Initializing a SINDY object with n_z = %d, Uniform_t_Grid = %s, lstsq_reg = %s" % (self.n_z, str(self.Uniform_t_Grid), str(self.lstsq_reg)));

        # Setup the loss functions used by compute_losses.
        self.MSE = torch.nn.MSELoss(reduction = 'mean');
        self.MAE = torch.nn.L1Loss(reduction = 'mean');
        return;



    def _native_from_matrix(self, coefs : torch.Tensor) -> dict[str, torch.Tensor]:
        r"""
        Convert the least-squares coefficient matrix into native trainable tensors.

        The least-squares solve naturally returns the legacy matrix with shape (n_z + 1, n_z),
        where the first row is the constant forcing and the remaining rows are A^T. We immediately
        convert that representation into {"A", "b"} and make both tensors detached trainable leaves
        so the optimizer can update them directly through `self.train_coefs`.
        """

        assert coefs.shape == (self.n_z + 1, self.n_z), "SINDy coefficient matrix shape mismatch";
        # Old flattened matrix convention was [b; A^T]. Native convention is z' = A z + b.
        b : torch.Tensor = coefs[0, :].detach().clone().requires_grad_(True);
        A : torch.Tensor = coefs[1:, :].T.detach().clone().requires_grad_(True);
        return {"A": A, "b": b};



    def trainable_tensors(self) -> list[torch.Tensor]:
        r"""
        Return the actual coefficient tensors that should be passed to torch optimizers.

        These are not copies. They are the same tensors stored in `self.train_coefs`, so optimizer
        updates modify the LD-owned coefficient dictionaries used by compute_losses/simulate.
        """

        if self.trainable == False:
            return [];

        tensors : list[torch.Tensor] = [];
        for coef_dict in self.train_coefs.values():
            tensors.extend([coef_dict["A"], coef_dict["b"]]);
        return tensors;



    def initialize_coefficients(
            self,
            Latent_States   : list[list[torch.Tensor]],
            t_Grid          : list[torch.Tensor],
            device          : torch.device,
            params          : numpy.ndarray) -> None:
        r"""
        Fit and store SINDy coefficients for one or more training parameters.

        This method is used for coefficient initialization, especially when the sampler adds a new
        training point. Unlike the previous interface, it does not return a flattened coefficient
        matrix. Instead, each fitted coefficient set is converted to native form and stored in
        `self.train_coefs` under the exact tuple key associated with the corresponding row of
        `params`.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        Latent_States : list[list[torch.Tensor]], len = n_param
            The i'th entry contains one tensor with shape (n_t(i), n_z), holding the encoded latent
            trajectory for the i'th parameter.

        t_Grid : list[torch.Tensor], len = n_param
            Time grid for each latent trajectory.

        device : torch.device
            The device where we want to store the new coefficients.
            
        params : numpy.ndarray, shape = (n_param, n_p)
            Parameter rows used as keys in `self.train_coefs`. 
            
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        None. 
        """

        # Checks.
        assert params is not None, "SINDy.initialize_coefficients requires params so coefficients can be stored";
        assert isinstance(t_Grid, list);
        assert isinstance(Latent_States, list);
        assert len(Latent_States) == len(t_Grid) == params.shape[0];

        # Cycle through each parameter independently. Each parameter may have its own time grid.
        for i in range(len(t_Grid)):
            t_Grid0 : torch.Tensor = t_Grid[i];
            Z       : torch.Tensor = Latent_States[i][0];
            n_t     : int          = len(t_Grid0);

            # Approximate dZ/dt using the finite-difference stencil appropriate for this time grid.
            if(self.Uniform_t_Grid == True):
                h       : float         = (t_Grid0[1] - t_Grid0[0]).item();
                dZdt    : torch.Tensor  = Derivative1_Order4(Z, h);
            else:
                dZdt                    = Derivative1_Order2_NonUniform(Z, t_Grid = t_Grid0);

            # Build the affine SINDy library [1, z_1, ..., z_n].
            Z_1 : torch.Tensor = torch.cat([torch.ones(n_t, 1, device = Z.device, dtype = Z.dtype), Z], dim = 1);
            # Solve the regularized normal equations for the coefficient matrix.
            n_lib   : int           = Z_1.shape[1];
            rhs     : torch.Tensor  = Z_1.T @ dZdt;
            if self.lstsq_reg > 0.0:
                gram    : torch.Tensor  = Z_1.T @ Z_1 + self.lstsq_reg * torch.eye(n_lib, device = Z_1.device, dtype = Z_1.dtype);
                coefs   : torch.Tensor  = torch.linalg.solve(gram, rhs);
            else:
                coefs   : torch.Tensor  = torch.linalg.lstsq(Z_1, dZdt).solution;

            # Store the result in native form. This intentionally overwrites the coefficient entry
            # for this exact parameter if it already exists.
            self.set_train_coefs(params[i, :], self._native_from_matrix(coefs), device);

        # Finally, update the interpolator using the new training coefficients!
        self.update_interpolator();

        # All done :) 
        return None;



    def compute_losses(  
        self,  
        Latent_States   : list[list[torch.Tensor]], 
        t_Grid          : list[torch.Tensor], 
        step            : int,
        params          : numpy.ndarray | None = None
    ) -> LD_Loss_Container:
        r"""
        Evaluate the SINDy latent-dynamics loss using LD-owned native coefficients.

        `compute_losses` no longer receives coefficient tensors from the Trainer. Instead, it looks 
        up the coefficient dictionary for each parameter row in `self.train_coefs`. Missing entries
        raise a KeyError through `get_train_coefs`, which is intentional: by the time training
        starts, the sampler/initialization path should already have fitted coefficients for every
        training parameter.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        Latent_States : list[list[torch.Tensor]], len = n_param
            Encoded latent trajectories. The i'th entry contains one tensor of shape (n_t(i), n_z).

        t_Grid : list[torch.Tensor], len = n_param
            Time grids corresponding to the latent trajectories.

        params : numpy.ndarray, shape = (n_param, n_p)
            Parameter rows used to fetch native coefficient dictionaries.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        losses : LD_Loss_Container
            Container housing the loss values, matching loss weights, and parameter rows used to
            compute the losses. Its `losses` dictionary has three keys: LD, coef, and stab.

            losses.losses['LD'] : list[torch.Tensor], len = n_param
                The i'th element of this list is a 0-dimensional tensor whose lone element holds the
                SINDy loss from the i'th combination of parameter values.

            losses.losses['coef'] : list[torch.Tensor], len = n_param
                The i'th element of this list is a 0-dimensional tensor whose lone element holds the
                coefficient loss (Frobenius norm) of the coefficients for the i'th combination
                of parameter values.

            losses.losses['stab'] : list[torch.Tensor], len = n_param
                The i'th element of this list is a 0-dimensional tensor whose lone element holds the
                stability penalty for the i'th combination of parameter values (see
                LatentDynamics.stability_penalty).
        """

        # Checks.
        assert params is not None, "SINDy.compute_losses requires params to look up train_coefs";
        assert isinstance(t_Grid, list);
        assert isinstance(Latent_States, list);
        assert len(Latent_States) == len(t_Grid) == params.shape[0];

        # Prepare lists for per-parameter losses. The Trainer is responsible for applying weights
        # and summing these scalar losses into the total objective.
        loss_LD_list : list[torch.Tensor] = [];
        loss_coef_list : list[torch.Tensor] = [];
        loss_stab_list : list[torch.Tensor] = [];

        for i in range(len(t_Grid)):
            # Fetch this parameter's latent trajectory and time grid.
            t_Grid0 : torch.Tensor  = t_Grid[i];
            Z       : torch.Tensor  = Latent_States[i][0];
            n_t     : int           = len(t_Grid0);

            # Compute dZ/dt. Uniform grids use the higher-order stencil; nonuniform grids use the
            # nonuniform finite-difference helper.
            if(self.Uniform_t_Grid == True):
                h       : float         = (t_Grid0[1] - t_Grid0[0]).item();
                dZdt    : torch.Tensor  = Derivative1_Order4(Z, h);
            else:
                dZdt                    = Derivative1_Order2_NonUniform(Z, t_Grid = t_Grid0);

            # Fetch native trainable coefficients. This direct lookup is deliberately strict.
            # If the sampler/initialization path forgot to fit coefficients for this parameter,
            # get_train_coefs raises KeyError and stops the run.
            coef_dict = self.get_train_coefs(params[i, :]);
            A = coef_dict["A"].to(device = Z.device, dtype = Z.dtype);
            b = coef_dict["b"].to(device = Z.device, dtype = Z.dtype);

            # Evaluate the affine latent dynamics z' = A z + b on the encoded trajectory.
            RHS = Z @ A.T + b.reshape(1, -1);

            # Compute the data-fit part of the latent-dynamics loss.
            loss_LD = self.MSE(dZdt, RHS);

            # Compute regularization terms. The stability penalty depends only on A, while the
            # coefficient penalty includes both A and the affine shift b.
            loss_stab = self.stability_penalty(A);
            loss_coef = torch.norm(A, 'fro') + torch.norm(b);

            # Store per-parameter losses for the Trainer to weight/sum.
            loss_LD_list.append(loss_LD);
            loss_coef_list.append(loss_coef);
            loss_stab_list.append(loss_stab);

        losses_dict = {'LD' : loss_LD_list, 'coef' : loss_coef_list, 'stab' : loss_stab_list};

        return LD_Loss_Container(losses = losses_dict, weights = self.loss_weights, params = params);


    def RHS(    self,
                Z       : list[list[torch.Tensor | numpy.ndarray]],
                t_Grid  : list[numpy.ndarray | torch.Tensor],
                params  : numpy.ndarray,
                sample  : bool = False) -> list[torch.Tensor | numpy.ndarray]:
        r"""
        Evaluate the affine SINDy right-hand side at a set of latent states and parameters.

        For each parameter value, theta, we evaluate

            z'(t) = A(theta) z(t) + b(theta)

        at each latent state in `Z[i][0]`. Training parameters use exact coefficients from
        `self.train_coefs`; testing parameters use the interpolator mean or a sample.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        Z : list[list[torch.Tensor | numpy.ndarray]], len = n_param
            The i'th element is a one-element list whose first entry is a tensor/array of shape
            (n_t(i), n_z) or (n_t(i), n_batch(i), n_z).

        t_Grid : list[numpy.ndarray | torch.Tensor], len = n_param
            The i'th entry is a one-dimensional time grid with length n_t(i). The SINDy RHS is
            autonomous, so these values are checked for consistency but not otherwise used.

        params : numpy.ndarray, shape = (n_param, n_p)
            Parameter rows corresponding to the latent states stored in Z.

        sample : bool
            If True, use one interpolator sample for each non-training parameter. Otherwise, use
            interpolator posterior means. Training parameters always use exact training
            coefficients.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        RH_Sides : list[numpy.ndarray | torch.Tensor], len = n_param
            The i'th entry has the same backend and leading dimensions as `Z[i][0]` and last
            dimension n_z. It stores A(theta) z + b evaluated at the supplied states.
        """

        # Checks.
        assert isinstance(params, numpy.ndarray), "params must be a 2d numpy.ndarray, not %s" % str(type(params));
        assert len(params.shape) == 2, "params must be a 2d numpy.ndarray of shape (n_param, n_p). Got shape %s" % str(params.shape);
        n_param : int = params.shape[0];
        assert isinstance(Z, list) and len(Z) == n_param;
        assert isinstance(t_Grid, list) and len(t_Grid) == n_param;


        # -----------------------------------------------------------------------------------------
        # Fetch coefficient dictionaries for the passed parameters.
        # -----------------------------------------------------------------------------------------

        coefs_list : list[dict[str, torch.Tensor]] = self._coefs_for_params(params = params, sample = sample);


        # -----------------------------------------------------------------------------------------
        # Compute right-hand sides.
        # -----------------------------------------------------------------------------------------

        RH_Sides : list[numpy.ndarray | torch.Tensor] = [];
        LOGGER.debug("Computing RHS with %d parameter combinations" % n_param);
        for i in range(n_param):
            ith_coefs  : dict[str, numpy.ndarray | torch.Tensor] = coefs_list[i];
            ith_Z      : list[numpy.ndarray | torch.Tensor]      = Z[i];
            ith_t_Grid : numpy.ndarray | torch.Tensor            = t_Grid[i];

            # Checks.
            assert isinstance(ith_coefs, dict) and set(ith_coefs.keys()) == {"A", "b"};
            assert isinstance(ith_Z, list) and len(ith_Z) == 1;
            ith_Z0 : numpy.ndarray | torch.Tensor = ith_Z[0];
            assert isinstance(ith_Z0, (numpy.ndarray, torch.Tensor));
            assert len(ith_Z0.shape) in {2, 3};
            assert ith_Z0.shape[-1] == self.n_z;
            assert len(ith_t_Grid.shape) == 1;
            assert ith_Z0.shape[0] == ith_t_Grid.shape[0];

            # Fetch native coefficients.
            A = ith_coefs["A"];
            b = ith_coefs["b"];
            b_shape : tuple[int, ...] = (1,)*(len(ith_Z0.shape) - 1) + (-1,);

            # Evaluate the affine RHS using the same backend as the latent states.
            if isinstance(ith_Z0, numpy.ndarray):
                if isinstance(A, torch.Tensor):
                    A = A.detach().cpu().numpy();
                    b = b.detach().cpu().numpy();
                RH_Sides.append(numpy.matmul(ith_Z0, A.T) + b.reshape(b_shape));
            else:
                if isinstance(A, numpy.ndarray):
                    A = torch.tensor(A, dtype = ith_Z0.dtype, device = ith_Z0.device);
                    b = torch.tensor(b, dtype = ith_Z0.dtype, device = ith_Z0.device);
                else:
                    A = A.to(device = ith_Z0.device, dtype = ith_Z0.dtype);
                    b = b.to(device = ith_Z0.device, dtype = ith_Z0.dtype);
                RH_Sides.append(torch.matmul(ith_Z0, A.T) + b.reshape(b_shape));

        # All done!
        return RH_Sides;



    def simulate(   self,
                    IC      : list[list[numpy.ndarray | torch.Tensor]],
                    t_Grid  : list[numpy.ndarray      | torch.Tensor],
                    params  : numpy.ndarray,
                    sample  : bool = False) -> list[list[numpy.ndarray | torch.Tensor]]:
        r"""
        Time-integrate the native SINDy latent dynamics.

        Coefficients are fetched from `self.train_coefs` for training parameters and from
        `self.interpolator` for non-training parameters.

        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        IC : list[list[numpy.ndarray | torch.Tensor]], len = n_param
            Initial latent states for each coefficient set. SINDy has one IC component.

        t_Grid : list[numpy.ndarray | torch.Tensor], len = n_param
            Time grids for simulation.

        params : numpy.ndarray, shape = (n_param, n_p)
            The i'th row holds the i'th combination of parameter values.
        
        sample : bool 
            If self is stochastic, setting this to true will sample from the posterior distribution 
            of the latent dynamics at each parameter value, then solve the latent dynamics using 
            the resulting sample. Otherwise, setting this to true will use the mean of that 
            posterior distribution. If self is not stochastic, this does nothing.
            

        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Z : list[list[numpy.ndarray | torch.Tensor]]
            Simulated latent trajectories. Z[i][0] has shape (n_t(i), n_initial_conditions, n_z).
        """

        # Checks.
        assert isinstance(params, numpy.ndarray);
        assert len(params.shape) == 2;
        n_param : int = params.shape[0];
        assert isinstance(t_Grid, list) and isinstance(IC, list);
        assert len(IC) == n_param and len(t_Grid) == n_param;


        # -----------------------------------------------------------------------------------------
        # Fetch coefficient dictionaries for the passed parameters.
        # -----------------------------------------------------------------------------------------

        coefs_list : list[dict[str, torch.Tensor]] = self._coefs_for_params(params = params, sample = sample);


        # -----------------------------------------------------------------------------------------
        # Loop through parameter combinations.
        # -----------------------------------------------------------------------------------------

        Z : list[list[numpy.ndarray | torch.Tensor]] = [];
        LOGGER.debug("Simulating with %d parameter combinations" % n_param);
        for i in range(n_param):
            # Fetch the i'th set of coefficients, initial conditions, and time grid.
            ith_coefs  : dict[str, numpy.ndarray | torch.Tensor] = coefs_list[i];
            ith_IC     : list[numpy.ndarray | torch.Tensor]      = IC[i];
            ith_t_Grid : numpy.ndarray | torch.Tensor            = t_Grid[i];

            # Set up the i'th single-parameter solve.
            assert isinstance(ith_coefs, dict) and set(ith_coefs.keys()) == {"A", "b"};
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

            # Fetch native coefficients for this parameter.
            A = ith_coefs["A"];
            b = ith_coefs["b"];

            # Match the coefficient backend to the initial-condition backend. This keeps the solver
            # purely NumPy for NumPy inputs and differentiable PyTorch for tensor inputs.
            if isinstance(ith_Z0, numpy.ndarray):
                if isinstance(A, torch.Tensor):
                    A = A.detach().cpu().numpy();
                    b = b.detach().cpu().numpy();
                b = b.reshape(1, -1);
                f = lambda t, z: b + numpy.matmul(z, A.T);
            else:
                if isinstance(A, numpy.ndarray):
                    A = torch.tensor(A, dtype = ith_Z0.dtype, device = ith_Z0.device);
                    b = torch.tensor(b, dtype = ith_Z0.dtype, device = ith_Z0.device);
                else:
                    A = A.to(device = ith_Z0.device, dtype = ith_Z0.dtype);
                    b = b.to(device = ith_Z0.device, dtype = ith_Z0.dtype);
                b = b.reshape(1, -1);
                f = lambda t, z: b + torch.matmul(z, A.T);

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
