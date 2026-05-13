# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

import  os;
import  sys;
src_Path        : str   = os.path.dirname(os.path.dirname(__file__));
util_Path       : str   = os.path.join(src_Path, "Utilities");
sys.path.append(src_Path);
sys.path.append(util_Path);

import  logging;

import  numpy;
import  torch;

from    LatentDynamics      import  LatentDynamics;
from    FiniteDifference    import  Derivative1_Order4, Derivative1_Order2_NonUniform;
from    FirstOrderSolvers   import  RK4;

LOGGER  : logging.Logger    = logging.getLogger(__name__);



# -------------------------------------------------------------------------------------------------
# SINDy class
# -------------------------------------------------------------------------------------------------

class SINDy(LatentDynamics):
    def __init__(   self, 
                    n_z             : int,
                    Uniform_t_Grid  : bool,
                    config          : dict,
                    lstsq_reg       : float = 1.0) -> None:
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

        config : dict
            The latent-dynamics configuration dictionary. The optional `lstsq_reg` entry controls
            ridge regularization used by `fit_coefficients(...)` when initializing coefficients
            from encoded trajectories.

        lstsq_reg : float
            Kept for compatibility with the previous constructor signature; the config value takes
            precedence when present.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Nothing!
        """

        # Run the base class initializer. This sets `n_z`, the time-grid convention, the config,
        # and the LD-owned `train_coefs` dictionary.
        super().__init__(n_z = n_z, Uniform_t_Grid = Uniform_t_Grid, config = config);
        self.lstsq_reg : float = config.get("lstsq_reg", 1.0);
        LOGGER.info("Initializing a SINDY object with n_z = %d, Uniform_t_Grid = %s, lstsq_reg = %s" % (self.n_z, str(self.Uniform_t_Grid), str(self.lstsq_reg)));

        # We keep `n_coefs` as the flattened count because several diagnostics/plotting utilities
        # still report the total scalar coefficient count. The storage itself is native: A and b.
        self.n_coefs    : int   = self.n_z*(self.n_z + 1);
        self.n_IC       : int   = 1;

        # Setup the loss functions used by calibrate.
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



    def trainable_coef_tensors(self) -> list[torch.Tensor]:
        r"""
        Return the actual coefficient tensors that should be passed to torch optimizers.

        These are not copies. They are the same tensors stored in `self.train_coefs`, so optimizer
        updates modify the LD-owned coefficient dictionaries used by calibrate/simulate.
        """

        tensors : list[torch.Tensor] = [];
        for coef_dict in self.train_coefs.values():
            tensors.extend([coef_dict["A"], coef_dict["b"]]);
        return tensors;



    def fit_coefficients(self,
                         Latent_States   : list[list[torch.Tensor]],
                         t_Grid          : list[torch.Tensor],
                         params          : numpy.ndarray | None = None) -> None:
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

        params : numpy.ndarray, shape = (n_param, n_p)
            Parameter rows used as keys in `self.train_coefs`. This is required; omitting it is a
            bookkeeping error and should stop the run.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        None. Coefficients are stored in `self.train_coefs`.
        """

        # Checks.
        assert params is not None, "SINDy.fit_coefficients requires params so coefficients can be stored";
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
            self.set_train_coefs(params[i, :], self._native_from_matrix(coefs));
        return None;



    def calibrate(  self,  
                    Latent_States   : list[list[torch.Tensor]], 
                    loss_type       : str,
                    t_Grid          : list[torch.Tensor], 
                    params          : numpy.ndarray | None = None) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        r"""
        Evaluate the SINDy latent-dynamics loss using LD-owned native coefficients.

        `calibrate` no longer receives coefficient tensors from the Trainer. Instead, it looks up
        the coefficient dictionary for each parameter row in `self.train_coefs`. Missing entries
        raise a KeyError through `get_train_coefs`, which is intentional: by the time training
        starts, the sampler/initialization path should already have fitted coefficients for every
        training parameter.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        Latent_States : list[list[torch.Tensor]], len = n_param
            Encoded latent trajectories. The i'th entry contains one tensor of shape (n_t(i), n_z).

        loss_type : str
            Either "MSE" or "MAE".

        t_Grid : list[torch.Tensor], len = n_param
            Time grids corresponding to the latent trajectories.

        params : numpy.ndarray, shape = (n_param, n_p)
            Parameter rows used to fetch native coefficient dictionaries.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        loss_LD_list, loss_coef_list, loss_stab_list
            Three lists of scalar tensors, one scalar per parameter.
        """

        # Checks.
        assert params is not None, "SINDy.calibrate requires params to look up train_coefs";
        assert isinstance(t_Grid, list);
        assert isinstance(Latent_States, list);
        assert len(Latent_States) == len(t_Grid) == params.shape[0];
        assert loss_type in ["MSE", "MAE"];

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
            if(loss_type == "MSE"):
                loss_LD = self.MSE(dZdt, RHS);
            else:
                loss_LD = self.MAE(dZdt, RHS);

            # Compute regularization terms. The stability penalty depends only on A, while the
            # coefficient penalty includes both A and the affine shift b.
            loss_stab = self.stability_penalty(A);
            loss_coef = torch.norm(A, 'fro') + torch.norm(b);

            # Store per-parameter losses for the Trainer to weight/sum.
            loss_LD_list.append(loss_LD);
            loss_coef_list.append(loss_coef);
            loss_stab_list.append(loss_stab);

        return loss_LD_list, loss_coef_list, loss_stab_list;



    def simulate(   self,
                    coefs   : dict[str, numpy.ndarray | torch.Tensor] | list[dict[str, numpy.ndarray | torch.Tensor]], 
                    IC      : list[list[numpy.ndarray | torch.Tensor]],
                    t_Grid  : list[numpy.ndarray      | torch.Tensor],
                    params  : numpy.ndarray | None = None) -> list[list[numpy.ndarray | torch.Tensor]]:
        r"""
        Time-integrate the native SINDy latent dynamics.

        The coefficient argument is now either a single native dictionary {"A", "b"} or a list of
        such dictionaries. This lets callers pass coefficients returned by `Interpolate.sample`,
        `Interpolate.mean`, or direct `train_coefs` lookups without any flattened unpacking.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        coefs : dict or list[dict]
            Native coefficient dictionary/dictionaries. For SINDy each dictionary must contain
            `A` with shape (n_z, n_z) and `b` with shape (n_z,).

        IC : list[list[numpy.ndarray | torch.Tensor]], len = n_param
            Initial latent states for each coefficient set. SINDy has one IC component.

        t_Grid : list[numpy.ndarray | torch.Tensor], len = n_param
            Time grids for simulation.

        params : numpy.ndarray, optional
            Accepted for API consistency with parameter-dependent LD subclasses; unused here.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Z : list[list[numpy.ndarray | torch.Tensor]]
            Simulated latent trajectories. Z[i][0] has shape (n_t(i), n_initial_conditions, n_z).
        """

        # Normalize the coefficient input to a list so the multi-parameter and single-parameter
        # cases share the same bookkeeping.
        if isinstance(coefs, dict):
            coefs_list = [coefs];
        else:
            coefs_list = coefs;
        assert isinstance(coefs_list, list);
        n_param : int = len(coefs_list);
        assert isinstance(t_Grid, list) and isinstance(IC, list);
        assert len(IC) == n_param and len(t_Grid) == n_param;

        # Multi-parameter simulation: recurse one parameter at a time. This keeps the backend
        # conversion and RK4 setup concentrated in the single-parameter branch below.
        # If multiple coefficient sets are provided, recurse on each one so all validation and
        # backend-specific setup only needs to be written once.
        if n_param > 1:
            return [self.simulate(coefs = coefs_list[i], IC = [IC[i]], t_Grid = [t_Grid[i]], params = None if params is None else params[i, :].reshape(1, -1))[0] for i in range(n_param)];

        # One-parameter simulation.
        assert isinstance(IC[0], list) and len(IC[0]) == 1;
        t_Grid0  : numpy.ndarray | torch.Tensor  = t_Grid[0];
        if(isinstance(t_Grid0, torch.Tensor)):
            t_Grid0 = t_Grid0.detach().cpu().numpy();
        Same_t_Grid : bool = (len(t_Grid0.shape) == 1);
        Z0  : numpy.ndarray | torch.Tensor  = IC[0][0];
        n_i : int = Z0.shape[0];

        # Fetch native coefficients for the single-parameter case.
        coef_dict = coefs_list[0];
        A = coef_dict["A"];
        b = coef_dict["b"];
        # Match the coefficient backend to the initial-condition backend. This keeps the solver
        # purely NumPy for NumPy inputs and differentiable PyTorch for tensor inputs.
        if isinstance(Z0, numpy.ndarray):
            if isinstance(A, torch.Tensor):
                A = A.detach().cpu().numpy();
                b = b.detach().cpu().numpy();
            b = b.reshape(1, -1);
            f = lambda t, z: b + numpy.matmul(z, A.T);
        else:
            if isinstance(A, numpy.ndarray):
                A = torch.tensor(A, dtype = Z0.dtype, device = Z0.device);
                b = torch.tensor(b, dtype = Z0.dtype, device = Z0.device);
            else:
                A = A.to(device = Z0.device, dtype = Z0.dtype);
                b = b.to(device = Z0.device, dtype = Z0.dtype);
            b = b.reshape(1, -1);
            f = lambda t, z: b + torch.matmul(z, A.T);

        # Solve the ODE. If all ICs share the same time grid we integrate them as a batch;
        # otherwise, integrate each initial condition separately and concatenate the results.
        if(Same_t_Grid == True):
            Z = [[RK4(f = f, y0 = Z0, t_Grid = t_Grid0)]]; 
        else:
            Z_list : list[torch.Tensor | numpy.ndarray] = [];   
            for j in range(n_i):
                Z_j = RK4(f = f, y0 = Z0[j, :].reshape(1, -1), t_Grid = t_Grid0[j, :]);
                Z_list.append(Z_j);
            if(isinstance(Z0, numpy.ndarray)):
                Z = [[numpy.concatenate(Z_list, axis = 1)]];
            else:
                Z = [[torch.cat(Z_list, dim = 1)]];
        return Z;
