# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

# Add the main directory to the search path.
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
from    SecondOrderSolvers  import  RK4;


# Setup Logger.
LOGGER : logging.Logger = logging.getLogger(__name__);



# -------------------------------------------------------------------------------------------------
# DampedSpring class
# -------------------------------------------------------------------------------------------------

class DampedSpring(LatentDynamics):
    def __init__(   self, 
                    n_z             :   int, 
                    Uniform_t_Grid  :   bool,
                    config          :   dict) -> None:
        r"""
        Initializes a DampedSpring latent-dynamics object.

        This class models second-order latent dynamics in native form as

            z''(t) = K z(t) + C z'(t) + b.

        Here, z is the latent state. K \in \mathbb{R}^{n x n} and C \in \mathbb{R}^{n x n}
        are the two linear coefficient matrices in the second-order latent model, while b is an
        offset/constant forcing vector. There is a separate set of coefficients for each
        combination of parameter values. We store the tensors in 
        `self.train_coefs` as

            {"K": K, "C": C, "b": b}.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        n_z : int
            Number of latent dimensions.

        Uniform_t_Grid : bool
            Selects uniform-grid or nonuniform-grid finite differences when estimating
            accelerations from latent trajectories.

        config : dict
            Latent-dynamics configuration. The optional `lstsq_reg` value controls ridge
            regularization in `fit_coefficients`.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Nothing!
        """

        # Run the base class initializer. This also creates the LD-owned train_coefs dictionary.
        super().__init__(n_z = n_z, Uniform_t_Grid = Uniform_t_Grid, config = config);
        self.lstsq_reg : float = config.get("lstsq_reg", 1.0);
        LOGGER.info("Initializing a DampedSpring object with n_z = %d, Uniform_t_Grid = %s, lstsq_reg = %s" % (self.n_z, str(self.Uniform_t_Grid), str(self.lstsq_reg)));        
        # The model needs displacement and velocity initial conditions. We keep the flattened scalar
        # count for diagnostics, even though storage is now native.
        self.n_IC       : int   = 2;
        self.n_coefs    : int   = n_z*(2*n_z + 1);
        self.MSE = torch.nn.MSELoss(reduction = 'mean');
        self.MAE = torch.nn.L1Loss(reduction = 'mean');
        return;



    def _native_from_matrix(self, coefs : torch.Tensor) -> dict[str, torch.Tensor]:
        r"""
        Convert a least-squares matrix into native second-order coefficients.

        The least-squares library is [z, z', 1], so the solved matrix E satisfies
        [z, z', 1] E ~= z''. Therefore E stores K^T, C^T, and b. We detach/clone these tensors
        into leaf tensors so optimizers can update them in place through `self.train_coefs`.
        """

        assert coefs.shape == (2*self.n_z + 1, self.n_z);
        K   = coefs[0:self.n_z, :].T.detach().clone().requires_grad_(True);
        C   = coefs[self.n_z:(2*self.n_z), :].T.detach().clone().requires_grad_(True);
        b   = coefs[2*self.n_z, :].detach().clone().requires_grad_(True);
        return {"K": K, "C": C, "b": b};



    def _matrix_from_native(self, coefs : dict[str, torch.Tensor]) -> torch.Tensor:
        r"""Reconstruct the [K^T; C^T; b] matrix used by the least-squares fit."""

        K = coefs["K"];
        C = coefs["C"];
        b   = coefs["b"];
        assert K.shape == (self.n_z, self.n_z);
        assert C.shape == (self.n_z, self.n_z);
        assert b.shape == (self.n_z,);
        return torch.cat([K.T, C.T, b.reshape(1, self.n_z)], dim = 0);



    def train_coef_tensors(self) -> list[torch.Tensor]:
        r"""Return the trainable coefficient tensors stored in `self.train_coefs`."""

        tensors : list[torch.Tensor] = [];
        for coef_dict in self.train_coefs.values():
            tensors.extend([coef_dict["K"], coef_dict["C"], coef_dict["b"]]);
        return tensors;



    def flatten_coefficients(self, coefs : dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]) -> numpy.ndarray:
        r"""Flatten damped-spring coefficients in the legacy [K^T; C^T; b] ordering."""

        coefs_list = [coefs] if isinstance(coefs, dict) else coefs;
        assert isinstance(coefs_list, list);
        rows : list[numpy.ndarray] = [];
        for coef_dict in coefs_list:
            mat = self._matrix_from_native(coef_dict);
            rows.append(mat.detach().cpu().numpy().reshape(1, -1));
        return numpy.concatenate(rows, axis = 0);



    def fit_coefficients(self,
                         Latent_States : list[list[torch.Tensor]],
                         t_Grid        : list[torch.Tensor],
                         params        : numpy.ndarray | None = None) -> None:
        r"""
        Fit coefficients for the damped-spring latent dynamics model from latent trajectories.

        This computes a least-squares (optionally ridge-regularized) estimate of the native
        coefficient tensors K, C, and b for each parameter combination. It is intended for
        coefficient initialization (for example, after greedy sampling adds a new training point).
        Unlike the previous interface, this method stores the coefficients in `self.train_coefs`
        rather than returning a flattened coefficient matrix.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        Latent_States : list[list[torch.Tensor]], len = n_param
            The i'th list element is a two-element list. The first tensor holds the latent
            displacement trajectory Z_D with shape (n_t(i), n_z), and the second tensor holds the
            latent velocity trajectory Z_V with shape (n_t(i), n_z).

        t_Grid : list[torch.Tensor], len = n_param
            The i'th element is a 1D tensor of shape (n_t(i)) holding the time grid for the i'th
            parameter combination.

        params : numpy.ndarray, shape = (n_param, n_p)
            The i'th row holds the parameter values associated with the i'th trajectory. These rows
            are converted to exact tuple keys in `self.train_coefs`.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        None. Coefficients are stored in `self.train_coefs`.
        """

        # Checks.
        assert params is not None, "DampedSpring.fit_coefficients requires params so coefficients can be stored";
        assert isinstance(t_Grid, list);
        assert isinstance(Latent_States, list);
        assert len(Latent_States) == len(t_Grid) == params.shape[0];

        for i in range(len(t_Grid)):
            Z       = Latent_States[i];
            t_Grid0 = t_Grid[i];
            assert isinstance(Z, list) and len(Z) == 2;
            Z_D : torch.Tensor = Z[0];
            Z_V : torch.Tensor = Z[1];

            # Library matrix [Z_D, Z_V, 1].
            ones    : torch.Tensor  = torch.ones((Z_D.shape[0], 1), device = Z_D.device, dtype = Z_D.dtype);
            ZD_ZV_1 : torch.Tensor  = torch.cat([Z_D, Z_V, ones], dim = 1);

            # Compute acceleration using d/dt of the velocity stream.
            if(self.Uniform_t_Grid  == True):
                h : float = (t_Grid0[1] - t_Grid0[0]).item();
                d2Z_dt2 : torch.Tensor = Derivative1_Order4(U = Z_V, h = h);
            else:
                d2Z_dt2 = Derivative1_Order2_NonUniform(U = Z_V, t_Grid = t_Grid0);

            # Solve for E in [Z_D, Z_V, 1] E ~= d2Z/dt2.
            n_lib : int = ZD_ZV_1.shape[1];
            rhs   : torch.Tensor = ZD_ZV_1.T @ d2Z_dt2;
            if self.lstsq_reg > 0.0:
                gram  : torch.Tensor = ZD_ZV_1.T @ ZD_ZV_1 + self.lstsq_reg * torch.eye(n_lib, device = ZD_ZV_1.device, dtype = ZD_ZV_1.dtype);
                coefs : torch.Tensor = torch.linalg.solve(gram, rhs);
            else:
                coefs = torch.linalg.lstsq(ZD_ZV_1, d2Z_dt2).solution;

            self.set_train_coefs(params[i, :], self._native_from_matrix(coefs));
        return None;



    def calibrate(self, 
                  Latent_States : list[list[torch.Tensor]],
                  loss_type     : str,
                  t_Grid        : list[torch.Tensor],
                  params        : numpy.ndarray | None = None) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        r"""
        Compute latent-dynamics, coefficient, and stability losses for training parameters.

        For each parameter row, this method fetches the native coefficient dictionary from
        `self.train_coefs` and evaluates the second-order latent dynamics

            z''(t) = K z(t) + C z'(t) + b.

        This method assumes coefficients have already been initialized by `fit_coefficients(...)`;
        missing entries are hard errors and indicate a sampler/initialization bug.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        Latent_States : list[list[torch.Tensor]], len = n_param
            The i'th list element is a two-element list whose entries are latent displacement and
            velocity tensors with shape (n_t(i), n_z).

        loss_type : str
            The type of loss function to use. Must be either "MSE" or "MAE".

        t_Grid : list[torch.Tensor], len = n_param
            The i'th element is a 1D tensor of shape (n_t(i)) holding the time grid for the i'th
            parameter combination.

        params : numpy.ndarray, shape = (n_param, n_p)
            The i'th row holds the parameter values used to look up the corresponding native
            coefficient dictionary.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        loss_LD_list : list[torch.Tensor], len = n_param
            The i'th element is a scalar tensor containing the latent-dynamics residual loss for
            the i'th parameter combination.

        loss_coef_list : list[torch.Tensor], len = n_param
            The i'th element is a scalar tensor containing the coefficient regularization value for
            the i'th parameter combination.

        loss_stab_list : list[torch.Tensor], len = n_param
            The i'th element is a scalar tensor containing the stability penalty for the i'th
            parameter combination.
        """

        # Checks.
        assert params is not None, "DampedSpring.calibrate requires params to look up train_coefs";
        assert isinstance(t_Grid, list);
        assert isinstance(Latent_States, list);
        assert len(Latent_States) == len(t_Grid) == params.shape[0];
        assert loss_type in ["MSE", "MAE"];

        loss_LD_list : list[torch.Tensor] = [];
        loss_coef_list : list[torch.Tensor] = [];
        loss_stab_list : list[torch.Tensor] = [];


        # -----------------------------------------------------------------------------------------
        # Loop over parameter combinations.
        # -----------------------------------------------------------------------------------------

        for i in range(len(t_Grid)):
            # Fetch latent displacement/velocity trajectories and the corresponding time grid.
            Z_D : torch.Tensor = Latent_States[i][0];
            Z_V : torch.Tensor = Latent_States[i][1];
            t_Grid0 : torch.Tensor = t_Grid[i];

            # Approximate acceleration. As in the previous implementation, we use d/dt of the
            # latent velocity stream.
            if(self.Uniform_t_Grid  == True):
                h : float = (t_Grid0[1] - t_Grid0[0]).item();
                d2Z_dt2 : torch.Tensor = Derivative1_Order4(U = Z_V, h = h);
            else:
                d2Z_dt2 = Derivative1_Order2_NonUniform(U = Z_V, t_Grid = t_Grid0);


            # -------------------------------------------------------------------------------------
            # Evaluate the native second-order model.
            # -------------------------------------------------------------------------------------

            # Fetch native trainable coefficients for this parameter.
            coef_dict = self.get_train_coefs(params[i, :]);
            K = coef_dict["K"].to(device = Z_D.device, dtype = Z_D.dtype);
            C = coef_dict["C"].to(device = Z_D.device, dtype = Z_D.dtype);
            b   = coef_dict["b"].to(device = Z_D.device, dtype = Z_D.dtype).reshape(1, -1);

            # Evaluate z'' = K z + C z' + b. The signs are important here: in the native
            # coefficient convention K and C are the actual linear operators appearing in the
            # right-hand side (not the old "spring/damping" matrices that were negated after
            # unpacking a flattened coefficient vector).
            LD_RHS = torch.matmul(Z_D, K.T) + torch.matmul(Z_V, C.T) + b;

            if(loss_type == "MSE"):
                Loss_LD = self.MSE(d2Z_dt2, LD_RHS);
            else:
                Loss_LD = self.MAE(d2Z_dt2, LD_RHS);


            # -------------------------------------------------------------------------------------
            # Stability and coefficient regularization.
            # -------------------------------------------------------------------------------------

            # Convert the second-order system to the first-order linear part
            #     [z, z']' = [[0, I], [K, C]] [z, z'] + [0, b].
            # The base stability penalty is defined for first-order systems.
            Z0  : torch.Tensor  = torch.zeros((self.n_z, self.n_z), device = Z_D.device, dtype = Z_D.dtype);
            I   : torch.Tensor  = torch.eye(self.n_z, device = Z_D.device, dtype = Z_D.dtype);
            A_top    = torch.cat([Z0, I], dim = 1);
            A_bottom = torch.cat([K, C], dim = 1);
            A = torch.cat([A_top, A_bottom], dim = 0);
            Loss_Stab = self.stability_penalty(A);

            # Penalize all native coefficient tensors.
            Loss_coef = torch.norm(K, 'fro') + torch.norm(C, 'fro') + torch.norm(b);

            # Store per-parameter losses for the Trainer to weight/sum.
            loss_LD_list.append(Loss_LD);
            loss_coef_list.append(Loss_coef);
            loss_stab_list.append(Loss_Stab);

        return loss_LD_list, loss_coef_list, loss_stab_list;



    def simulate(   self,
                    coefs   : dict[str, numpy.ndarray   | torch.Tensor] | list[dict[str, numpy.ndarray | torch.Tensor]], 
                    IC      : list[list[numpy.ndarray   | torch.Tensor]],
                    t_Grid  : list[numpy.ndarray        | torch.Tensor],
                    params  : numpy.ndarray | None = None) -> list[list[numpy.ndarray | torch.Tensor]]:
        r"""
        Time integrates the latent dynamics from multiple initial conditions for each combination
        of coefficients in coefs. 
 

        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------
        
        coefs : dict or list[dict]
            Native coefficient dictionary/dictionaries. For DampedSpring each dictionary must
            contain `K` with shape (n_z, n_z), `C` with shape (n_z, n_z), and `b` with shape
            (n_z,). These are the coefficients in z'' = K z + C z' + b.

        IC : list[list[numpy.ndarray]] or list[list[torch.Tensor]], len = n_param
            i'th element is an n_IC element list whose j'th element is a 2d numpy.ndarray or 
            torch.Tensor object of shape (n(i), n_z). Here, n(i) is the number of initial 
            conditions (for a fixed set of coefficients) we want to simulate forward using the i'th 
            set of coefficients. Further, n_z is the latent dimension. If you want to simulate a 
            single IC, for the i'th set of coefficients, then n(i) == 1. IC[i][j][k, :] should hold 
            the k'th initial condition for the j'th derivative of the latent state when we use the 
            i'th combination of parameter values. 

        t_Grid : list[numpy.ndarray] or list[torch.Tensor], len = n_param
            i'th entry is a 2d numpy.ndarray or torch.Tensor whose shape is either (n(i), n_t(i)) 
            or shape (n_t(i)). The shape should be 2d if we want to use different times for each 
            initial condition and 1d if we want to use the same times for all initial conditions. 
        
            In the former case, the j,k array entry specifies k'th time value at which we solve for 
            the latent state when we use the j'th initial condition and the i'th set of 
            coefficients. Each row should be in ascending order. 
        
            In the latter case, the j'th entry should specify the j'th time value at which we solve 
            for each latent state when we use the i'th combination of parameter values.

        params: numpy.ndarray, shape = (n_param, n_p), optional
            The i'th row holds the i'th combination of parameter values. This class doesn't use 
            parameters, so it ignores this argument. Default is None.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------        
        
        Z : list[list[numpy.ndarray]] or list[list[torch.Tensor]], len = n_parm
            i'th element is a list of length n_IC whose j'th entry is a 3d array of shape 
            (n_t(i), n(i), n_z). The p, q, r entry of this array should hold the r'th component of 
            the p'th frame of the j'th tine derivative of the solution to the latent dynamics when 
            we use the q'th initial condition for the i'th combination of parameter values.
        """

        # Normalize coefficient input to a list so the multi-parameter case is straightforward.
        if isinstance(coefs, dict):
            coefs_list = [coefs];
        else:
            coefs_list = coefs;
        assert isinstance(coefs_list, list);
        n_param : int = len(coefs_list);
        assert isinstance(t_Grid, list) and isinstance(IC, list);
        assert len(IC) == n_param and len(t_Grid) == n_param;

        # Multi-parameter simulation: recurse one parameter at a time so all backend conversion and
        # RK4 setup lives in the single-parameter branch.
        if(n_param > 1):
            return [self.simulate(coefs = coefs_list[i], IC = [IC[i]], t_Grid = [t_Grid[i]], params = None if params is None else params[i, :].reshape(1, -1))[0] for i in range(n_param)];


        # -----------------------------------------------------------------------------------------
        # One-parameter setup.
        # -----------------------------------------------------------------------------------------

        assert isinstance(IC[0], list) and len(IC[0]) == 2;
        t_Grid0  : numpy.ndarray | torch.Tensor  = t_Grid[0];
        if(isinstance(t_Grid0, torch.Tensor)):
            t_Grid0 = t_Grid0.detach().cpu().numpy();
        Same_t_Grid : bool = (len(t_Grid0.shape) == 1);
        D0  : numpy.ndarray | torch.Tensor  = IC[0][0]; 
        V0  : numpy.ndarray | torch.Tensor  = IC[0][1];
        n_i : int = D0.shape[0];

        # Fetch native coefficients and match their backend to the IC backend.
        coef_dict = coefs_list[0];
        K   = coef_dict["K"];
        C   = coef_dict["C"];
        b   = coef_dict["b"];

        # Define the RHS with the same backend as the initial conditions. Tensor inputs preserve
        # differentiability for training rollouts; NumPy inputs keep sampling/plotting lightweight.
        if isinstance(D0, numpy.ndarray):
            if isinstance(K, torch.Tensor):
                K   = K.detach().cpu().numpy();
                C   = C.detach().cpu().numpy();
                b   = b.detach().cpu().numpy();
            b = b.reshape(1, -1);
            f = lambda t, z, dz_dt: numpy.matmul(z, K.T) + numpy.matmul(dz_dt, C.T) + b;
        else:
            if isinstance(K, numpy.ndarray):
                K   = torch.tensor(K, dtype = D0.dtype, device = D0.device);
                C   = torch.tensor(C, dtype = D0.dtype, device = D0.device);
                b   = torch.tensor(b, dtype = D0.dtype, device = D0.device);
            else:
                K   = K.to(device = D0.device, dtype = D0.dtype);
                C   = C.to(device = D0.device, dtype = D0.dtype);
                b   = b.to(device = D0.device, dtype = D0.dtype);
            b = b.reshape(1, -1);
            f = lambda t, z, dz_dt: torch.matmul(z, K.T) + torch.matmul(dz_dt, C.T) + b;

        # Integrate all ICs together when they share one time grid. If each IC has its own time
        # grid, integrate them separately and concatenate along the IC axis.
        if(Same_t_Grid == True):
            D, V = RK4(f = f, y0 = D0, Dy0 = V0, t_Grid = t_Grid0); # shape = (n_t, n_i, n_z)
        else:
            # Cycle through the ICs.
            D_list : list[torch.Tensor | numpy.ndarray] = [];
            V_list : list[torch.Tensor | numpy.ndarray] = []; 
           
            for j in range(n_i):
                D_j, V_j = RK4(f = f, y0 = D0[j, :].reshape(1, -1), Dy0 = V0[j, :].reshape(1, -1), t_Grid = t_Grid0[j, :]);
                D_list.append(D_j);
                V_list.append(V_j);
            
            # Stack the results.
            if(isinstance(D0, numpy.ndarray)):
                D = numpy.concatenate(D_list, axis = 1);    # shape = (n_t, n_i, n_z)
                V = numpy.concatenate(V_list, axis = 1);    # shape = (n_t, n_i, n_z)
            else:
                D = torch.cat(D_list, dim = 1);             # shape = (n_t, n_i, n_z)
                V = torch.cat(V_list, dim = 1);             # shape = (n_t, n_i, n_z)
        
        # All done!
        return [[D, V]];
