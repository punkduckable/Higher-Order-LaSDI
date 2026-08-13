# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

import  logging;

import  numpy;
import  torch;

from    LatentDynamics.Interpolatable   import  InterpolatableLatentDynamics;
from    Utilities.FiniteDifference      import  Derivative1_Order4, Derivative1_Order2_NonUniform;
from    Utilities.SecondOrderSolvers    import  RK4;


# Setup Logger.
LOGGER : logging.Logger = logging.getLogger(__name__);



# -------------------------------------------------------------------------------------------------
# DampedSpring class
# -------------------------------------------------------------------------------------------------

class DampedSpring(InterpolatableLatentDynamics):
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
            The latent-dynamics configuration dictionary. It must three keys: `type`, `trainable`,
            and `spring`. It must have `config["type"] == "spring"` and `config["spring"]` should 
            be a dictionary housing sub-class specific settings. The required `lstsq_reg` entry 
            controls ridge regularization used by `initialize_coefficients(...)` when initializing 
            coefficients from encoded trajectories.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Nothing!
        """
        
        # Checks
        assert  'type'      in config;
        assert  'trainable' in config;
        assert  isinstance(config["type"], str);
        assert  isinstance(config["trainable"], bool);
        assert  config['type'] == "spring";
        assert  "spring"    in config;

        # Run the base class initializer. This also creates the LD-owned train_coefs dictionary.
        InterpolatableLatentDynamics.__init__(   
            self,
            n_z             = n_z, 
            n_coefs         = n_z*(2*n_z + 1),
            n_IC            = 2,
            Uniform_t_Grid  = Uniform_t_Grid, 
            trainable       = config["trainable"],
            config          = config);

        # Class-specific variables.
        self.lstsq_reg : float = config["spring"]["lstsq_reg"];
        LOGGER.info("Initializing a DampedSpring object with n_z = %d, Uniform_t_Grid = %s, lstsq_reg = %s" % (self.n_z, str(self.Uniform_t_Grid), str(self.lstsq_reg)));        
        
        # Setup the loss functions used by compute_losses.
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



    def trainable_coef_tensors(self) -> list[torch.Tensor]:
        r"""Return the trainable coefficient tensors stored in `self.train_coefs`."""

        if self.trainable == False:
            return [];

        tensors : list[torch.Tensor] = [];
        for coef_dict in self.train_coefs.values():
            tensors.extend([coef_dict["K"], coef_dict["C"], coef_dict["b"]]);
        return tensors;



    def initialize_coefficients(
            self,
            Latent_States : list[list[torch.Tensor]],
            t_Grid        : list[torch.Tensor],
            device        : torch.device,
            params        : numpy.ndarray) -> None:
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

        device : torch.device
            The device where we want to store the new coefficients.
            
        params : numpy.ndarray, shape = (n_param, n_p)
            The i'th row holds the parameter values associated with the i'th trajectory. These rows
            are converted to exact tuple keys in `self.train_coefs`.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        None. Coefficients are stored in `self.train_coefs`.
        """

        # Checks.
        assert params is not None, "DampedSpring.initialize_coefficients requires params so coefficients can be stored";
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

            self.set_train_coefs(params[i, :], self._native_from_matrix(coefs), device);

        # Finally, update the interpolator using the new training coefficients!
        self.update_interpolator();

        # All done :) 
        return None;



    def compute_losses(
        self, 
        Latent_States : list[list[torch.Tensor]],
        loss_type     : str,
        t_Grid        : list[torch.Tensor],
        params        : numpy.ndarray | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        r"""
        Compute latent-dynamics, coefficient, and stability losses for training parameters.

        For each parameter row, this method fetches the native coefficient dictionary from
        `self.train_coefs` and evaluates the second-order latent dynamics

            z''(t) = K z(t) + C z'(t) + b.

        This method assumes coefficients have already been initialized by 
        `initialize_coefficients(...)`; missing entries are hard errors and indicate a 
        sampler/initialization bug.


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
        assert params is not None, "DampedSpring.compute_losses requires params to look up train_coefs";
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
                    IC      : list[list[numpy.ndarray   | torch.Tensor]],
                    t_Grid  : list[numpy.ndarray        | torch.Tensor],
                    params  : numpy.ndarray,
                    sample  : bool = False) -> list[list[numpy.ndarray | torch.Tensor]]:
        r"""
        Time integrates the latent dynamics from multiple initial conditions for each combination
        of parameter values.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

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

        params: numpy.ndarray, shape = (n_param, n_p)
            The i'th row holds the i'th combination of parameter values.

        sample : bool 
            If self is stochastic, setting this to true will sample from the posterior distribution 
            of the latent dynamics at each parameter value, then solve the latent dynamics using 
            the resulting sample. Otherwise, setting this to true will use the mean of that 
            posterior distribution. If self is not stochastic, this does nothing.

        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Z : list[list[numpy.ndarray]] or list[list[torch.Tensor]], len = n_parm
            i'th element is a list of length n_IC whose j'th entry is a 3d array of shape
            (n_t(i), n(i), n_z). The p, q, r entry of this array should hold the r'th component of
            the p'th frame of the j'th tine derivative of the solution to the latent dynamics when
            we use the q'th initial condition for the i'th combination of parameter values.
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
            assert isinstance(ith_coefs, dict) and set(ith_coefs.keys()) == {"K", "C", "b"};
            assert isinstance(ith_IC, list) and len(ith_IC) == 2;
            assert len(ith_t_Grid.shape) == 1 or len(ith_t_Grid.shape) == 2;
            if(isinstance(ith_t_Grid, torch.Tensor)):
                ith_t_Grid = ith_t_Grid.detach().cpu().numpy();
            Same_t_Grid : bool = (len(ith_t_Grid.shape) == 1);
            ith_D0 : numpy.ndarray | torch.Tensor = ith_IC[0];
            ith_V0 : numpy.ndarray | torch.Tensor = ith_IC[1];
            n_i    : int                          = ith_D0.shape[0];
            assert len(ith_D0.shape) == 2 and ith_D0.shape[1] == self.n_z;
            assert len(ith_V0.shape) == 2 and ith_V0.shape[1] == self.n_z;
            assert ith_D0.shape == ith_V0.shape;
            if(Same_t_Grid == False):
                assert ith_t_Grid.shape[0] == n_i;

            # Fetch native coefficients and match their backend/device/dtype to the initial condition.
            K = ith_coefs["K"];
            C = ith_coefs["C"];
            b = ith_coefs["b"];

            # Set up a lambda function to approximate (d^2/dt^2)z(t) \approx K z(t) + C (d/dt)z(t) + b.
            # In this case, we expect dz_dt and z to have shape (n(i), n_z). Thus, matmul(z, K.T) will
            # have shape (n(i), n_z). The i'th row of this should hold the z portion of the rhs of the
            # latent dynamics for the i'th IC. Similar results hold for dot(dz_dt, C.T). The final
            # result should have shape (n(i), n_z). The i'th row should hold the rhs of the latent
            # dynamics for the i'th IC.
            if(isinstance(ith_D0, numpy.ndarray)):
                if isinstance(K, torch.Tensor):
                    K = K.detach().cpu().numpy();
                    C = C.detach().cpu().numpy();
                    b = b.detach().cpu().numpy();
                b = b.reshape(1, -1);
                f   = lambda t, z, dz_dt: numpy.matmul(z, K.T) + numpy.matmul(dz_dt, C.T) + b;
            else:
                if isinstance(K, numpy.ndarray):
                    K = torch.tensor(K, dtype = ith_D0.dtype, device = ith_D0.device);
                    C = torch.tensor(C, dtype = ith_D0.dtype, device = ith_D0.device);
                    b = torch.tensor(b, dtype = ith_D0.dtype, device = ith_D0.device);
                else:
                    K = K.to(device = ith_D0.device, dtype = ith_D0.dtype);
                    C = C.to(device = ith_D0.device, dtype = ith_D0.dtype);
                    b = b.to(device = ith_D0.device, dtype = ith_D0.dtype);
                b = b.reshape(1, -1);
                f   = lambda t, z, dz_dt: torch.matmul(z, K.T) + torch.matmul(dz_dt, C.T) + b;

            # Solve the ODE forward in time. D and V should have shape (n_t, n(i), n_z). If we use the
            # same t values for each IC, then we can exploit the fact that the latent dynamics are
            # autonomous to solve using each IC simultaneously. Otherwise, we need to run the latent
            # dynamics one IC at a time.
            if(Same_t_Grid == True):
                ith_D, ith_V = RK4(f = f, y0 = ith_D0, Dy0 = ith_V0, t_Grid = ith_t_Grid);  # shape = (n_t, n_i, n_z)
            else:
                # Cycle through the ICs.
                ith_D_list : list[torch.Tensor | numpy.ndarray] = [];
                ith_V_list : list[torch.Tensor | numpy.ndarray] = [];

                for j in range(n_i):
                    D_ij, V_ij = RK4(f = f, y0 = ith_D0[j, :].reshape(1, -1), Dy0 = ith_V0[j, :].reshape(1, -1), t_Grid = ith_t_Grid[j, :]);
                    ith_D_list.append(D_ij);
                    ith_V_list.append(V_ij);

                # Stack the results.
                if(isinstance(ith_D0, numpy.ndarray)):
                    ith_D = numpy.concatenate(ith_D_list, axis = 1);    # shape = (n_t, n_i, n_z)
                    ith_V = numpy.concatenate(ith_V_list, axis = 1);    # shape = (n_t, n_i, n_z)
                else:
                    ith_D = torch.cat(ith_D_list, dim = 1);             # shape = (n_t, n_i, n_z)
                    ith_V = torch.cat(ith_V_list, dim = 1);             # shape = (n_t, n_i, n_z)

            # Add this parameter's trajectory to the output list.
            Z.append([ith_D, ith_V]);

        # All done!
        return Z;
