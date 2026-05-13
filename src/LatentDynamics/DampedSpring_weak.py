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
from    FiniteDifference    import  Derivative1_Order4, Derivative2_Order4, Derivative1_Order2_NonUniform, Derivative2_Order2_NonUniform;
from    SecondOrderSolvers  import  RK4;


# Setup Logger.
LOGGER : logging.Logger = logging.getLogger(__name__);



# -------------------------------------------------------------------------------------------------
# DampedSpring class
# -------------------------------------------------------------------------------------------------

class DampedSpring_weak(LatentDynamics):
    def __init__(   self, 
                    n_z             :   int, 
                    Uniform_t_Grid  :   bool,
                    config          :   dict) -> None:
        r"""
        Initializes a DampedSpring_weak object. This is a subclass of the LatentDynamics class which 
        implements the following latent dynamics
        
                z''(t) = K z(t) + C z'(t) + b
        
        Here, z is the latent state. K \in \mathbb{R}^{n x n} represents a generalized spring 
        matrix, C represents a damping matrix, and b is an offset/constant forcing function. 
        In this expression, K, C, and b are the model's coefficients. There is a separate set of
        coefficients for each combination of parameter values. 
            

        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        n_z : int
            The number of dimensions in the latent space, where the latent dynamics takes place.
            frame corresponds to time t0, the second to t0 + h, the k'th to t0 + (k - 1)h, etc 
            (note that h may depend on the parameter value, but it needs to be constant for a 
            specific parameter value). The value of this setting determines which finite difference 
            method we use to compute time derivatives. 

        config : dict 
            The "latent_dynamics" sub-dictionary of the config file. It should include the 
            following keys:
                - test_func: Specifies the kind of bump function. Either "bump" or "PC-poly" 
                - test_func_width: The width of each bump.
                - overlap: The amount of overlap between successive bumps.
                - pq: Only required if test_fun = "PC-poly". This should specify the order of the 
                polynomials?
            
            It may also include an optional "lstsq_reg" key, which specifies the ridge-regression 
            regularization strength used when fitting coefficients from scratch (i.e., when no 
            input_coefs are supplied to calibrate). Replaces plain lstsq with the 
            Tikhonov-regularized normal equations  (A^T A + λI) c = A^T b  where A = [Z_D, Z_V, 1]
            and b = d²Z/dt². Setting lstsq_reg = 0 falls back to plain least squares.

            
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------
        
        Nothing!
        """

        # Checks
        assert 'type' in config;
        assert config['type'] == "spring_w";
        assert "spring_w" in config;

        # Run the base class initializer. The only thing this does is set the n_z and n_t 
        # attributes.;
        super().__init__(n_z = n_z, Uniform_t_Grid = Uniform_t_Grid, config = config);
        self.lstsq_reg : float = config.get("lstsq_reg", 1.0);
        LOGGER.info("Initializing a DampedSpring_weak object with n_z = %d, Uniform_t_Grid = %s, lstsq_reg = %s" % (
            self.n_z,
            str(self.Uniform_t_Grid),
            str(self.lstsq_reg),
        ));
        
        # Set n_coefs and n_IC.
        # Because K and C are n_z x n_z matrices, and b is in \mathbb{R}^n_z, there are 
        # n_z*(2*n_z + 1) coefficients in the latent dynamics.
        self.n_IC       : int   = 2;
        self.n_coefs    : int   = n_z*(2*n_z + 1);

        # Setup the loss function.
        self.MSE = torch.nn.MSELoss(reduction = 'mean');
        self.MAE = torch.nn.L1Loss(reduction = 'mean');

        # Weak-form weight functions (set by the Trainer).
        # These are dictionaries keyed by a parameter tuple (p0, p1, ..., p_{n_p-1}).
        self.Phis_by_param   : dict[tuple, torch.Tensor] | None = None;
        self.dPhis_by_param  : dict[tuple, torch.Tensor] | None = None;
        self.d2Phis_by_param : dict[tuple, torch.Tensor] | None = None;

        # Set weak form specific settings.
        self.test_func          = config['spring_w']['test_func'];
        self.test_func_width    = config['spring_w']['test_func_width'];
        self.overlap            = config['spring_w']['overlap'];
        # Only required for PC-poly.
        self.pq                 = config['spring_w'].get('pq', None);
        self.LS_loss_type       = config['spring_w']['LS_loss_type'];
        return;




    def _native_from_matrix(self, coefs : torch.Tensor) -> dict[str, torch.Tensor]:
        r"""
        Convert a weak-form least-squares coefficient matrix to native trainable tensors.

        The weak normal equations still solve for the same library coefficients as the strong-form
        damped spring model. We store them directly as K, C, and b.
        """

        assert coefs.shape == (2*self.n_z + 1, self.n_z);
        K = coefs[0:self.n_z, :].T.detach().clone().requires_grad_(True);
        C = coefs[self.n_z:(2*self.n_z), :].T.detach().clone().requires_grad_(True);
        b   = coefs[2*self.n_z, :].detach().clone().requires_grad_(True);
        return {"K": K, "C": C, "b": b};



    def _matrix_from_native(self, coefs : dict[str, torch.Tensor]) -> torch.Tensor:
        r"""Reconstruct the weak-form library matrix [K^T; C^T; b]."""

        K = coefs["K"];
        C = coefs["C"];
        b   = coefs["b"];
        return torch.cat([K.T, C.T, b.reshape(1, self.n_z)], dim = 0);



    def train_coef_tensors(self) -> list[torch.Tensor]:
        r"""Return the actual weak-form coefficient tensors to optimize."""

        tensors : list[torch.Tensor] = [];
        for coef_dict in self.train_coefs.values():
            tensors.extend([coef_dict["K"], coef_dict["C"], coef_dict["b"]]);
        return tensors;



    def flatten_coefficients(self, coefs : dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]) -> numpy.ndarray:
        r"""Flatten weak-form damped-spring coefficients in the legacy [K^T; C^T; b] ordering."""

        coefs_list = [coefs] if isinstance(coefs, dict) else coefs;
        assert isinstance(coefs_list, list);
        rows : list[numpy.ndarray] = [];
        for coef_dict in coefs_list:
            mat = self._matrix_from_native(coef_dict);
            rows.append(mat.detach().cpu().numpy().reshape(1, -1));
        return numpy.concatenate(rows, axis = 0);



    # ---------------------------------------------------------------------------------------------
    # Weak-form weight functions
    # ---------------------------------------------------------------------------------------------

    def set_weight_functions(self,
                             Phis_by_param  : dict[tuple, torch.Tensor],
                             dPhis_by_param : dict[tuple, torch.Tensor],
                             d2Phis_by_param: dict[tuple, torch.Tensor]) -> None:
        """
        Store the weak-form test/weight functions internally.

        The intended workflow is:
            trainer builds (Phis, dPhis, d2Phis) for all relevant parameter combinations
            -> calls latent_dynamics.set_weight_functions(...)
            -> calls latent_dynamics.calibrate(...) (which looks them up by param tuple)
        """
        assert isinstance(Phis_by_param, dict) and isinstance(dPhis_by_param, dict) and isinstance(d2Phis_by_param, dict);
        assert set(Phis_by_param.keys()) == set(dPhis_by_param.keys()) == set(d2Phis_by_param.keys()), (
            "Weight function dictionaries must have identical key sets");

        # Lightweight validation of shapes/types per key.
        for k in Phis_by_param.keys():
            Phi  = Phis_by_param[k];
            dPhi = dPhis_by_param[k];
            d2Phi= d2Phis_by_param[k];
            assert isinstance(Phi, torch.Tensor),  "Phis_by_param[%s] must be a torch.Tensor" % str(k);
            assert isinstance(dPhi, torch.Tensor), "dPhis_by_param[%s] must be a torch.Tensor" % str(k);
            assert isinstance(d2Phi, torch.Tensor),"d2Phis_by_param[%s] must be a torch.Tensor" % str(k);
            assert Phi.shape == dPhi.shape == d2Phi.shape, "Phi shapes must match for key %s" % str(k);

        self.Phis_by_param  = Phis_by_param;
        self.dPhis_by_param = dPhis_by_param;
        self.d2Phis_by_param= d2Phis_by_param;
        return;



    def getUniformGrid(self, T : float, L : float, s : float, p : int):
        """
        Generates a uniform grid of support intervals for compactly-supported test functions.

        T : final time
        L : test-function support width
        s : overlap amount between adjacent supports
        p : unused legacy argument kept for backward compatibility
        """

        overlap = s;
        grid = [];
        a = 0.0;
        b = float(L);
        grid.append([a, b]);
        while (b - overlap + L) <= T:
            a = b - overlap;
            b = a + L;
            grid.append([a, b]);

        grid = numpy.asarray(grid, dtype = numpy.float64);
        a_s = grid[:, 0];
        b_s = grid[:, 1];
        return a_s, b_s;



    def get_test_functions(self,
                           T               : float,
                           n_t             : int,
                           timesteps       : torch.Tensor,
                           H               : int = 30) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build compactly-supported weak-form test functions and their first/second derivatives.

        Returns tensors of shape (H, n_t).
        """

        assert isinstance(timesteps, torch.Tensor), "timesteps must be a torch.Tensor";
        assert timesteps.ndim == 1, "timesteps must be a 1D tensor";
        assert int(timesteps.shape[0]) == int(n_t), "n_t does not match timesteps length";

        t       : torch.Tensor  = timesteps;
        dtype   = t.dtype;
        device  = t.device;

        if self.test_func == 'bump':
            L = float(self.test_func_width);
            s = float(self.test_func_width) * float(self.overlap);
            a_s, b_s = self.getUniformGrid(float(T), L, s, 1);

            H_eff    : int = len(a_s);
            LOGGER.info("Number of bump test functions: %d" % H_eff);

            Phis    = torch.zeros((H_eff, n_t), dtype = dtype, device = device);
            dPhis   = torch.zeros((H_eff, n_t), dtype = dtype, device = device);
            d2Phis  = torch.zeros((H_eff, n_t), dtype = dtype, device = device);

            eta     : float = 5.0;
            a       : float = L / 2.0;
            const   : float = eta;
            nugget  : float = 1.0e-7;
            a_space = numpy.linspace(-a + nugget, a - nugget, 1000);
            bump    = numpy.exp(-eta / (1.0 - (a_space / a) ** 2));
            C       : float = float(1.0 / numpy.trapz(bump, a_space) / numpy.exp(const));

            h = torch.linspace(a, float(T) - a, H_eff, dtype = dtype, device = device);
            for j, ji in enumerate(h):
                for i, ti in enumerate(t):
                    x       = (ti - ji) / a;
                    denom   = 1.0 - x ** 2;
                    f       = -eta / denom + const;
                    fp      = -eta / (denom ** 2) * 2.0 * x / a;
                    fpp     = (-eta / (denom ** 2) * 2.0 / (a * a)) + (-eta / (denom ** 3) * 2.0 * x / a * -2.0 * x / a * -2.0);
                    if denom > 0:
                        expf            = torch.exp(f);
                        Phis[j, i]      = C * expf;
                        dPhis[j, i]     = C * expf * fp;
                        d2Phis[j, i]    = C * (expf * (fp ** 2) + expf * fpp);

        elif self.test_func == 'PC-poly':
            assert self.pq is not None, "spring_w.pq must be provided in config when test_func == 'PC-poly'";
            L = float(self.test_func_width);
            s = float(self.test_func_width) * float(self.overlap);
            a_s, b_s = self.getUniformGrid(float(T), L, s, 1);

            H_eff    : int = len(a_s);
            LOGGER.info("Number of PC-poly test functions: %d" % H_eff);

            Phis    = torch.zeros((H_eff, n_t), dtype = dtype, device = device);
            dPhis   = torch.zeros((H_eff, n_t), dtype = dtype, device = device);
            d2Phis  = torch.zeros((H_eff, n_t), dtype = dtype, device = device);

            p, q = self.pq, self.pq;
            for h in range(H_eff):
                a = float(a_s[h]);
                b = float(b_s[h]);
                C = 1.0 / (p ** p * q ** q) * ((p + q) / (b - a)) ** (p + q);
                mask = (t >= a) * (t <= b);
                Phis[h, :] = C * (t - a) ** p * (b - t) ** q * mask;
                dPhis[h, :] = C * (p * (t - a) ** (p - 1) * (b - t) ** q - q * (t - a) ** p * (b - t) ** (q - 1)) * mask;
                d2Phis[h, :] = C * (
                    p * (p - 1) * (t - a) ** (p - 2) * (b - t) ** q
                    - q * p * (t - a) ** (p - 1) * (b - t) ** (q - 1)
                    - q * p * (t - a) ** (p - 1) * (b - t) ** (q - 1)
                    + q * (q - 1) * (t - a) ** p * (b - t) ** (q - 2)
                ) * mask;

        else:
            raise ValueError("Unsupported weak-form test function type: %s" % str(self.test_func));

        return Phis, dPhis, d2Phis;



    def _ensure_weight_functions(self,
                                 t_Grid : list[torch.Tensor],
                                 params : numpy.ndarray) -> None:
        """
        Ensure weak-form weight functions exist for each parameter combination in `params`.

        If a Trainer has already called `set_weight_functions(...)`, this is a no-op. Otherwise,
        we generate and store the test functions for missing keys using the weak-form settings
        stored from `config` at initialization time.
        """
        if self.Phis_by_param is None:
            self.Phis_by_param = {};
            self.dPhis_by_param = {};
            self.d2Phis_by_param = {};

        assert self.dPhis_by_param is not None and self.d2Phis_by_param is not None;

        for i in range(params.shape[0]):
            key = self._param_key(params[i, :]);
            if key in self.Phis_by_param:
                continue;

            assert self.test_func is not None and self.test_func_width is not None and self.overlap is not None, (
                "Missing weak-form config. Call set_weight_functions(...) before fitting coefficients.");
            if self.test_func == "PC-poly":
                assert self.pq is not None, "spring_w.pq must be provided in config when test_func == 'PC-poly'";

            t_i : torch.Tensor = t_Grid[i];
            T_i : float = float(t_i[-1].detach().cpu().item());
            Phi_i, dPhi_i, d2Phi_i = self.get_test_functions(
                T               = T_i,
                n_t             = int(t_i.shape[0]),
                timesteps       = t_i);

            self.Phis_by_param[key]  = Phi_i;
            self.dPhis_by_param[key] = dPhi_i;
            self.d2Phis_by_param[key]= d2Phi_i;

        return;



    # ---------------------------------------------------------------------------------------------
    # fit_coefficients
    # ---------------------------------------------------------------------------------------------

    def fit_coefficients(self,
                         Latent_States : list[list[torch.Tensor]],
                         t_Grid        : list[torch.Tensor],
                         params        : numpy.ndarray | None = None) -> None:
        r"""
        Fit coefficients for the weak-form damped-spring model using the weak-form normal
        equations.

        This is intended for coefficient initialization. Weight functions must be available
        either via `set_weight_functions(...)` (Trainer-provided) or they will be generated
        on-demand from the time grids using the weak-form settings stored from `config`.
        """
        assert params is not None, "DampedSpring_weak.fit_coefficients requires `params`";
        assert isinstance(t_Grid, list) and isinstance(Latent_States, list);
        assert len(Latent_States) == len(t_Grid) == params.shape[0];

        # Ensure weight functions exist for all requested parameter combinations.
        self._ensure_weight_functions(t_Grid = t_Grid, params = params);
        assert self.Phis_by_param is not None and self.dPhis_by_param is not None and self.d2Phis_by_param is not None;

        for i in range(params.shape[0]):
            key = self._param_key(params[i, :]);
            Phi  : torch.Tensor = self.Phis_by_param[key];
            dPhi : torch.Tensor = self.dPhis_by_param[key];
            d2Phi: torch.Tensor = self.d2Phis_by_param[key];

            Z      = Latent_States[i];
            Z_D    : torch.Tensor = Z[0];
            Z_V    : torch.Tensor = Z[1];

            ones    : torch.Tensor = torch.ones((Z_D.shape[0], 1), device = Z_D.device, dtype = Z_D.dtype);
            Theta   : torch.Tensor = torch.cat([Z_D, Z_V, ones], dim = 1);          # (n_t, 2*n_z+1)

            # Solve weak-form least squares system for vec(E^T), then reshape to E.
            Gk, bk = self.compute_Gk_bk(self.n_z, Phi.to(Theta), dPhi.to(Theta), d2Phi.to(Theta), Theta, [Z_D, Z_V]);
            n_lib      : int           = Gk.shape[1];
            rhs        : torch.Tensor  = Gk.T @ bk;
            if self.lstsq_reg > 0.0:
                gram   : torch.Tensor  = Gk.T @ Gk + self.lstsq_reg * torch.eye(n_lib, device = Gk.device, dtype = Gk.dtype);
                coef_v : torch.Tensor  = torch.linalg.solve(gram, rhs);
            else:
                coef_v : torch.Tensor  = torch.linalg.lstsq(Gk, bk).solution;
            coefs = coef_v.reshape((self.n_z, Theta.shape[-1])).T;                  # (2*n_z+1, n_z)

            # Store the initialized weak-form coefficients in native LD-owned form.
            self.set_train_coefs(params[i, :], self._native_from_matrix(coefs));

        return None;





    # ---------------------------------------------------------------------------------------------
    # Calibrate
    # ---------------------------------------------------------------------------------------------

    def calibrate(self,
                  Latent_States : list[torch.Tensor],
                  loss_type     : str,
                  t_Grid        : list[torch.Tensor],
                  params        : numpy.ndarray | None = None) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        r"""
        For each combination of parameter values, this function computes the weak-form
        latent-dynamics loss using the K, C, and b coefficients stored in `self.train_coefs`.
        
        Specifically, let us consider the case when Z has two axes (the case when it has three is 
        identical, just with different coefficients for each instance of the leading dimension of 
        Z). In this case, we assume the i'th row of Z holds the latent state t_0 + i*dt. We use 
        We assume that the latent state is governed by an ODE of the form
        
                z''(t) = K z(t) + C z'(t) + b
        
        Coefficients are initialized by `fit_coefficients(...)` and then looked up directly from
        `self.train_coefs` using `params`. Missing entries are intentional hard errors because
        they indicate that the sampler/training-data path failed to initialize a training
        parameter.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        Latent_States : list[list[torch.Tensor]], len = n_param
            The i'th list element is a 2 element list whose j'th element is a 2d numpy array of 
            shape (n_t(i), n_z) whose p, q element holds the q'th component of the j'th derivative 
            of the latent state during the p'th time step (whose time value corresponds to the p'th 
            element of t_Grid) when we use the i'th combination of parameter values. 
        
        loss_type : str
            The type of loss function to use. Must be either "MSE" or "MAE".

        t_Grid : list[torch.Tensor], len = n_param
            i'th element should be a 1d tensor of shape (n_t(i)) whose j'th element holds the time 
            value corresponding to the j'th frame when we use the i'th combination of parameter 
            values.

        params: numpy.ndarray, shape = (n_param, n_p)
            The i'th row holds the i'th combination of parameter values. These rows are used to
            fetch weak-form test functions and the corresponding native coefficient dictionaries.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        loss_sindy, loss_coef, loss_stab.

        loss_sindy : list[torch.Tensor], len = n_param
            The i'th element of this list is a 0-dimensional tensor whose lone element holds the 
            weak-form latent-dynamics loss from the i'th combination of parameter values.

        loss_coef : list[torch.Tensor], len = n_para
            The i'th element of this list is a 0-dimensional tensor whose lone element holds the
            coefficient loss (Frobenius norm) of the coefficients for the i'th combination 
            of parameter values.  
        
        loss_stab : list[torch.Tensor], len = n_param
            The i'th element of this list is a 0-dimensional tensor whose lone element holds the
            stability penalty for the i'th combination of parameter values (see
            LatentDynamics.stability_penalty).
        """

        # Run checks.
        assert(isinstance(t_Grid, list));
        assert(isinstance(Latent_States, list));
        assert(len(Latent_States)   == len(t_Grid));

        n_param : int   = len(t_Grid);
        n_IC    : int   = 2;
        n_z     : int   = self.n_z;
        for i in range(n_param):
            assert(isinstance(Latent_States[i], list));
            assert(len(Latent_States[i]) == n_IC);

            for j in range(n_IC):
                assert(isinstance(Latent_States[i][j], torch.Tensor));
                assert(len(Latent_States[i][j].shape)   == 2);
                assert(Latent_States[i][j].shape[-1]    == n_z);

        # Run checks on loss_type.
        assert(loss_type in ["MSE", "MAE"]);

        assert params is not None, (
            "DampedSpring_weak requires `params` so it can look up (or generate) weight functions by parameter tuple.");
        self._ensure_weight_functions(t_Grid = t_Grid, params = params);
        assert self.Phis_by_param is not None and self.dPhis_by_param is not None and self.d2Phis_by_param is not None;


        # -----------------------------------------------------------------------------------------
        # If there are multiple combinations of parameter values, loop through them.
        # -----------------------------------------------------------------------------------------

        if (n_param > 1):
            loss_sindy_list : list[torch.Tensor] = [];
            loss_stab_list  : list[torch.Tensor] = [];
            loss_coef_list  : list[torch.Tensor] = [];

            for i in range(n_param):
                params_i = params[i, :].reshape(1, -1);
                
                # Calibrate on the i'th combination of parameter values.
                loss_sindy_i, loss_coef_i, loss_stab_i = self.calibrate(  Latent_States = [Latent_States[i]],
                                                                                        t_Grid        = [t_Grid[i]],
                                                                                        loss_type     = loss_type,
                                                                                        params        = params_i);

                # Package the results from this combination of parameter values.
                loss_sindy_list.append(loss_sindy_i[0]);
                loss_stab_list.append(loss_stab_i[0]);
                loss_coef_list.append(loss_coef_i[0]);
            
            return loss_sindy_list, loss_coef_list, loss_stab_list;
        


        # -----------------------------------------------------------------------------------------
        # Evaluate for one combination of parameter values case.
        # -----------------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------------
        # Concatenate the latent displacement and velocity.

        Z       : torch.Tensor  = Latent_States[0];         # len = n_IC, i'th element is a torch.Tensor of shape (n_t, n_z)
        t_Grid0 : torch.Tensor  = t_Grid[0];                # shape = (n_t)

        Z_D     : torch.Tensor  = Z[0];                     # shape = (n_t, n_z)
        Z_V     : torch.Tensor  = Z[1];                     # shape = (n_t, n_z)

        key0 : tuple = self._param_key(params[0, :]);
        assert key0 in self.Phis_by_param, "No weight functions found for params=%s (key=%s)" % (str(params[0, :]), str(key0));

        Phis    : torch.Tensor  = self.Phis_by_param[key0].to(device = Z_D.device, dtype = Z_D.dtype);
        dPhis   : torch.Tensor  = self.dPhis_by_param[key0].to(device = Z_D.device, dtype = Z_D.dtype);
        d2Phis  : torch.Tensor  = self.d2Phis_by_param[key0].to(device = Z_D.device, dtype = Z_D.dtype);

        # Concatenate Z_D, Z_V and a column of 1's. We will solve for the matrix, E, which gives 
        # the best fit for the system d2Z_dt2 = cat[Z_D, Z_V, 1] E. This matrix has the form 
        # E^T = [K, C, b]. Thus, we can extract K, C, and b from Z_1.
        ones      : torch.Tensor = torch.ones((Z_D.shape[0], 1), device = Z_D.device, dtype = Z_D.dtype);
        ZD_ZV_1   : torch.Tensor = torch.cat([Z_D, Z_V, ones], dim = 1);          # shape = (n_t, 2*n_z + 1)


        # -----------------------------------------------------------------------------------------
        # Compute the second time derivative of the latent state.

        # if(self.Uniform_t_Grid  == True):
        #     h : float = (t_Grid0[1] - t_Grid0[0]).item();
        #     #d2Z_dt2_from_Z_D    : torch.Tensor  = Derivative2_Order4(U = Z_D,   h = h);                     # shape = (n_t, n_z)
        #     d2Z_dt2_from_Z_V    : torch.Tensor  = Derivative1_Order4(U = Z_V,   h = h);                     # shape = (n_t, n_z)
        # else:
        #     #d2Z_dt2_from_Z_D                    = Derivative2_Order2_NonUniform(U = Z_D, t_Grid = t_Grid0);  # shape = (n_t, n_z)
        #     d2Z_dt2_from_Z_V                    = Derivative1_Order2_NonUniform(U = Z_V, t_Grid = t_Grid0);  # shape = (n_t, n_z)
        # d2Z_dt2             : torch.Tensor  = d2Z_dt2_from_Z_V #0.5*(d2Z_dt2_from_Z_D + d2Z_dt2_from_Z_V);  # shape = (n_t, n_z)

        if(self.Uniform_t_Grid  == True):
            h : float = (t_Grid0[1] - t_Grid0[0]).item();
            d2Z_dt2_from_Z_D    : torch.Tensor  = Derivative2_Order4(U = Z_D,   h = h);                     # shape = (n_t, n_z)
            d2Z_dt2_from_Z_V    : torch.Tensor  = Derivative1_Order4(U = Z_V,   h = h);                     # shape = (n_t, n_z)
            d2Z_dt2             : torch.Tensor  = 0.5*(d2Z_dt2_from_Z_D + d2Z_dt2_from_Z_V);  # shape = (n_t, n_z)
        else:
            d2Z_dt2_from_Z_D                    = Derivative2_Order2_NonUniform(U = Z_D, t_Grid = t_Grid0);  # shape = (n_t, n_z)
            d2Z_dt2_from_Z_V                    = Derivative1_Order2_NonUniform(U = Z_V, t_Grid = t_Grid0);  # shape = (n_t, n_z)
            # d2Z_dt2             : torch.Tensor  = d2Z_dt2_from_Z_V #0.5*(d2Z_dt2_from_Z_D + d2Z_dt2_from_Z_V);  # shape = (n_t, n_z)
            d2Z_dt2             : torch.Tensor  = 0.5*(d2Z_dt2_from_Z_D + d2Z_dt2_from_Z_V);  # shape = (n_t, n_z)


        # -----------------------------------------------------------------------------------------
        # Set up coefs using the provided coefficients.

        # Fetch native trainable coefficients for this parameter. Missing entries intentionally
        # raise KeyError because coefficient initialization should have happened in the sampler.
        coef_dict = self.get_train_coefs(params[0, :]);
        K = coef_dict["K"].to(device = Z_D.device, dtype = Z_D.dtype);
        C = coef_dict["C"].to(device = Z_D.device, dtype = Z_D.dtype);
        b   = coef_dict["b"].to(device = Z_D.device, dtype = Z_D.dtype);
        coefs = torch.cat([K.T, C.T, b.reshape(1, self.n_z)], dim = 0);
    
        LD_RHS = torch.matmul(Z_D, K.T) + torch.matmul(Z_V, C.T) + b.reshape(1, -1);

        # Compute the weak residual used for the latent-dynamics loss.
        # weak_LHS    : torch.Tensor = 0.5 * (torch.matmul(d2Phis, Z_D) - torch.matmul(dPhis, Z_V));
        # weak_LHS    : torch.Tensor =  - torch.matmul(dPhis, Z_V);
        lhs_D = torch.matmul(d2Phis, Z_D)
        lhs_V = -torch.matmul(dPhis, Z_V)
        weak_RHS    : torch.Tensor = torch.matmul(torch.matmul(Phis, ZD_ZV_1), coefs);

        # -----------------------------------------------------------------------------------------
        # Compute the stability losses and return.

        scale_D = torch.linalg.norm(d2Phis, dim=1, keepdim=True).clamp(min = 1.0e-10);
        scale_V = torch.linalg.norm(dPhis,  dim=1, keepdim=True).clamp(min = 1.0e-10);

        if loss_type == "MSE":
            loss_D = self.MSE(lhs_D / scale_D, weak_RHS / scale_D)
            loss_V = self.MSE(lhs_V / scale_V, weak_RHS / scale_V)
        elif loss_type == "MAE":
            loss_D = self.MAE(lhs_D / scale_D, weak_RHS / scale_D)
            loss_V = self.MAE(lhs_V / scale_V, weak_RHS / scale_V)

        Loss_LD = 0.5 * loss_D + 0.5 * loss_V

        # if(loss_type == "MSE"):
        #     Loss_LD     = self.MSE(weak_LHS, weak_RHS);
        # elif(loss_type == "MAE"):
        #     Loss_LD     = self.MAE(weak_LHS, weak_RHS);

        # if(loss_type == "MSE"):
        #     Loss_LD     = self.MSE(d2Z_dt2, LD_RHS);
        # elif(loss_type == "MAE"):
        #     Loss_LD     = self.MAE(d2Z_dt2, LD_RHS);

        # Stability penalty on the equivalent first-order system y' = A y (+ f).
        # For z'' = K z + C z' + b, define y = [z, z'] so A = [[0, I], [K, C]].
        Z0  : torch.Tensor  = torch.zeros((self.n_z, self.n_z), device = coefs.device, dtype = coefs.dtype);
        I   : torch.Tensor  = torch.eye(self.n_z, device = coefs.device, dtype = coefs.dtype);
        A_top    = torch.cat([Z0, I], dim = 1);
        A_bottom = torch.cat([K, C], dim = 1);
        A = torch.cat([A_top, A_bottom], dim = 0);
        Loss_Stab = self.stability_penalty(A);

        # Compute coefficient loss.
        Loss_coef = torch.norm(K, 'fro') + torch.norm(C, 'fro') + torch.norm(b);

        return [Loss_LD], [Loss_coef], [Loss_Stab];
    


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
            Native coefficient dictionary/dictionaries. For DampedSpring_weak each dictionary
            must contain `K` with shape (n_z, n_z), `C` with shape (n_z, n_z), and `b` with shape
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

        # Normalize coefficient input to a list so the multi-parameter and single-parameter cases
        # share the same validation and recursion logic.
        if isinstance(coefs, dict):
            coefs_list = [coefs];
        else:
            coefs_list = coefs;
        assert isinstance(coefs_list, list);
        n_param : int = len(coefs_list);
        assert isinstance(t_Grid, list);
        assert isinstance(IC, list);
        assert len(IC)     == n_param;
        assert len(t_Grid) == n_param;

        assert isinstance(IC[0], list);
        n_IC : int = len(IC[0]);
        assert n_IC == 2;
        for i in range(n_param):
            assert isinstance(coefs_list[i], dict);
            assert set(coefs_list[i].keys()) == {"K", "C", "b"};
            assert isinstance(IC[i], list);
            assert len(IC[i]) == n_IC;
            assert len(t_Grid[i].shape) == 2 or len(t_Grid[i].shape) == 1;
            for j in range(n_IC):
                assert len(IC[i][j].shape) == 2;
                assert IC[i][j].shape[1] == self.n_z;
                if(len(t_Grid[i].shape) == 2):
                    assert t_Grid[i].shape[0] == IC[i][j].shape[0];


        # -----------------------------------------------------------------------------------------
        # If there are multiple combinations of parameter values, loop through them.

        # This function behaves differently if there is one set of coefficients or multiple of them.
        if(n_param > 1):
            LOGGER.debug("Simulating with %d parameter combinations" % n_param);

            # Cycle through the parameter combinations
            Z   : list[list[numpy.ndarray | torch.Tensor]]  = [];
            for i in range(n_param): 
                # Fetch the i'th set of coefficients, the corresponding collection of initial
                # conditions, and the set of time values.
                ith_coefs   : dict[str, numpy.ndarray | torch.Tensor]       = coefs_list[i];
                ith_IC      : list[list[numpy.ndarray   | torch.Tensor]]    = [IC[i]];
                ith_t_Grid  : list[numpy.ndarray        | torch.Tensor]     = [t_Grid[i]];
                ith_params  = None if params is None else params[i, :].reshape(1, -1);

                # Call this function using them. This should return a 2 element holding the 
                # displacement and velocity of the solution for the i'th combination of 
                # parameter values.
                ith_Results : list[numpy.ndarray | torch.Tensor]    = self.simulate(coefs   = ith_coefs, 
                                                                                    IC      = ith_IC, 
                                                                                    t_Grid  = ith_t_Grid, 
                                                                                    params  = ith_params)[0];

                # Add these results to Z.
                Z.append(ith_Results);

            # All done.
            return Z;


        # -----------------------------------------------------------------------------------------
        # Evaluate for one combination of parameter values case.

        # In this case, there is just one parameter. Extract t_Grid, which has shape 
        # (n(i), n_t(i)) or (n_t(i)).
        t_Grid0  : numpy.ndarray | torch.Tensor  = t_Grid[0];
        if(isinstance(t_Grid0, torch.Tensor)):
            # Support CUDA/MPS tensors by moving to CPU before NumPy conversion.
            t_Grid0 = t_Grid0.detach().cpu().numpy();
        n_t_i   : int           = t_Grid0.shape[-1];
        if(len(t_Grid0.shape) == 1):
            Same_t_Grid : bool  = True;
        else:
            Same_t_Grid         = False;
        
        # Each element of IC should have shape (n(i), n_z). 
        D0  : numpy.ndarray | torch.Tensor  = IC[0][0]; 
        V0  : numpy.ndarray | torch.Tensor  = IC[0][1];
        n_i : int                           = D0.shape[0];

        # Fetch native coefficients and match their backend/device/dtype to the initial condition.
        coef_dict = coefs_list[0];
        K = coef_dict["K"];
        C = coef_dict["C"];
        b = coef_dict["b"];

        # Set up a lambda function to approximate (d^2/dt^2)z(t) \approx K z(t) + C (d/dt)z(t) + b.
        # In this case, we expect dz_dt and z to have shape (n(i), n_z). Thus, matmul(z, K.T) will 
        # have shape (n(i), n_z). The i'th row of this should hold the z portion of the rhs of the 
        # latent dynamics for the i'th IC. Similar results hold for dot(dz_dt, C.T). The final 
        # result should have shape (n(i), n_z). The i'th row should hold the rhs of the latent 
        # dynamics for the i'th IC.
        if(isinstance(D0, numpy.ndarray)):
            if isinstance(K, torch.Tensor):
                K = K.detach().cpu().numpy();
                C = C.detach().cpu().numpy();
                b = b.detach().cpu().numpy();
            b = b.reshape(1, -1);
            f   = lambda t, z, dz_dt: numpy.matmul(z, K.T) + numpy.matmul(dz_dt, C.T) + b;
        else:
            if isinstance(K, numpy.ndarray):
                K = torch.tensor(K, dtype = D0.dtype, device = D0.device);
                C = torch.tensor(C, dtype = D0.dtype, device = D0.device);
                b = torch.tensor(b, dtype = D0.dtype, device = D0.device);
            else:
                K = K.to(device = D0.device, dtype = D0.dtype);
                C = C.to(device = D0.device, dtype = D0.dtype);
                b = b.to(device = D0.device, dtype = D0.dtype);
            b = b.reshape(1, -1);
            f   = lambda t, z, dz_dt: torch.matmul(z, K.T) + torch.matmul(dz_dt, C.T) + b;

        # Solve the ODE forward in time. D and V should have shape (n_t, n(i), n_z). If we use the 
        # same t values for each IC, then we can exploit the fact that the latent dynamics are 
        # autonomous to solve using each IC simultaneously. Otherwise, we need to run the latent
        # dynamics one IC at a time. 
        if(Same_t_Grid == True):
            D, V = RK4(f = f, y0 = D0, Dy0 = V0, t_Grid = t_Grid0);  # shape = (n_t, n_i, n_z)
        else:
            # Cycle through the ICs.
            D_list : list[torch.Tensor | numpy.ndarray] = [];
            V_list : list[torch.Tensor | numpy.ndarray] = []; 

            for j in range(n_i):
                D_j, V_j    = RK4(f = f, y0 = D0[j, :].reshape(1, -1), Dy0 = V0[j, :].reshape(1, -1), t_Grid = t_Grid0[j, :]);
                D_list.append(D_j);
                V_list.append(V_j);

            # Stack the results.
            if(isinstance(D0, numpy.ndarray)):
                D = numpy.concatenate(D_list, axis = 1);    # shape = (n_t, n_i, n_z)
                V = numpy.concatenate(V_list, axis = 1);    # shape = (n_t, n_i, n_z)
            else:
                D = torch.cat(D_list, dim = 1);            # shape = (n_t, n_i, n_z)
                V = torch.cat(V_list, dim = 1);            # shape = (n_t, n_i, n_z)
        
        # All done!
        return [[D, V]];


    def compute_Gk_bk(self, n_s,Phi,dPhi,d2Phi,Theta,Zs):


        r'''
        n_s: reduced dim
        Phi: dimension (H,n_T)
        dPhi: dimension (H,n_T)
        Theta: dimension (n_T, J)
        Gk = I_{n_s} \otimes \Phi \Theta :  dimension (H*n_s,J*n_s)
        bk = -vec(dPhi*U): dimension (H*n_s)
        '''

        H = Phi.shape[0];
        J = Theta.shape[1];

        Ins         = torch.eye(n_s, device = Theta.device, dtype = Theta.dtype);
        PhiTheta    = Phi @ Theta;
        # Gk          = torch.kron(Ins, PhiTheta);

        # bk = 0.5 * (d2Phi @ Zs[0] - dPhi @ Zs[1]);
        # bk = bk.permute(1, 0).reshape((H * n_s, 1));

        lhs_D = d2Phi @ Zs[0]
        lhs_V = -(dPhi @ Zs[1])

        scale_D = torch.linalg.norm(d2Phi, dim=1, keepdim=True).clamp(min = 1.0e-10)
        scale_V = torch.linalg.norm(dPhi,  dim=1, keepdim=True).clamp(min = 1.0e-10)

        Gk_D = torch.kron(Ins, PhiTheta / scale_D)
        Gk_V = torch.kron(Ins, PhiTheta / scale_V)

        bk_D = (lhs_D / scale_D).permute(1, 0).reshape(H * n_s, 1)
        bk_V = (lhs_V / scale_V).permute(1, 0).reshape(H * n_s, 1)

        Gk = torch.cat([Gk_D, Gk_V], dim=0)
        bk = torch.cat([bk_D, bk_V], dim=0)

        return Gk, bk
