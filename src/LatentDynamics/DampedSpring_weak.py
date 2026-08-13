# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

import  logging;

import  numpy;
import  torch;

from    LatentDynamics.Weak             import  WeakLatentDynamics;
from    LatentDynamics.Interpolatable   import  InterpolatableLatentDynamics;
from    Utilities.SecondOrderSolvers    import  RK4;


# Setup Logger.
LOGGER : logging.Logger = logging.getLogger(__name__);



# -------------------------------------------------------------------------------------------------
# DampedSpring class
# -------------------------------------------------------------------------------------------------

class DampedSpring_weak(InterpolatableLatentDynamics, WeakLatentDynamics):
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
            The latent-dynamics configuration dictionary. It must three keys: `type`, `trainable`,
            and `spring_w`. It must have `config["type"] == "spring_w"` and `config["spring_w"]` 
            should be a weak-form sub-dictionary containing the following keys:
                - test_func_type: Specifies the kind of bump function. Either "bump" or "PC-poly".
                - test_func_width: The width of each bump.
                - overlap: The amount of overlap between successive bumps.
            
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
        assert  config['type'] == "spring_w";
        assert  "spring_w"  in config;

        # Run the base class initializer. This does not set the n_t attribute. 
        # Because K and C are n_z x n_z matrices, and b is in \mathbb{R}^n_z, there are 
        # n_z*(2*n_z + 1) coefficients in the latent dynamics.
        InterpolatableLatentDynamics.__init__(   
            self,
            n_z             = n_z,
            n_coefs         = n_z*(2*n_z + 1),
            n_IC            = 2, 
            Uniform_t_Grid  = Uniform_t_Grid, 
            trainable       = config["trainable"],
            config          = config);

        WeakLatentDynamics.__init__(   
            self,
            n_z             = n_z,
            n_coefs         = n_z*(2*n_z + 1),
            n_IC            = 2, 
            Uniform_t_Grid  = Uniform_t_Grid, 
            trainable       = config["trainable"],
            config          = config);

        
        LOGGER.info("Initializing a DampedSpring_weak object with n_z = %d, Uniform_t_Grid = %s" % (
            self.n_z,
            str(self.Uniform_t_Grid),
        ));

        # Setup the loss function.
        self.MSE = torch.nn.MSELoss(reduction = 'mean');
        self.MAE = torch.nn.L1Loss(reduction = 'mean');

        return;


    def trainable_tensors(self) -> list[torch.Tensor]:
        r"""Return the actual weak-form coefficient tensors to optimize."""

        if self.trainable == False:
            return [];
    
        tensors : list[torch.Tensor] = [];
        for coef_dict in self.train_coefs.values():
            tensors.extend([coef_dict["K"], coef_dict["C"], coef_dict["b"]]);
        return tensors;



    # ---------------------------------------------------------------------------------------------
    # initialize_coefficients
    # ---------------------------------------------------------------------------------------------

    def initialize_coefficients(
            self,
            Latent_States : list[list[torch.Tensor]],
            t_Grid        : list[torch.Tensor],
            device        : torch.device,
            params        : numpy.ndarray) -> None:
        r"""
        Initialize weak-form damped-spring coefficients to zero.

        This method intentionally does not solve a weak-form least-squares system. For noisy weak
        training, least-squares coefficients computed from a randomly initialized encoder can be a
        poor starting point. Instead, each requested parameter receives trainable zero tensors for
        `K`, `C`, and `b`; the optimizer learns them jointly with the encoder/decoder.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        Latent_States : list[list[torch.Tensor]], len = n_param
            The i'th list element contains two tensors: latent displacement and latent velocity,
            each with shape (n_t(i), n_z). This method uses the displacement dtype to initialize
            coefficients with matching precision.

        t_Grid : list[torch.Tensor], len = n_param
            Time grids corresponding to the latent trajectories. These are checked for length
            consistency but are not otherwise used because weak coefficients are zero-initialized.

        device : torch.device
            Device on which the new coefficient tensors should be stored.

        params : numpy.ndarray, shape = (n_param, n_p)
            Parameter rows used as keys in `self.train_coefs`.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        None. Zero coefficient dictionaries are stored in `self.train_coefs`, and the interpolator
        is updated from the full training-coefficient dictionary.
        """
        assert params is not None, "DampedSpring_weak.initialize_coefficients requires `params`";
        assert isinstance(t_Grid, list) and isinstance(Latent_States, list);
        assert len(Latent_States) == len(t_Grid) == params.shape[0];

        for i in range(params.shape[0]):
            assert isinstance(Latent_States[i], list);
            assert len(Latent_States[i]) == self.n_IC;
            assert isinstance(Latent_States[i][0], torch.Tensor);
            dtype  = Latent_States[i][0].dtype;

            K : torch.Tensor = torch.zeros((self.n_z, self.n_z), device = device, dtype = dtype, requires_grad = True);
            C : torch.Tensor = torch.zeros((self.n_z, self.n_z), device = device, dtype = dtype, requires_grad = True);
            b : torch.Tensor = torch.zeros((self.n_z,),          device = device, dtype = dtype, requires_grad = True);
            self.set_train_coefs(params[i, :], {"K": K, "C": C, "b": b}, device);

        # Finally, update the interpolator using the new training coefficients!
        self.update_interpolator();

        return None;





    # ---------------------------------------------------------------------------------------------
    # Compute losses
    # ---------------------------------------------------------------------------------------------

    def compute_losses(
        self,
        Latent_States : list[list[torch.Tensor]],
        loss_type     : str,
        t_Grid        : list[torch.Tensor],
        params        : numpy.ndarray | None = None
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        r"""
        For each combination of parameter values, this function computes the weak-form
        latent-dynamics loss using the K, C, and b coefficients stored in `self.train_coefs`.
        
        Specifically, let us consider the case when Z has two axes (the case when it has three is 
        identical, just with different coefficients for each instance of the leading dimension of 
        Z). In this case, we assume the i'th row of Z holds the latent state t_0 + i*dt. We use 
        We assume that the latent state is governed by an ODE of the form
        
                z''(t) = K z(t) + C z'(t) + b
        
        Coefficients are initialized by `initialize_coefficients(...)` and then looked up directly 
        from `self.train_coefs` using `params`. Missing entries are intentional hard errors because
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

        loss_LD, loss_coef, loss_stab.

        loss_LD : list[torch.Tensor], len = n_param
            The i'th element of this list is a 0-dimensional tensor whose lone element holds the 
            weak-form latent-dynamics loss from the i'th combination of parameter values.

        loss_coef : list[torch.Tensor], len = n_param
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
        assert(loss_type in ["MSE", "MAE"]);
        assert params is not None, "DampedSpring_weak.compute_losses requires `params` so it can look up weight functions by parameter tuple.";
        assert len(Latent_States) == len(t_Grid) == params.shape[0];

        # Setup 
        loss_LD_list   : list[torch.Tensor] = [];
        loss_coef_list : list[torch.Tensor] = [];
        loss_stab_list : list[torch.Tensor] = [];

        # -----------------------------------------------------------------------------------------
        # Loop over parameter combinations.
        # -----------------------------------------------------------------------------------------

        for i in range(len(t_Grid)):
            assert isinstance(Latent_States[i], list);
            assert len(Latent_States[i]) == self.n_IC;
            for j in range(self.n_IC):
                assert(isinstance(Latent_States[i][j], torch.Tensor));
                assert(len(Latent_States[i][j].shape)   == 2);
                assert(Latent_States[i][j].shape[-1]    == self.n_z);
            
            params_i = params[i, :].reshape(1, -1);

            # -------------------------------------------------------------------------------------
            # Concatenate the latent displacement and velocity.

            Z       : list[torch.Tensor]  = Latent_States[i];   # len = n_IC, j'th element has shape (n_t, n_z)

            Z_D     : torch.Tensor  = Z[0];                     # shape = (n_t, n_z)
            Z_V     : torch.Tensor  = Z[1];                     # shape = (n_t, n_z)

            Phis0, dPhis0, d2Phis0 = self.get_test_functions(params_i[0, :]);
            Phis    : torch.Tensor  = Phis0.to(device = Z_D.device, dtype = Z_D.dtype);
            dPhis   : torch.Tensor  = dPhis0.to(device = Z_D.device, dtype = Z_D.dtype);
            d2Phis  : torch.Tensor  = d2Phis0.to(device = Z_D.device, dtype = Z_D.dtype);

            # Concatenate Z_D, Z_V and a column of 1's to evaluate the weak-form RHS
            # Phis @ cat[Z_D, Z_V, 1] @ E, where E^T = [K, C, b].
            ones      : torch.Tensor = torch.ones((Z_D.shape[0], 1), device = Z_D.device, dtype = Z_D.dtype);
            ZD_ZV_1   : torch.Tensor = torch.cat([Z_D, Z_V, ones], dim = 1);          # shape = (n_t, 2*n_z + 1)

            # -------------------------------------------------------------------------------------
            # Set up coefs using the provided coefficients.

            # Fetch native trainable coefficients for this parameter. Missing entries intentionally
            # raise KeyError because coefficient initialization should have happened in the sampler.
            coef_dict = self.get_train_coefs(params_i[0, :]);
            K = coef_dict["K"].to(device = Z_D.device, dtype = Z_D.dtype);
            C = coef_dict["C"].to(device = Z_D.device, dtype = Z_D.dtype);
            b   = coef_dict["b"].to(device = Z_D.device, dtype = Z_D.dtype);
            coefs = torch.cat([K.T, C.T, b.reshape(1, self.n_z)], dim = 0);
        
            # Compute the weak residual used for the latent-dynamics loss.
            lhs_D = torch.matmul(d2Phis, Z_D)
            lhs_V = -torch.matmul(dPhis, Z_V)
            weak_RHS    : torch.Tensor = torch.matmul(torch.matmul(Phis, ZD_ZV_1), coefs);

            # -------------------------------------------------------------------------------------
            # Compute the stability losses and return.

            scale_D = torch.linalg.norm(d2Phis, dim=1, keepdim=True).clamp(min = 1.0e-10);
            scale_V = torch.linalg.norm(dPhis,  dim=1, keepdim=True).clamp(min = 1.0e-10);

            if loss_type == "MSE":
                loss_D = self.MSE(lhs_D / scale_D, weak_RHS / scale_D)
                loss_V = self.MSE(lhs_V / scale_V, weak_RHS / scale_V)
            elif loss_type == "MAE":
                loss_D = self.MAE(lhs_D / scale_D, weak_RHS / scale_D)
                loss_V = self.MAE(lhs_V / scale_V, weak_RHS / scale_V)

            Loss_LD_i = 0.5 * loss_D + 0.5 * loss_V

            # Stability penalty on the equivalent first-order system y' = A y (+ f).
            # For z'' = K z + C z' + b, define y = [z, z'] so A = [[0, I], [K, C]].
            Z0  : torch.Tensor  = torch.zeros((self.n_z, self.n_z), device = coefs.device, dtype = coefs.dtype);
            I   : torch.Tensor  = torch.eye(self.n_z, device = coefs.device, dtype = coefs.dtype);
            A_top    = torch.cat([Z0, I], dim = 1);
            A_bottom = torch.cat([K, C], dim = 1);
            A = torch.cat([A_top, A_bottom], dim = 0);
            Loss_Stab_i = self.stability_penalty(A);

            # Compute coefficient loss.
            Loss_coef_i = torch.norm(K, 'fro') + torch.norm(C, 'fro') + torch.norm(b);

            # Package the results from this combination of parameter values.
            loss_LD_list.append(Loss_LD_i);
            loss_stab_list.append(Loss_Stab_i);
            loss_coef_list.append(Loss_coef_i);

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
                # Support CUDA/MPS tensors by moving to CPU before NumPy conversion.
                ith_t_Grid = ith_t_Grid.detach().cpu().numpy();
            Same_t_Grid : bool = (len(ith_t_Grid.shape) == 1);
            
            ith_D0 : numpy.ndarray | torch.Tensor = ith_IC[0];
            ith_V0 : numpy.ndarray | torch.Tensor = ith_IC[1];
            n_i    : int                          = ith_D0.shape[0];

            # Each element of IC should have shape (n(i), n_z). 
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
