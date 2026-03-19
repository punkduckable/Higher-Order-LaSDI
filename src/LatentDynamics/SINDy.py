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
from    FirstOrderSolvers   import  RK4;

# Setup logger.
LOGGER  : logging.Logger    = logging.getLogger(__name__);



# -------------------------------------------------------------------------------------------------
# SINDy class
# -------------------------------------------------------------------------------------------------

class SINDy(LatentDynamics):
    def __init__(   self, 
                    n_z             : int,
                    Uniform_t_Grid  : bool,
                    lstsq_reg       : float = 1.0) -> None:
        r"""
        Initializes a SINDy object. This is a subclass of the LatentDynamics class which uses the 
        SINDy algorithm as its model for the ODE governing the latent state. Specifically, we 
        assume there is a library of functions, f_1(z), ... , f_N(z), each one of which is a 
        monomial of the components of the latent space, z, and a set of coefficients c_{i,j}, 
        i = 1, 2, ... , n_z and j = 1, 2, ... , N such that

            z_i'(t) = \sum_{j = 1}^{N} c_{i,j} f_j(z)
        
        In this case, we assume that f_1, ... , f_N consists of the set of order <= 1 monomials. 
        That is, f_1(z), ... , f_N(z) = 1, z_1, ... , z_{n_z}.
            

        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        n_z : int
            The number of dimensions in the latent space, where the latent dynamics takes place.
            frame corresponds to time t0, the second to t0 + h, the k'th to t0 + (k - 1)h, etc 
            (note that h may depend on the parameter value, but it needs to be constant for a 
            specific parameter value). The value of this setting determines which finite difference 
            method we use to compute time derivatives. 

        lstsq_reg : float, optional (default 1.0)
            Ridge-regression regularization strength used when fitting SINDy coefficients from
            scratch (i.e., when no input_coefs are supplied to calibrate). The least-squares
            problem is replaced by the Tikhonov-regularized normal equations:

                (Z^T Z + lstsq_reg * I) c = Z^T dZ/dt

            Plain least-squares (lstsq_reg = 0) can produce arbitrarily large coefficients when
            the Gram matrix Z^T Z is ill-conditioned, which commonly happens for newly-added
            greedy-sampling training points whose latent trajectory hasn't been seen by the encoder
            before. Setting lstsq_reg > 0 bounds ||c|| <= ||Z^T dZ/dt|| / lstsq_reg and prevents
            the coefficient explosion that would otherwise blow up the training loss on the next
            round. A value of 1.0 is a reasonable starting point; increase it (e.g., 10 or 100)
            if you still see large initial coefficients for new training points.

            
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------
        
        Nothing!
        """

        # Run the base class initializer. The only thing this does is set the n_z and n_t 
        # attributes.
        super().__init__(n_z = n_z, Uniform_t_Grid = Uniform_t_Grid);
        self.lstsq_reg : float = lstsq_reg;
        LOGGER.info("Initializing a SINDY object with n_z = %d, Uniform_t_Grid = %s, lstsq_reg = %s" % (  self.n_z, 
                                                                                                                str(self.Uniform_t_Grid),
                                                                                                                str(self.lstsq_reg)));

        # Set n_IC and n_coefs.
        # We only allow library terms of order <= 1. If we let z(t) \in \mathbb{R}^{n_z} denote the 
        # latent state at some time, t, then the possible library terms are 1, z_1(t), ... , 
        # z_{n_z}(t). Since each component function gets its own set of coefficients, there must 
        # be n_z*(n_z + 1) total coefficients.
        #TODO(kevin): generalize for high-order dynamics
        self.n_coefs    : int   = self.n_z*(self.n_z + 1);
        self.n_IC       : int   = 1;

        # TODO(kevin): other loss functions
        self.MSE = torch.nn.MSELoss(reduction = 'mean');
        self.MAE = torch.nn.L1Loss(reduction = 'mean');
        return;
    


    def calibrate(  self,  
                    Latent_States   : list[list[torch.Tensor]], 
                    loss_type       : str,
                    t_Grid          : list[torch.Tensor], 
                    params          : numpy.ndarray | None = None,
                    input_coefs     : list[torch.Tensor] = []) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        This function computes the SINDy and stability losses for a given combination of 
        parameter values. If no input_coefs are provided, then we learn the coefficients using 
        Least Squares. Otherwise we use the passed values. Once we have the coefficients, we 
        subsitue the passed latent states into the governing ODE (defined by the SINDy model). 
        The SINDy loss is simply the mean (across time values) squared error between the computed 
        time derivative and the right hand side of the governing equation. The stability loss is 
        the L1 norm of the coefficients. 


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        Latent_States : list[list[torch.Tensor]], len = n_param
            The i'th list element is a one element list whose only element is a 2d numpy array of 
            shape (n_t(i), n_z) whose p, q element holds the q'th component of the latent state 
            during the p'th time step (whose time value corresponds to the p'th element of t_Grid) 
            when we use the i'th combination of parameter values. 
        
        loss_type : str
            The type of loss function to use. Must be either "MSE" or "MAE".

        t_Grid : list[torch.Tensor], len = n_param
            i'th element should be a 1d tensor of shape (n_t(i)) whose j'th element holds the time 
            value corresponding to the j'th frame when we use the i'th combination of parameter 
            values.

        input_coefs : list[torch.Tensor], len = n_param, optional
            The i'th element of this list is a 1d tensor of shape (n_coefs) holding the 
            coefficients for the i'th combination of parameter values. If input_coefs is None, 
            then we will learn the coefficients using Least Squares. If input_coefs is not None, 
            then we will use the provided coefficients to compute the loss.

        params: numpy.ndarray, shape = (n_param, n_p), optional
            The i'th row holds the i'th combination of parameter values. This class doesn't use 
            parameters, so it ignores this argument. Default is None.

            
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        output_coefs, loss_sindy, loss_stab. 
        
        output_coefs : torch.Tensor, shape = (n_param, n_coef)
            A matrix of shape (n_param, n_coef). The i,j entry of this array holds the value of 
            the j'th coefficient when we use the i'th combination of parameter values.

        loss_sindy : list[torch.Tensor], len = n_param
            The i'th element of this list is a 0-dimensional tensor whose lone element holds the 
            sum of the SINDy losses from the i'th combination of parameter values. 

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
        n_IC    : int   = 1;
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

        # Run checks on input_coefs.
        assert(isinstance(input_coefs, list));
        if(len(input_coefs) > 0):
            assert(isinstance(input_coefs, list));
            assert(len(input_coefs) == n_param);
            for i in range(n_param):
                assert(isinstance(input_coefs[i], torch.Tensor));
                assert(len(input_coefs[i].shape) == 1);
                assert(input_coefs[i].shape[0] == self.n_coefs);


        # -----------------------------------------------------------------------------------------
        # If there are multiple combinations of parameter values, loop through them.
        
        if (n_param > 1):
            # Prepare an array to house the flattened coefficient matrices for each combination of
            # parameter values.
            output_coefs_list : list[torch.Tensor] = [];

            # Compute the losses, coefficients for each combination of parameter values.
            loss_sindy_list  : list[torch.Tensor] = [];
            loss_stab_list   : list[torch.Tensor] = [];
            for i in range(n_param):
                """"
                Get the optimal SINDy coefficients for the i'th combination of parameter values. 
                Remember that Latent_States[i][0] is a tensor of shape (n_t(j), n_z) whose (j, k) 
                entry holds the k'th component of the j'th frame of the latent trajectory for the 
                i'th combination of parameter values. 
                
                Note that Result a 3 element tuple.
                """
                # Extract params for this iteration (handle None case)
                params_i = None if params is None else params[i, :].reshape(1, -1);
                
                if(len(input_coefs) == 0):
                    output_coefs, loss_sindy_i, loss_stab_i = self.calibrate(   
                                                                    Latent_States = [Latent_States[i]], 
                                                                    t_Grid        = [t_Grid[i]],
                                                                    loss_type     = loss_type, 
                                                                    params        = params_i);
                else:
                    output_coefs, loss_sindy_i, loss_stab_i = self.calibrate(
                                                                    Latent_States = [Latent_States[i]], 
                                                                    t_Grid        = [t_Grid[i]],
                                                                    input_coefs   = [input_coefs[i]],
                                                                    loss_type     = loss_type, 
                                                                    params        = params_i);

                # Package the results from this combination of parameter values.
                output_coefs_list.append(output_coefs);
                loss_sindy_list.append(loss_sindy_i[0]);
                loss_stab_list.append(loss_stab_i[0]);
            
            # Package everything to return!
            # Use cat instead of stack since each output_coefs already has shape (1, n_coefs)
            # cat along dim=0 gives (n_param, n_coefs) as expected
            return torch.cat(output_coefs_list, dim = 0), loss_sindy_list, loss_stab_list;
            

        # -----------------------------------------------------------------------------------------
        # Evaluate for one combination of parameter values case.

        t_Grid0 : torch.Tensor  = t_Grid[0];
        Z       : torch.Tensor  = Latent_States[0][0];
        n_t     : int           = len(t_Grid0);

        # First, compute the time derivatives. Which method we use depends on if we have a uniform 
        # grid spacing or not. If so, we use an O(h^4) method. Otherwise, we use an O(h^2) one. In
        # either case, this yields a 2d torch.Tensor object of shape (n_t, n_z) whose i,j element 
        # holds the holds an approximation of (d/dt) Z_j(t_Grid0[i]).
        if(self.Uniform_t_Grid == True):
            h       : float         = (t_Grid0[1] - t_Grid0[0]).item();
            dZdt    : torch.Tensor  = Derivative1_Order4(Z, h);
        else:
            dZdt                    = Derivative1_Order2_NonUniform(Z, t_Grid = t_Grid0);

        # Log diagnostics to check for time scaling or derivative issues
        LOGGER.debug("SINDy calibration: Z shape=%s, min=%.6e, max=%.6e, std=%.6e" % (
            str(Z.shape), float(Z.min().item()), float(Z.max().item()), float(Z.std().item())));
        LOGGER.debug("SINDy calibration: dZ/dt min=%.6e, max=%.6e, mean=%.6e, std=%.6e" % (
            float(dZdt.min().item()), float(dZdt.max().item()), 
            float(dZdt.mean().item()), float(dZdt.std().item())));
        LOGGER.debug("SINDy calibration: t_Grid min=%.6e, max=%.6e, range=%.6e, mean_dt=%.6e" % (
            float(t_Grid0.min().item()), float(t_Grid0.max().item()),
            float((t_Grid0.max() - t_Grid0.min()).item()),
            float((t_Grid0[1:] - t_Grid0[:-1]).mean().item()) if n_t > 1 else 0.0));

        # Concatenate a column of ones. This will correspond to a constant term in the latent 
        # dynamics. Ensure the ones tensor is on the same device as Z.
        Z_1     : torch.Tensor  = torch.cat([torch.ones(n_t, 1, device = Z.device, dtype =Z .dtype), Z], dim = 1);
        
        if(len(input_coefs) == 0):
            # Solve for SINDy coefficients using Tikhonov-regularized least squares (ridge
            # regression).  Plain lstsq can return coefficients with magnitude in the hundreds
            # when the Gram matrix Z_1^T Z_1 is ill-conditioned, which is common for newly-added
            # greedy-sampling training points whose latent trajectory is unfamiliar to the encoder.
            # Ridge regression caps ||coefs|| <= ||Z_1^T dZdt|| / lstsq_reg and is equivalent to
            # the standard (unregularized) lstsq when lstsq_reg = 0.
            #
            # Normal equations:  (Z_1^T Z_1 + λ I) c = Z_1^T dZdt
            #   where  λ = self.lstsq_reg
            n_lib   : int           = Z_1.shape[1];    # number of library terms (n_z + 1)
            rhs     : torch.Tensor  = Z_1.T @ dZdt;    # shape (n_lib, n_z)
            if self.lstsq_reg > 0.0:
                gram    : torch.Tensor  = Z_1.T @ Z_1 + self.lstsq_reg * torch.eye(n_lib, device = Z_1.device, dtype = Z_1.dtype);
                coefs   : torch.Tensor  = torch.linalg.solve(gram, rhs);
            else:
                # lstsq_reg == 0: fall back to plain least squares (no regularization).
                coefs   : torch.Tensor  = torch.linalg.lstsq(Z_1, dZdt).solution;
            LOGGER.debug("SINDy calibration: Learned coefs (lstsq_reg=%.2e) min=%.6e, max=%.6e, mean=%.6e, abs_mean=%.6e" % (
                self.lstsq_reg, float(coefs.min().item()), float(coefs.max().item()),
                float(coefs.mean().item()), float(torch.abs(coefs).mean().item())));
        else:
            coefs   : torch.Tensor  = input_coefs[0].reshape(self.n_z + 1, self.n_z);
            LOGGER.debug("SINDy calibration: Using input coefs min=%.6e, max=%.6e, mean=%.6e" % (
                float(coefs.min().item()), float(coefs.max().item()), float(coefs.mean().item())));

        # Compute the losses.
        if(loss_type == "MSE"):
            loss_sindy = self.MSE(dZdt, Z_1 @ coefs);
        elif(loss_type == "MAE"):
            loss_sindy = self.MAE(dZdt, Z_1 @ coefs);
        A = coefs[1:, :].T;  # linear term in z' = b + A z
        loss_stab = self.stability_penalty(A)

        # Prepare coefs and the losses to return. Note that we flatten the coefficient matrix.
        # Note: output of lstsq is not contiguous in memory.
        output_coefs   : torch.Tensor  = coefs.reshape(1, -1);
        return output_coefs, [loss_sindy], [loss_stab]



    def simulate(   self,
                    coefs   : numpy.ndarray           | torch.Tensor, 
                    IC      : list[list[numpy.ndarray | torch.Tensor]],
                    t_Grid  : list[numpy.ndarray      | torch.Tensor],
                    params  : numpy.ndarray | None = None) -> list[list[numpy.ndarray | torch.Tensor]]:
        """
        Time integrates the latent dynamics from multiple initial conditions for each combination
        of coefficients in coefs. 
 

        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------
        
        coefs : numpy.ndarray or torch.Tensor, shape = (n_param, n_coef)
            i'th row represents the optimal set of coefficients when we use the i'th combination 
            of parameter values. We inductively call simulate on each row of coefs. 

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

        # Run checks.
        assert(len(coefs.shape)     == 2);
        n_param : int = coefs.shape[0];
        assert(isinstance(t_Grid, list));
        assert(isinstance(IC, list));
        assert(len(IC)              == n_param);
        assert(len(t_Grid)          == n_param);
        
        assert(isinstance(IC[0], list));
        n_IC : int = len(IC[0]);
        assert(n_IC == 1);
        for i in range(n_param):
            assert isinstance(IC[i], list),                                     "IC[%d] is not a list" % i;
            assert len(IC[i]) == n_IC,                                          "len(IC[%d]) = %d, n_IC = %d" % (i, len(IC[i]), n_IC);
            assert len(t_Grid[i].shape) == 2 or len(t_Grid[i].shape) == 1,      "len(t_Grid[%d].shape) = %d" % (i, len(t_Grid[i].shape));
            for j in range(n_IC):
                assert len(IC[i][j].shape) == 2,                                "IC[%d][%d].shape = %s" % (i, j, str(IC[i][j].shape));
                assert type(coefs)          == type(IC[i][j]),                  "type(coefs) = %s, type(IC[%d][%d]) = %s" % (str(type(coefs)), i, j, str(type(IC[i][j])));
                assert IC[i][j].shape[1]    == self.n_z,                        "IC[%d][%d].shape[1] = %d, self.n_z = %d" % (i, j, IC[i][j].shape[1], self.n_z);
                if(len(t_Grid[i].shape) == 2):
                    assert t_Grid[i].shape[0] == IC[i][j].shape[0];


        # -----------------------------------------------------------------------------------------
        # If there are multiple combinations of coefficients, loop through them.
        
        if(n_param > 1):
            LOGGER.debug("Simulating with %d parameter combinations" % n_param);

            # Cycle through the parameter combinations
            Z   : list[list[numpy.ndarray | torch.Tensor]]  = [];
            for i in range(n_param): 
                # Fetch the i'th set of coefficients, the corresponding collection of initial
                # conditions, and the set of time values.
                ith_coefs   : numpy.ndarray             | torch.Tensor              = coefs[i, :].reshape(1, -1);
                ith_IC      : list[list[numpy.ndarray   | torch.Tensor]]            = [IC[i]];
                ith_t_Grid  : list[numpy.ndarray | torch.Tensor]                    = [t_Grid[i]];
                ith_params  = None if params is None else params[i, :].reshape(1, -1);

                # Call this function using them. This should return a 1 element holding the 
                # the solution for the i'th combination of parameter values.
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

        # coefs has shape (1, n_coefs). Each element of IC should have shape (n(i), n_z). 
        Z0  : numpy.ndarray | torch.Tensor  = IC[0][0]; 
        n_i : int                           = Z0.shape[0];

        # First, we need to extract the matrix of coefficients. We know that coefs is the least 
        # squares solution to dZ_dt = hstack[1, Z] E^T. 
        E   : numpy.ndarray | torch.Tensor = coefs.reshape([self.n_z + 1, self.n_z]).T;

        # Extract A and b. Note that we need to reshape b to have shape (1, n_z) to enable
        # broadcasting.
        b   : numpy.ndarray | torch.Tensor = E[:, 0 ].reshape(1, -1);
        A   : numpy.ndarray | torch.Tensor = E[:, 1:];


        # Set up a lambda function to approximate 
        #   z'(t) \approx b + A z(t)
        # In this case, we expect dz_dt and z to have shape (n(i), n_z). Thus, matmul(z, A.T) will 
        # have shape (n(i), n_z). The i'th row of this should hold the z portion of the rhs of the 
        # latent dynamics for the i'th IC. Similar results hold for dot(dz_dt, C.T). The final 
        # result should have shape (n, n_z). The i'th row should hold the rhs of the latent 
        # dynamics for the i'th IC.
        if(isinstance(coefs, numpy.ndarray)):
            f   = lambda t, z: b + numpy.matmul(z, A.T);
        if(isinstance(coefs, torch.Tensor)):
            f   = lambda t, z: b + torch.matmul(z, A.T);

        # Solve the ODE forward in time. U should have shape (n_t, n(i), n_z). If we use the 
        # same t values for each IC, then we can exploit the fact that the latent dynamics are 
        # autonomous to solve using each IC simultaneously. Otherwise, we need to run the latent
        # dynamics one IC at a time. 
        if(Same_t_Grid == True):
            Z = [[RK4(f = f, y0 = Z0, t_Grid = t_Grid0)]]; 
        else:
            # Cycle through the ICs.
            Z_list : list[torch.Tensor | numpy.ndarray] = [];   
            for j in range(n_i):
                Z_j         = RK4(f = f, y0 = Z0[j, :].reshape(1, -1), t_Grid = t_Grid0[j, :]);
                Z_list.append(Z_j);

            # Stack the results.
            if(isinstance(coefs, numpy.ndarray)):
                Z = [[numpy.concatenate(Z_list, axis = 1)]];    # shape = (n_t, n_i, n_z)
            elif(isinstance(coefs, torch.Tensor)):
                Z = [[torch.cat(Z_list, dim = 1)]];            # shape = (n_t, n_i, n_z)
        
        # All done!
        return Z;
