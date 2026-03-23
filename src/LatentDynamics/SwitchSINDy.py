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
# SwitchSINDy class
# -------------------------------------------------------------------------------------------------

class SwitchSINDy(LatentDynamics):
    def __init__(   self, 
                    n_z             : int,
                    Uniform_t_Grid  : bool, 
                    switch_time     : callable,
                    lstsq_reg       : float = 1.0) -> None:
        r"""
        Initializes a SwitchSINDy object. This is a subclass of the LatentDynamics class which 
        uses switches between a pair of SINDy models for the governing ODE dpeneding on the 
        current time and parameter values. Specifically, we assume that for each parameter 
        combination, there is a "switch time" (defined by the "switch_time" function). For each 
        parameter value, we call this function, which returns a critical time value. If the 
        current time is less than the switch time, then we use the first SINDy model. Otherwise, 
        we use the second SINDy model. That is, given paramerter values, theta, the governing 
        ODE for the i'th component of the latent state is given by:

            z_i'(t) = { \sum_{j = 1}^{N} c_{i,j, theta} f_j(z)  if t < switch_time(theta)
                      { \sum_{j = 1}^{N} d_{i,j, theta} f_j(z) if t >= switch_time(theta)

        where f_1(z), ... , f_N(z) is a library of functions of the latent space, z, c_{i,j, theta} 
        define the coefficients for the first SINDy model and d_{i,j, theta} define the 
        coefficients for the second SINDy model.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        n_z : int
            The number of dimensions in the latent space, where the latent dynamics takes place.
            frame corresponds to time t0, the second to t0 + h, the k'th to t0 + (k - 1)h, etc 
            (note that h may depend on the parameter value, but it needs to be constant for a 
            specific parameter value). The value of this setting determines which finite difference 
            method we use to compute time derivatives. 

        switch_time: callable
            A function that takes a numpy.ndarray of parameter values and returns a float 
            specifying the switch time for the specified parameter values.

        lstsq_reg : float, optional (default 1.0)
            Ridge-regression regularization strength used when fitting coefficients from scratch
            (i.e., when no input_coefs are supplied to calibrate). Replaces plain lstsq with the
            Tikhonov-regularized normal equations  (A^T A + λI) c = A^T b  applied separately to
            the before-switch and after-switch segments. Setting lstsq_reg = 0 falls back to plain
            least squares.

        

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

        # Set the switch_time function.
        self.switch_time : callable = switch_time;

        # Set n_IC and n_coefs.
        # We only allow library terms of order <= 1. If we let z(t) \in \mathbb{R}^{n_z} denote the 
        # latent state at some time, t, then the possible library terms are 1, z_1(t), ... , 
        # z_{n_z}(t). Since each component function gets its own set of coefficients, there must 
        # be n_z*(n_z + 1) total coefficients per sindy model. Since we have two sindy models, 
        # there are 2*n_z*(n_z + 1) total coefficients.
        self.n_coefs    : int   = self.n_z*(self.n_z + 1)*2;
        self.n_IC       : int   = 1;

        # Set the loss functions.
        self.MSE = torch.nn.MSELoss(reduction = 'mean');
        self.MAE = torch.nn.L1Loss(reduction = 'mean');
        return;
    


    def calibrate(  self,  
                    Latent_States   : list[list[torch.Tensor]], 
                    loss_type       : str,
                    t_Grid          : list[torch.Tensor], 
                    params          : numpy.ndarray,
                    input_coefs     : list[torch.Tensor] = []) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        r"""
        This function computes the SINDy and stability losses for a given combination of 
        parameter values. If no input_coefs are provided, then we learn the coefficients using 
        Least Squares. Otherwise we use the passed values. Once we have the coefficients, we 
        subsitue the passed latent states into the governing ODE (defined by the SINDy models). 
        The SINDy loss is simply the mean (across time values) squared error between the computed 
        time derivative and the right hand side of the governing equation. The stability loss is 
        the L1 norm of the coefficients. 

        Note: the SINDy and stability losses are computed using the losses from both SINDy
        models.

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

        params: numpy.ndarray, shape = (n_param, n_p), optional
            The i'th row holds the i'th combination of parameter values. We use this to compute 
            the switch time.

        input_coefs : list[torch.Tensor], len = n_param, optional
            The i'th element of this list is a 1d tensor of shape (n_coefs) holding the 
            coefficients for the i'th combination of parameter values. If input_coefs is None, 
            then we will learn the coefficients using Least Squares. If input_coefs is not None, 
            then we will use the provided coefficients to compute the loss.


            
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
        assert isinstance(t_Grid, list), "t_Grid is %s, not a list" % str(type(t_Grid));
        assert isinstance(Latent_States, list), "Latent_States is %s, not a list" % str(type(Latent_States));
        assert len(Latent_States)   == len(t_Grid), "len(Latent_States) = %d, len(t_Grid) = %d" % (len(Latent_States), len(t_Grid));

        n_param : int   = len(t_Grid);
        n_IC    : int   = 1;
        n_z     : int   = self.n_z;
        assert params is not None, "SwitchSINDy requires params to be provided (needed for switch_time calculation)";
        assert params.shape[0] == n_param,  "params.shape = %s, n_param = %d" % (params.shape, n_param);
        for i in range(n_param):
            assert isinstance(Latent_States[i], list),  "Latent_States[%d] is %s, not a list" % (i, str(type(Latent_States[i])));
            assert len(Latent_States[i]) == n_IC,       "len(Latent_States[%d]) = %d, n_IC = %d" % (i, len(Latent_States[i]), n_IC);

            for j in range(n_IC):
                assert isinstance(Latent_States[i][j], torch.Tensor),   "Latent_States[%d][%d] is %s, not a torch.Tensor" % (i, j, str(type(Latent_States[i][j])));
                assert len(Latent_States[i][j].shape)   == 2,           "len(Latent_States[%d][%d].shape) = %s, should be 2" % (i, j, str(Latent_States[i][j].shape));
                assert Latent_States[i][j].shape[-1]    == n_z,         "Latent_States[%d][%d].shape[-1] = %d, should be %d" % (i, j, Latent_States[i][j].shape[-1], n_z);

        # Run checks on loss_type.
        assert loss_type in ["MSE", "MAE"], "loss_type = %s, should be 'MSE' or 'MAE'" % loss_type;

        # Run checks on input_coefs.
        assert isinstance(input_coefs, list),       "input_coefs is %s, not a list" % str(type(input_coefs));
        if(len(input_coefs) > 0):
            assert isinstance(input_coefs, list),   "input_coefs is %s, not a list" % str(type(input_coefs));
            assert len(input_coefs) == n_param,     "len(input_coefs) = %d, n_param = %d" % (len(input_coefs), n_param);
            for i in range(n_param):
                assert isinstance(input_coefs[i], torch.Tensor),    "input_coefs[%d] is %s, not a torch.Tensor" % (i, str(type(input_coefs[i])));
                assert len(input_coefs[i].shape) == 1,              "len(input_coefs[%d].shape) = %s, should be 1" % (i, str(input_coefs[i].shape));
                assert input_coefs[i].shape[0] == self.n_coefs,     "input_coefs[%d].shape[0] = %d, should be %d" % (i, input_coefs[i].shape[0], self.n_coefs);


        # -----------------------------------------------------------------------------------------
        # If there are multiple combinations of parameter values, loop through them.
        # -----------------------------------------------------------------------------------------

        if (n_param > 1):
            # Prepare an array to house the flattened coefficient matrices for each combination of
            # parameter values.
            output_coefs_list : list[torch.Tensor] = [];

            # Compute the losses, coefficients for each combination of parameter values.
            loss_sindy_list : list[torch.Tensor] = [];
            loss_stab_list  : list[torch.Tensor] = [];
            loss_coef_list  : list[torch.Tensor] = []; 
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
                    output_coefs, loss_sindy_i, loss_coef_i, loss_stab_i = self.calibrate(   
                                                                    Latent_States = [Latent_States[i]], 
                                                                    t_Grid        = [t_Grid[i]],
                                                                    loss_type     = loss_type, 
                                                                    params        = params_i);
                else:
                    output_coefs, loss_sindy_i, loss_coef_i, loss_stab_i = self.calibrate(
                                                                    Latent_States = [Latent_States[i]], 
                                                                    t_Grid        = [t_Grid[i]],
                                                                    input_coefs   = [input_coefs[i]],
                                                                    loss_type     = loss_type, 
                                                                    params        = params_i);

                # Package the results from this combination of parameter values.
                output_coefs_list.append(output_coefs);
                loss_sindy_list.append(loss_sindy_i[0]);
                loss_stab_list.append(loss_stab_i[0]);
                loss_coef_list.append(loss_coef_i[0]);
            
            # Package everything to return!
            # Use cat instead of stack since each output_coefs already has shape (1, n_coefs)
            # cat along dim=0 gives (n_param, n_coefs) as expected
            return torch.cat(output_coefs_list, dim=0), loss_sindy_list, loss_coef_list, loss_stab_list;
            

        # -----------------------------------------------------------------------------------------
        # One combination of parameter values.
        # -----------------------------------------------------------------------------------------


        # -----------------------------------------------------------------------------------------
        # Setup

        t_Grid0 : torch.Tensor  = t_Grid[0];
        Z       : torch.Tensor  = Latent_States[0][0];      # shape = (n_t, n_z)
        n_t     : int           = len(t_Grid0);


        # -----------------------------------------------------------------------------------------
        # Compute the time derivatives.

        # Which method we use depends on if we have a uniform grid spacing or not. If so, we use 
        # an O(h^4) method. Otherwise, we use an O(h^2) one. In either case, this yields a 2d 
        # torch.Tensor object of shape (n_t, n_z) whose i,j element holds the holds an 
        # approximation of (d/dt) Z_j(t_Grid0[i]).
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
        Z_with_ones     : torch.Tensor  = torch.cat([torch.ones(n_t, 1, device = Z.device, dtype =Z .dtype), Z], dim = 1); # shape = (n_t, n_z + 1)

        # Next, let's find the switch time, then use this to make time, Z, and dZdt 
        # tensors for before and after the switch time.
        switch_time_theta : float = self.switch_time(params);
        LOGGER.debug("Switch time for parameter %s is: %.3e" % (str(params), switch_time_theta));
        t_Grid_before   : torch.Tensor  = t_Grid0[t_Grid0 < switch_time_theta];     # shape = (n_t_before)
        t_Grid_after    : torch.Tensor  = t_Grid0[t_Grid0 >= switch_time_theta];    # shape = (n_t_after)
        Z_before        : torch.Tensor  = Z_with_ones[t_Grid0 < switch_time_theta]; # shape = (n_t_before, n_z + 1)
        Z_after         : torch.Tensor  = Z_with_ones[t_Grid0 >= switch_time_theta];# shape = (n_t_after, n_z + 1)
        dZdt_before     : torch.Tensor  = dZdt[t_Grid0 < switch_time_theta];        # shape = (n_t_before, n_z)
        dZdt_after      : torch.Tensor  = dZdt[t_Grid0 >= switch_time_theta];       # shape = (n_t_after, n_z)
        n_t_before      : int           = len(t_Grid_before);
        n_t_after       : int           = len(t_Grid_after);


        # -----------------------------------------------------------------------------------------
        # Learn the coefficients (if needed)

        if(len(input_coefs) == 0):
            # Solve for before/after coefficients via Tikhonov-regularized least squares.
            # Normal equations:  (A^T A + λ I) c = A^T b  applied to each regime separately.
            # Setting lstsq_reg = 0 falls back to plain least squares (no regularization).
            n_lib   : int = Z_before.shape[1];  # n_z + 1

            if(n_t_before > 0):
                if self.lstsq_reg > 0.0:
                    gram_b      : torch.Tensor  = Z_before.T @ Z_before + self.lstsq_reg * torch.eye(n_lib, device = Z.device, dtype = Z.dtype);
                    coefs_before                = torch.linalg.solve(gram_b, Z_before.T @ dZdt_before);
                else:
                    coefs_before                = torch.linalg.lstsq(Z_before, dZdt_before).solution;
            else:
                LOGGER.warning("No time points before switch_time=%.6e! Setting all before-coefficients to zero. Consider using data with earlier time points." % switch_time_theta);
                coefs_before            = torch.zeros(self.n_z + 1, self.n_z, device = Z.device, dtype = Z.dtype);
            
            if(n_t_after > 0):
                if self.lstsq_reg > 0.0:
                    gram_a      : torch.Tensor  = Z_after.T @ Z_after + self.lstsq_reg * torch.eye(n_lib, device = Z.device, dtype = Z.dtype);
                    coefs_after                 = torch.linalg.solve(gram_a, Z_after.T @ dZdt_after);
                else:
                    coefs_after                 = torch.linalg.lstsq(Z_after, dZdt_after).solution;
            else:
                LOGGER.warning("No time points after switch_time=%.6e! Setting all after-coefficients to zero. Consider using data with later time points." % switch_time_theta);
                coefs_after             = torch.zeros(self.n_z + 1, self.n_z, device = Z.device, dtype = Z.dtype);
        else:
            coefs_before            = input_coefs[0][:self.n_z*(self.n_z + 1)].reshape(self.n_z + 1, self.n_z);
            coefs_after             = input_coefs[0][self.n_z*(self.n_z + 1):].reshape(self.n_z + 1, self.n_z);
        
        LOGGER.debug("SINDy before switch time calibration: coefs_before min = %.6e, max = %.6e, mean = %.6e, abs_mean = %.6e" % (
            float(coefs_before.min().item()), float(coefs_before.max().item()),
            float(coefs_before.mean().item()), float(torch.abs(coefs_before).mean().item())));
        LOGGER.debug("SINDy after switch time calibration: coefs_after min = %.6e, max = %.6e, mean = %.6e, abs_mean = %.6e" % (
            float(coefs_after.min().item()), float(coefs_after.max().item()),
            float(coefs_after.mean().item()), float(torch.abs(coefs_after).mean().item())));
        LOGGER.debug("Time point distribution: n_t_before = %d, n_t_after = %d, switch_time = %.6e" % (
            n_t_before, n_t_after, switch_time_theta));
        
        # Warn if time distribution is very imbalanced (less than 10% in either regime)
        if n_t_before > 0 and n_t_after > 0:
            ratio = min(n_t_before, n_t_after) / (n_t_before + n_t_after);
            if ratio < 0.1:
                LOGGER.warning("Time points heavily imbalanced: %.1f%% before switch, %.1f%% after switch. " 
                             "Consider adjusting simulation time range for better coefficient learning." % 
                             (100*n_t_before/n_t, 100*n_t_after/n_t));



        # -----------------------------------------------------------------------------------------
        # Compute the losses.

        # Compute the sindy losses.
        if(loss_type == "MSE"):
            if(n_t_before > 0):
                loss_sindy_before = torch.sum(torch.square(dZdt_before - Z_before @ coefs_before));
            else:
                loss_sindy_before = torch.tensor(0.0, dtype = coefs_before.dtype, device = coefs_before.device);
            
            if(n_t_after > 0):
                loss_sindy_after = torch.sum(torch.square(dZdt_after - Z_after @ coefs_after));
            else:
                loss_sindy_after = torch.tensor(0.0, dtype=coefs_after.dtype, device=coefs_after.device);
        elif(loss_type == "MAE"):
            if(n_t_before > 0):
                loss_sindy_before = torch.sum(torch.abs(dZdt_before - Z_before @ coefs_before));
            else:
                loss_sindy_before = torch.tensor(0.0, dtype = coefs_before.dtype, device = coefs_before.device);
            
            if(n_t_after > 0):
                loss_sindy_after = torch.sum(torch.abs(dZdt_after - Z_after @ coefs_after));
            else:
                loss_sindy_after = torch.tensor(0.0, dtype = coefs_after.dtype, device = coefs_after.device);
        
        # Divide by the number of time steps to get the mean loss. Note that it is impossible 
        # for both n_t_before and n_t_after to be zero, so the divisor will always be non-zero.
        loss_sindy = (loss_sindy_before + loss_sindy_after) / (n_t_before + n_t_after);

        # Stability losses
        A_before            = coefs_before[1:, :].T;  # z' = b_before + A_before z
        A_after             = coefs_after[1:, :].T;   # z' = b_after  + A_after  z
        loss_stab           = self.stability_penalty(A_before) + self.stability_penalty(A_after);
        
        # Coefficient losses
        loss_coef_before    = torch.norm(coefs_before, 'fro');
        loss_coef_after     = torch.norm(coefs_after, 'fro');
        loss_coef           = loss_coef_before + loss_coef_after;

        # Prepare coefs and the losses to return. Note that we flatten the coefficient matrices.
        # Concatenate flattened coefficients: [coefs_before_flat, coefs_after_flat]
        # Shape should be (1, 2*n_z*(n_z+1)) to match expected (n_param, n_coefs) format
        output_coefs   : torch.Tensor  = torch.cat([coefs_before.flatten(), coefs_after.flatten()]).reshape(1, -1);
        return output_coefs, [loss_sindy], [loss_coef], [loss_stab]



    def simulate(   self,
                    coefs   : numpy.ndarray           | torch.Tensor, 
                    IC      : list[list[numpy.ndarray | torch.Tensor]],
                    t_Grid  : list[numpy.ndarray      | torch.Tensor],
                    params  : numpy.ndarray) -> list[list[numpy.ndarray | torch.Tensor]]:
        """
        Time integrates the latent dynamics from multiple initial conditions for each combination
        of coefficients in coefs. 
 

        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------
        
        coefs : numpy.ndarray or torch.Tensor, shape = (n_param, n_coef)
            i'th row holds the coefficients (both for before and after the switch time) when we 
            use the i'th combination of parameter values. Specifically, we assume the i'th row of 
            coefs holds the concatenation of the flattened coefficient matrices for before and 
            after the switch time. We inductively call simulate on each row of coefs. 

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
            The i'th row holds the i'th combination of parameter values. We use this to compute 
            the switch time.


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
        assert len(coefs.shape)     == 2,       "coefs.shape = %s, should be (n_param, n_coefs)" % str(coefs.shape);
        n_param : int = coefs.shape[0];
        assert isinstance(t_Grid, list),        "t_Grid is %s, not a list" % str(type(t_Grid));
        assert isinstance(IC, list),            "IC is %s, not a list" % str(type(IC));
        assert len(IC)              == n_param, "len(IC) = %d, n_param = %d" % (len(IC), n_param);
        assert len(t_Grid)          == n_param, "len(t_Grid) = %d, n_param = %d" % (len(t_Grid), n_param);
        assert isinstance(IC[0], list),         "IC[0] is %s, not a list" % str(type(IC[0]));
        n_IC : int = len(IC[0]);
        assert n_IC == 1,                       "n_IC = %d, should be 1" % n_IC;
        for i in range(n_param):
            assert isinstance(IC[i], list),                                     "IC[%d] is %s, not a list" % (i, str(type(IC[i])));
            assert len(IC[i]) == n_IC,                                          "len(IC[%d]) = %d, n_IC = %d" % (i, len(IC[i]), n_IC);
            assert len(t_Grid[i].shape) == 2 or len(t_Grid[i].shape) == 1,      "len(t_Grid[%d].shape) = %d" % (i, len(t_Grid[i].shape));
            for j in range(n_IC):
                assert len(IC[i][j].shape) == 2,                                "IC[%d][%d].shape = %s" % (i, j, str(IC[i][j].shape));
                assert type(coefs)          == type(IC[i][j]),                  "type(coefs) = %s, type(IC[%d][%d]) = %s" % (str(type(coefs)), i, j, str(type(IC[i][j])));
                assert IC[i][j].shape[1]    == self.n_z,                        "IC[%d][%d].shape[1] = %d, self.n_z = %d" % (i, j, IC[i][j].shape[1], self.n_z);
                if(len(t_Grid[i].shape) == 2):
                    assert t_Grid[i].shape[0] == IC[i][j].shape[0],             "t_Grid[%d].shape[0] = %d, IC[%d][%d].shape[0] = %d" % (i, t_Grid[i].shape[0], i, j, IC[i][j].shape[0]);


        # -----------------------------------------------------------------------------------------
        # If there are multiple combinations of coefficients, loop through them.
        # -----------------------------------------------------------------------------------------

        if(n_param > 1):
            LOGGER.debug("Simulating with %d parameter combinations" % n_param);
            assert params is not None, "SwitchSINDy requires params to be provided (needed for switch_time calculation)";

            # Cycle through the parameter combinations
            Z   : list[list[numpy.ndarray | torch.Tensor]]  = [];
            for i in range(n_param): 
                # Fetch the i'th set of coefficients, the corresponding collection of initial
                # conditions, and the set of time values.
                ith_coefs   : numpy.ndarray             | torch.Tensor              = coefs[i, :].reshape(1, -1);
                ith_IC      : list[list[numpy.ndarray   | torch.Tensor]]            = [IC[i]];
                ith_t_Grid  : list[numpy.ndarray | torch.Tensor]                    = [t_Grid[i]];
                ith_params  = params[i, :].reshape(1, -1);

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
        # One combination of parameter values.
        # -----------------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------------
        # Setup

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

        # Compute the switch time for this parameter combination
        assert params is not None, "SwitchSINDy requires params to be provided (needed for switch_time calculation)";
        switch_time_theta : float = self.switch_time(params);

        # First, we need to extract the matrix of coefficients for before and after the switch time. 
        # We know that coefs holds the coefficient matrix for before and after the switch time. 
        # These appear in the latent model z'(t) = hstack[1, z(t)] E_{before/after}^T.
        E_before   : numpy.ndarray | torch.Tensor = coefs[0, :self.n_z*(self.n_z + 1)].reshape(self.n_z + 1, self.n_z).T;
        E_after    : numpy.ndarray | torch.Tensor = coefs[0, self.n_z*(self.n_z + 1):].reshape(self.n_z + 1, self.n_z).T;

        # Extract A and b for before and after the switch time. Note that we need to reshape 
        # b to have shape (1, n_z) to enable broadcasting.
        b_before   : numpy.ndarray | torch.Tensor = E_before[:, 0 ].reshape(1, -1);
        A_before   : numpy.ndarray | torch.Tensor = E_before[:, 1:];
        b_after    : numpy.ndarray | torch.Tensor = E_after[:, 0 ].reshape(1, -1);
        A_after    : numpy.ndarray | torch.Tensor = E_after[:, 1:];

        # Set up a lambda function to approximate 
        #   z'(t) \approx b_{before/after} + A_{before/after} z(t)
        # Which coefficients we use depends on the time.
        # Check the type of coefs to determine which matmul to use.
        if(isinstance(coefs, numpy.ndarray)):
            def f(t : float, z : numpy.ndarray) -> numpy.ndarray:
                if(t < switch_time_theta):
                    return b_before + numpy.matmul(z, A_before.T);
                else:
                    return b_after + numpy.matmul(z, A_after.T);
        elif(isinstance(coefs, torch.Tensor)):
            def f(t : float, z : torch.Tensor) -> torch.Tensor:
                if(t < switch_time_theta):
                    return b_before + torch.matmul(z, A_before.T);
                else:
                    return b_after + torch.matmul(z, A_after.T);
        else:
            raise TypeError("coefs must be either numpy.ndarray or torch.Tensor, got %s" % type(coefs));

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
