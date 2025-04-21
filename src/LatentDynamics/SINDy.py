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
                    coef_norm_order : str | float,
                    Uniform_t_Grid  : bool) -> None:
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

        n_z: The number of dimensions in the latent space, where the latent dynamics takes place.

        coef_norm_order: A string or float specifying which norm we want to use when computing
        the coefficient loss. We pass as the "p" argument to torch.norm. 

        Uniform_t_Grid: A boolean which, if True, specifies that for each parameter value, the 
        times corresponding to the frames of the solution for that parameter value will be 
        uniformly spaced. In other words, the first frame corresponds to time t0, the second to 
        t0 + h, the k'th to t0 + (k - 1)h, etc (note that h may depend on the parameter value, but
        it needs to be constant for a specific parameter value). The value of this setting 
        determines which finite difference method we use to compute time derivatives. 
        """

        # Run the base class initializer. The only thing this does is set the n_z and n_t 
        # attributes.
        super().__init__(n_z                = n_z, 
                         coef_norm_order    = coef_norm_order, 
                         Uniform_t_Grid     = Uniform_t_Grid);
        LOGGER.info("Initializing a SINDY object with n_z = %d, coef_norm_order = %s, Uniform_t_Grid = %s" % (  self.n_z, 
                                                                                                                str(self.coef_norm_order), 
                                                                                                                str(self.Uniform_t_Grid)));

        # Set n_IC and n_coefs.
        # We only allow library terms of order <= 1. If we let z(t) \in \mathbb{R}^{n_z} denote the 
        # latent state at some time, t, then the possible library terms are 1, z_1(t), ... , 
        # z_{n_z}(t). Since each component function gets its own set of coefficients, there must 
        # be n_z*(n_z + 1) total coefficients.
        #TODO(kevin): generalize for high-order dynamics
        self.n_coefs    : int   = self.n_z*(self.n_z + 1);
        self.n_IC       : int   = 1;

        # TODO(kevin): other loss functions
        self.MSE = torch.nn.MSELoss();
        return;
    


    def calibrate(  self,  
                    Latent_States   : list[list[torch.Tensor]], 
                    t_Grid          : list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        This function computes the optimal SINDy coefficients using the current latent time 
        series. Specifically, let us consider the case when Z has two dimensions (the case when 
        it has three is identical, just with different coefficients for each instance of the 
        leading dimension of Z). In this case, we assume that the rows of Z correspond to a 
        trajectory of latent states. Specifically, we assume the i'th row holds the latent state,
        z, at time t_0 + i*dt. We use SINDy to find the coefficients in the dynamical system
        z'(t) = C \Phi(z(t)), where C is a matrix of coefficients and \Phi(z(t)) represents a
        library of terms. We find the matrix C corresponding to the dynamical system that best 
        agrees with the data in the rows of Z. 


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        Latent_States: An n_param (number of parameter combinations we want to calibrate) element
        list. The i'th list element should be an one element list whose j'th element is a 2d numpy 
        array of shape (n_t(i), n_z) whose p, q element holds the q'th component of the j'th 
        derivative of the latent state during the p'th time step (whose time value corresponds to 
        the p'th element of t_Grid) when we use the i'th combination of parameter values. 
        
        t_Grid: An n_param element list of 1d torch.Tensor objects. The i'th element should be a 
        1d tensor of length n_t(i) whose j'th element holds the time value corresponding to the 
        j'th frame when we use the i'th combination of parameter values.

        
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Three variables: coefs, loss_sindy, and loss_coef. 
        
        coefs holds the coefficients. It is a matrix of shape (n_train, n_coef), where n_train 
        is the number of parameter combinations in the training set and n_coef is the number of 
        coefficients in the latent dynamics. The i,j entry of this array holds the value of the 
        j'th coefficient when we use the i'th combination of parameter values.

        loss_sindy holds the total SINDy loss. It is a single element tensor whose lone entry holds
        the sum of the SINDy losses across the set of combinations of parameters in the training 
        set. 

        loss_coef is a single element tensor whose lone element holds the sum of the L1 norms of 
        the coefficients across the set of combinations of parameters in the training set.
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


        # -----------------------------------------------------------------------------------------
        # If there are multiple combinations of parameter values, loop through them.
        
        if (n_param > 1):
            # Prepare an array to house the flattened coefficient matrices for each combination of
            # parameter values.
            coefs = torch.empty([n_param, self.n_coefs], dtype = torch.float32);

            # Compute the losses, coefficients for each combination of parameter values.
            loss_sindy  = torch.zeros(1, dtype = torch.float32);
            loss_coef   = torch.zeros(1, dtype = torch.float32);
            for i in range(n_param):
                """"
                Get the optimal SINDy coefficients for the i'th combination of parameter values. 
                Remember that Latent_States[i][0] is a tensor of shape (n_t(j), n_z) whose (j, k) 
                entry holds the k'th component of the j'th frame of the latent trajectory for the 
                i'th combination of parameter values. 
                
                Note that Result a 3 element tuple.
                """
                result : tuple[torch.Tensor] = self.calibrate(Latent_States = [Latent_States[i]], 
                                                              t_Grid        = [t_Grid[i]]);

                # Package the results from this combination of parameter values.
                coefs[i, :] = result[0];
                loss_sindy += result[1];
                loss_coef  += result[2];
            
            # Package everything to return!
            return coefs, loss_sindy, loss_coef;
            

        # -----------------------------------------------------------------------------------------
        # Evaluate for one combination of parameter values case.

        t_Grid  : torch.Tensor  = t_Grid[0];
        Z       : torch.Tensor  = Latent_States[0];
        n_t     : int           = len(t_Grid);

        # First, compute the time derivatives. Which method we use depends on if we have a uniform 
        # grid spacing or not. If so, we use an O(h^4) method. Otherwise, we use an O(h^2) one. In
        # either case, this yields a 2d torch.Tensor object of shape (n_t, n_z) whose i,j element 
        # holds the holds an approximation of (d/dt) Z_j(t_Grid[i]).
        if(self.Uniform_t_Grid == True):
            h       : float         = t_Grid[1] - t_Grid[0];
            dZdt    : torch.Tensor  = Derivative1_Order4(Z, h);
        else:
            dZdt    : torch.Tensor  = Derivative1_Order2_NonUniform(Z, t_Grid = t_Grid);

        # Concatenate a column of ones. This will correspond to a constant term in the latent 
        # dynamics.
        Z_1     : torch.Tensor  = torch.cat([torch.ones(n_t, 1), Z], dim = 1)
        
        # For each j, solve the least squares problem 
        #   min{ || dZdt[:, j] - Z_1 c_j|| : C_j \in \mathbb{R}ˆNl }
        # where Nl is the number of library terms (in this case, just n_z + 1, since we only allow
        # constant and linear terms). We store the resulting solutions in a matrix, coefs, whose 
        # j'th column holds the results for the j'th column of dZdt. Thus, coefs is a 2d tensor
        # with shape (Nl, n_z).
        coefs   : torch.Tensor  = torch.linalg.lstsq(Z_1, dZdt).solution

        # Compute the losses.
        loss_sindy = self.MSE(dZdt, Z_1 @ coefs)
        # NOTE(kevin): by default, this will be L1 norm.
        loss_coef = torch.norm(coefs, self.coef_norm_order)

        # Prepare coefs and the losses to return. Note that we flatten the coefficient matrix.
        # Note: output of lstsq is not contiguous in memory.
        coefs   : torch.Tensor  = coefs.detach().flatten()
        return coefs, loss_sindy, loss_coef



    def simulate(   self,
                    coefs   : numpy.ndarray             | torch.Tensor, 
                    IC      : list[list[numpy.ndarray]] | list[list[torch.Tensor]],
                    t_Grid  : list[numpy.ndarray]       | list[torch.Tensor]) -> list[list[numpy.ndarray]]  | list[list[torch.Tensor]]:
        """
        Time integrates the latent dynamics from multiple initial conditions for each combination
        of coefficients in coefs. 
 

        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------
        
        coefs: A two dimensional numpy.ndarray or torch.Tensor objects of shape (n_param, n_coef)
        whose i'th row represents the optimal set of coefficients when we use the i'th combination 
        of parameter values. We inductively call simulate on each row of coefs. 

        IC: An n_param element list whose i'th element is an n_IC element list whose j'th element
        is a 2d numpy.ndarray or torch.Tensor object of shape (n(i), n_z). Here, n(i) is the 
        number of initial conditions (for a fixed set of coefficients) we want to simulate forward 
        using the i'th set of coefficients. Further, n_z is the latent dimension. If you want to 
        simulate a single IC, for the i'th set of coefficients, then n(i) == 1. IC[i][j][k, :] 
        should hold the k'th initial condition for the j'th derivative of the latent state when
        we use the i'th combination of parameter values. 

        t_Grid: A n_param element list whose i'th entry is a 2d numpy.ndarray or torch.Tensor 
        object. The i'th entry should either have shape (n(i), n_t(i)) or shape (n_t(i)). Use the
        former case when we want to use different times for each initial condition and the latter
        case when we want to use the same times for all initial conditions. 
        
        In the former case, the j,k array entry specifies k'th time value at which we solve for 
        the latent state when we use the j'th initial condition and the i'th set of coefficients. 
        Each row should be in ascending order. 
        
        In the latter case, the j'th entry should specify the j'th time value at which we solve for 
        each latent state when we use the i'th combination of parameter values.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------        
        
        An n_param element list whose i'th item is a list of length n_IC whose j'th entry is a 3d 
        array of shape (n_t(i), n(i), n_z). The p, q, r entry of this array should hold the r'th 
        component of the p'th frame of the j'th tine derivative of the solution to the latent 
        dynamics when we use the q'th initial condition for the i'th combination of parameter 
        values.
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
            assert(isinstance(IC[i], list));
            assert(len(IC[i]) == n_IC);
            assert(len(t_Grid[i].shape) == 2);
            for j in range(n_IC):
                assert(len(IC[i][j].shape) == 2);
                assert(type(coefs)          == type(IC[i][j]));
                assert(IC[i][j].shape[1]    == self.n_z);


        # -----------------------------------------------------------------------------------------
        # If there are multiple combinations of coefficients, loop through them.
        
        if(n_param > 1):
            LOGGER.debug("Simulating with %d parameter combinations" % n_param);

            # Cycle through the parameter combinations
            Z   : list[list[numpy.ndarray]] | list[list[torch.Tensor]]  = [];
            for i in range(n_param): 
                # Fetch the i'th set of coefficients, the corresponding collection of initial
                # conditions, and the set of time values.
                ith_coefs   : numpy.ndarray             | torch.Tensor              = coefs[i, :].reshape(1, -1);
                ith_IC      : list[list[numpy.ndarray]] | list[list[torch.Tensor]]  = [IC[i]];
                ith_t_Grid  : list[numpy.ndarray]                                   = [t_Grid[i]];

                # Call this function using them. This should return a 1 element holding the 
                # the solution for the i'th combination of parameter values.
                ith_Results : list[numpy.ndarray]   | list[torch.Tensor]    = self.simulate(coefs   = ith_coefs, 
                                                                                            IC      = ith_IC, 
                                                                                            t_Grid  = ith_t_Grid)[0];

                # Add these results to X.
                Z.append(ith_Results);

            # All done.
            return Z;
        

        # -----------------------------------------------------------------------------------------
        # Evaluate for one combination of parameter values case.

        # In this case, there is just one parameter. Extract t_Grid, which has shape 
        # (n(i), n_t(i)).
        t_Grid  : numpy.ndarray | torch.Tensor  = t_Grid[0];
        if(isinstance(t_Grid, torch.Tensor)):
            t_Grid = t_Grid.detach().numpy();
        n_i     : int           = t_Grid.shape[0];
        n_t_i   : int           = t_Grid.shape[1];

        # If we get here, then coefs has one row. In this case, each element of IC should 
        # have shape (n(i), n_z). First, reshape coefs as a matrix. Since we only allow for linear 
        # terms, there are n_z + 1 library terms and n_z equations, where n_z = self.n_z.
        c_1 : numpy.ndarray | torch.Tensor  = coefs.reshape([self.n_z + 1, self.n_z]).T;
        if(isinstance(c_1, torch.Tensor)):
            c_1 = c_1.detach().numpy();

        # Set up a lambda function to approximate dz_dt. In SINDy, we learn a coefficient matrix 
        # C such that the latent state evolves according to the dynamical system 
        #   z'(t) = C \Phi(z(t)), 
        # where \Phi(z(t)) is the library of terms. Note that the zero column of C corresponds 
        # to the constant library term, 1. 
        f   = lambda z, t : c_1[:, 1:] @ z + c_1[:, 0];

        # Set up an array to hold the results of each simulation.
        if(isinstance(coefs, numpy.ndarray)):
            X : numpy.ndarray   = numpy.empty((n_t_i, n_i, self.n_z), dtype = numpy.float32);
        elif(isinstance(coefs, torch.Tensor)):
            X : torch.Tensor    = torch.empty((n_t_i, n_i, self.n_z), dtype = torch.float32);

        # Solve the ODE forward in time for each set of initial conditions. Remember that IC
        # should be a 1 element list whose lone element is a n_IC element = 1 element whose 
        # lone element is a 2d numpy.ndarray object with shape (n(i), n_z).
        for j in range(n_i):
            X[:, j, :] = RK4(f, IC[0][0][j, :], t_Grid[j, :]);
        
        # All done!
        return [[X]];