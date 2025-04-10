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
from    scipy.integrate     import  odeint;

from    LatentDynamics      import  LatentDynamics;
from    Stencils            import  FDdict;


# Setup logger.
LOGGER  : logging.Logger    = logging.getLogger(__name__);



# -------------------------------------------------------------------------------------------------
# SINDy class
# -------------------------------------------------------------------------------------------------

class SINDy(LatentDynamics):
    fd_type     = ''
    fd          = None
    fd_oper     = None


    def __init__(self, 
                 n_z        : int, 
                 config     : dict) -> None:
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

        config: A dictionary housing the settings we need to set up a SINDy object. Specifically, 
        this dictionary should have a key called "sindy" whose corresponding value is another 
        dictionary with the following two keys:
            - fd_type: A string specifying which finite-difference scheme we should use when
            approximating the time derivative of the solution to the latent dynamics at a 
            specific time. Currently, the following options are allowed:
                - 'sbp12': summation-by-parts 1st/2nd (boundary/interior) order operator
                - 'sbp24': summation-by-parts 2nd/4th order operator
                - 'sbp36': summation-by-parts 3rd/6th order operator
                - 'sbp48': summation-by-parts 4th/8th order operator
            - coef_norm_order: A string specifying which norm we want to use when computing
            the coefficient loss.
        """

        # Run the base class initializer. The only thing this does is set the n_z and n_t 
        # attributes.
        super().__init__(n_z, n_t);
        LOGGER.info("Initializing a SINDY object with n_z = %d, n_t = %d" % (self.n_z, self.n_t));

        # Set n_IC and n_coefs.
        # We only allow library terms of order <= 1. If we let z(t) \in \mathbb{R}^{n_z} denote the 
        # latent state at some time, t, then the possible library terms are 1, z_1(t), ... , 
        # z_{n_z}(t). Since each component function gets its own set of coefficients, there must 
        # be n_z*(n_z + 1) total coefficients.
        #TODO(kevin): generalize for high-order dynamics
        self.n_coefs    : int   = self.n_z*(self.n_z + 1);
        self.n_IC       : int   = 1;


        # Make sure config is actually a dictionary for a SINDY type latent dynamics.
        assert('sindy' in config)

        """
        Determine which finite difference scheme we should use to approximate the time derivative
        of the latent space dynamics. Currently, we allow the following values for "fd_type":
            - 'sbp12': summation-by-parts 1st/2nd (boundary/interior) order operator
            - 'sbp24': summation-by-parts 2nd/4th order operator
            - 'sbp36': summation-by-parts 3rd/6th order operator
            - 'sbp48': summation-by-parts 4th/8th order operator
        """
        self.fd_type    : str       = config['sindy']['fd_type'];
        self.fd         : callable  = FDdict[self.fd_type];

        r"""
        Fetch the operator matrix. What does this do? Suppose we have a time series with n_t points, 
        x(t_0), ... , x(t_{nt - 1}) \in \mathbb{R}^d. Further assume that for each j, 
        t_j = t_0 + j \delta t, where \delta t is some positive constant. Let j \in {0, 1, ... , 
        n_t - 1}. Let xj be j'th vector whose k'th element is x_j(t_k). Then, the i'th element of 
        M xj holds the approximation to x_j'(t_k) using the stencil we selected above. 

        For instance, if we selected sdp12, corresponding to the central difference scheme, then 
        we have (for j != 0, n_t - 1)
            [M xj]_i = (x_j(t_{i + 1}) - x(t_{j - 1}))/(2 \delta t).
        """
        self.fd_oper, _, _          = self.fd.getOperators(self.n_t);

        # Fetch the norm we are going to use on the sindy coefficients.
        # NOTE(kevin): by default, this will be L1 norm.
        self.coef_norm_order = config['sindy']['coef_norm_order'];

        # TODO(kevin): other loss functions
        self.MSE = torch.nn.MSELoss();

        # All done!
        return;
    


    def calibrate(  self,  
                    Latent_States   : list[list[torch.Tensor]]  | list[torch.Tensor], 
                    t_Grid          : list[numpy.ndarray]       | numpy.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

        Latent_States: Either a list of lists of 2d tensors, or a list of 2d tensors. 
        
        If Latent_States is a list of lists, then t_Grid should be a list of 1d numpy.ndarray 
        object. In this case, Latent_States should be a list of length n_param (number of parameter 
        combinations we want to calibrate). The i'th list element should be an n_IC element list 
        whose j'th element is a 2d numpy array of shape (n_t(i), n_z) whose p, q element holds the 
        q'th component of the j'th derivative of the latent state during the p'th time step (whose 
        time value corresponds to the p'th element of t_Grid) when we use the i'th combination of 
        parameter values. 
        
        if Latent_States is a list of numpy arrays, then t_Grid should be a 1d numpy arrays. In 
        this case, Latent_States should be a list of length n_IC. The j'th element of this list 
        should be an numpy ndarray of shape (n_t(i), n_z) whose p, q element holds the q'th 
        component of the j'th derivative of the latent state during the p'th time step (whose 
        time value corresponds to the p'th element t_Grid).
        
        t_Grid: Either a list of 1d numpy ndarray objects or a 1d numpy ndarray object. See the 
        description of Latent_States for details. 

        
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
        if(isinstance(t_Grid, list)):
            n_param : int   = len(t_Grid);
            for i in range(n_param):
                assert(len(Latent_States[i]) == 1);
        else: 
            n_param : int   = 1;
            assert(len(Latent_States) == 1);


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
                result : tuple[torch.Tensor] = self.calibrate(Latent_States = Latent_States[i], 
                                                              t_Grid        = t_Grid[i]);

                # Package the results from this combination of parameter values.
                coefs[i, :] = result[0];
                loss_sindy += result[1];
                loss_coef  += result[2];
            
            # Package everything to return!
            return coefs, loss_sindy, loss_coef;
            

        # -----------------------------------------------------------------------------------------
        # evaluate for one combination of parameter values case.

        # First, compute the time derivatives. This yields a torch.Tensor object whose i,j entry 
        # holds an approximation of (d/dt) Z_j(t_0 + i*dt).
        Z       : torch.Tensor = Latent_States[0];
        dZdt    : torch.Tensor = self.compute_time_derivative(Z, t_Grid[1] - t_Grid[0])
        time_dim, space_dim = dZdt.shape

        # Concatenate a column of ones. This will correspond to a constant term in the latent 
        # dynamics.
        Z_i     : torch.Tensor  = torch.cat([torch.ones(time_dim, 1), Z], dim = 1)
        
        # For each j, solve the least squares problem 
        #   min{ || dZdt[:, j] - Z_i c_j|| : C_j \in \mathbb{R}ˆNl }
        # where Nl is the number of library terms (in this case, just n_z + 1, since we only allow
        # constant and linear terms). We store the resulting solutions in a matrix, coefs, whose 
        # j'th column holds the results for the j'th column of dZdt. Thus, coefs is a 2d tensor
        # with shape (Nl, n_z).
        coefs   : torch.Tensor  = torch.linalg.lstsq(Z_i, dZdt).solution

        # Compute the losses.
        loss_sindy = self.MSE(dZdt, Z_i @ coefs)
        # NOTE(kevin): by default, this will be L1 norm.
        loss_coef = torch.norm(coefs, self.coef_norm_order)

        # Prepare coefs and the losses to return. Note that we flatten the coefficient matrix.
        # Note: output of lstsq is not contiguous in memory.
        coefs   : torch.Tensor  = coefs.detach().flatten()
        return coefs, loss_sindy, loss_coef



    def compute_time_derivative(self, Z : torch.Tensor, Dt : float) -> torch.Tensor:
        """
        This function builds the SINDy dataset, assuming only linear terms in the SINDy dataset. 
        The time derivatives are computed through finite difference.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        Z: A 2d tensor of shape (n_t, n_z) whose i, j entry holds the j'th component of the i'th 
        time step in the latent time series. We assume that Z[i, :] represents the latent state
        at time t_0 + i*Dt

        Dt: The time step between latent frames (the time between Z[:, i] and Z[:, i + 1])


        -------------------------------------------------------------------------------------------
        Returns 
        -------------------------------------------------------------------------------------------

        The output dZdt is a 2D tensor with the same shape as Z. It's i, j entry holds an 
        approximation to (d/dt)Z_j(t_0 + j Dt). We compute this approximation using self's stencil.
        """

        return (1. / Dt) * torch.sparse.mm(self.fd_oper, Z)



    def simulate(   self,
                    coefs   : numpy.ndarray             | torch.Tensor, 
                    IC      : list[list[numpy.ndarray]] | list[list[torch.Tensor]],
                    t_Grid  : list[numpy.ndarray]) -> list[list[numpy.ndarray]]  | list[list[torch.Tensor]]:
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

        t_Grid: A n_param element list whose i'th entry is a 2d numpy ndarray object of shape 
        (n(i), n_t(i)) whose j, k entry specifies the k'th time value we want to find the latent 
        states when we use the j'th initial conditions and the i'th set of coefficients. Each 
        row of each array should have elements in ascending order. 


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------        
        
        An n_param element list whose i'th item is a list of length n_IC whose j'th entry is a 3d 
        array of shape (n_t(i), n(i), n_z). The p, q, r entry of this array should hold the r'th 
        component of the q'th sample of the p'th time step of the j'th derivative latent state 
        use the i'th combination of parameter values to define the latent dynamics. 
        """

        # Run checks.
        assert(len(coefs.shape)     == 2);
        n_param = coefs.shape[0];
        assert(isinstance(t_Grid, list));
        assert(len(IC)              == n_param);
        assert(len(t_Grid)          == n_param);
        
        assert(isinstance(IC[0], list));
        n_IC : int = len(IC[0]);
        assert(n_IC == 1);
        for i in range(n_param):
            assert(isinstance(IC[i], list));
            assert(len(IC[i] == n_IC));
            assert(len(t_Grid[i].shape) == 2);
            for j in range(n_IC):
                assert(len(IC[i][j].shape) == 2);
                assert(type(coefs)          == type(IC[i][j]));
                assert(IC[i][j].shape[1]    == self.n_z);

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

                # Call this function using them.
                ith_Results : list[numpy.ndarray]   | list[torch.Tensor]    = self.simulate(
                                                                                    coefs   = ith_coefs, 
                                                                                    IC      = ith_IC, 
                                                                                    times   = ith_t_Grid)[0];

                # Add these results to X.
                Z.append(ith_Results);

            # All done.
            return Z;
        
        # In this case, there is just one parameter. Extract t_Grid, which has shape 
        # (n(i), n_t(i)).
        t_Grid  : numpy.ndarray = t_Grid[0];
        n_i     : int           = t_Grid.shape[0];
        n_t_i   : int           = t_Grid.shape[1];

        # If we get here, then coefs has one row. In this case, each element of IC should 
        # have shape (n(i), n_z). First, reshape coefs as a matrix. Since we only allow for linear 
        # terms, there are n_z + 1 library terms and n_z equations, where n_z = self.n_z.
        c_i : numpy.ndarray = coefs.reshape([self.n_z + 1, self.n_z]).T;

        # Set up a lambda function to approximate dz_dt. In SINDy, we learn a coefficient matrix 
        # C such that the latent state evolves according to the dynamical system 
        #   z'(t) = C \Phi(z(t)), 
        # where \Phi(z(t)) is the library of terms. Note that the zero column of C corresponds 
        # to the constant library term, 1. 
        dz_dt               = lambda z, t : c_i[:, 1:] @ z + c_i[:, 0];

        # Set up an array to hold the results of each simulation.
        if(isinstance(coefs, numpy.ndarray)):
            X : numpy.ndarray   = numpy.empty((n_t_i, n_i, self.n_z), dtype = numpy.float32);
        elif(isinstance(coefs, torch.Tensor)):
            X : torch.Tensor    = torch.empty((n_t_i, n_i, self.n_z), dtype = torch.float32);

        # Solve the ODE forward in time for each set of initial conditions. Remember that IC
        # should be a 1 element list whose lone element is a n_IC element = 1 element whose 
        # lone element is a 2d numpy.ndarray object with shape (n(i), n_z).
        for j in range(n_i):
            X[:, j, :] = odeint(dz_dt, IC[0][0][j, :], t_Grid);
        
        # All done!
        return [[X]];



    def export(self) -> dict:
        """
        This function packages self's contents into a dictionary which it then returns. We can use 
        this dictionary to create a new SINDy object which has the same internal state as self. 
        
        
        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------
        
        None!


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------
        
        A dictionary with two keys: fd_type and coef_norm_order. The former specifies which finite
        different scheme we use while the latter specifies which norm we want to use when computing
        the coefficient loss. 
        """

        param_dict                      = super().export()
        param_dict['fd_type']           = self.fd_type
        param_dict['coef_norm_order']   = self.coef_norm_order
        return param_dict