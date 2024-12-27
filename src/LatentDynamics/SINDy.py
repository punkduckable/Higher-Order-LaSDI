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
from    InputParser         import  InputParser;
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
                 dim        : int, 
                 nt         : int, 
                 config     : dict) -> None:
        r"""
        Initializes a SINDy object. This is a subclass of the LatentDynamics class which uses the 
        SINDy algorithm as its model for the ODE governing the latent state. Specifically, we 
        assume there is a library of functions, f_1(z), ... , f_N(z), each one of which is a 
        monomial of the components of the latent space, z, and a set of coefficients c_{i,j}, 
        i = 1, 2, ... , dim and j = 1, 2, ... , N such that
            z_i'(t) = \sum_{j = 1}^{N} c_{i,j} f_j(z)
        In this case, we assume that f_1, ... , f_N consists of the set of order <= 1 monomials. 
        That is, f_1(z), ... , f_N(z) = 1, z_1, ... , z_{dim}.
            

        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        dim: The number of dimensions in the latent space, where the latent dynamics takes place.

        nt: The number of time steps we want to generate when solving (numerically) the latent 
        space dynamics.

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

        # Run the base class initializer. The only thing this does is set the dim and nt 
        # attributes.
        super().__init__(dim, nt);
        LOGGER.info("Initializing a SINDY object with dim = %d, nt = %d" % (self.dim, self.nt));

        # We only allow library terms of order <= 1. If we let z(t) \in \mathbb{R}^{dim} denote the 
        # latent state at some time, t, then the possible library terms are 1, z_1(t), ... , 
        # z_{dim}(t). Since each component function gets its own set of coefficients, there must 
        # be dim*(dim + 1) total coefficients.
        #TODO(kevin): generalize for high-order dynamics
        self.ncoefs = self.dim * (self.dim + 1);

        # Now, set up an Input parser to process the contents of the config['sindy'] dictionary. 
        assert('sindy' in config)
        input_parser = InputParser(config['sindy'], name = 'sindy_input');

        """
        Determine which finite difference scheme we should use to approximate the time derivative
        of the latent space dynamics. Currently, we allow the following values for "fd_type":
            - 'sbp12': summation-by-parts 1st/2nd (boundary/interior) order operator
            - 'sbp24': summation-by-parts 2nd/4th order operator
            - 'sbp36': summation-by-parts 3rd/6th order operator
            - 'sbp48': summation-by-parts 4th/8th order operator
        """
        self.fd_type    : str       = input_parser.getInput(['fd_type'], fallback = 'sbp12');
        self.fd         : callable  = FDdict[self.fd_type];

        r"""
        Fetch the operator matrix. What does this do? Suppose we have a time series with nt points, 
        x(t_0), ... , x(t_{nt - 1}) \in \mathbb{R}^d. Further assume that for each j, 
        t_j = t_0 + j \delta t, where \delta t is some positive constant. Let j \in {0, 1, ... , 
        nt - 1}. Let xj be j'th vector whose k'th element is x_j(t_k). Then, the i'th element of 
        M xj holds the approximation to x_j'(t_k) using the stencil we selected above. 

        For instance, if we selected sdp12, corresponding to the central difference scheme, then 
        we have (for j != 0, nt - 1)
            [M xj]_i = (x_j(t_{i + 1}) - x(t_{j - 1}))/(2 \delta t).
        """
        self.fd_oper, _, _          = self.fd.getOperators(self.nt);

        # Fetch the norm we are going to use on the sindy coefficients.
        # NOTE(kevin): by default, this will be L1 norm.
        self.coef_norm_order = input_parser.getInput(['coef_norm_order'], fallback = 1);

        # TODO(kevin): other loss functions
        self.MSE = torch.nn.MSELoss();

        # All done!
        return;
    


    def calibrate(self, 
                  Latent_States : list[torch.Tensor],
                  dt            : float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

        Latent_States: A single element list housing a 2d or 3d tensor. Let Z = Latent_States[0].
        If Z is a 2d tensor, then it has shape (Nt, dim), where Nt specifies the length of the 
        sequence of latent states and dim is the dimension of the latent space. In this case, the 
        i,j entry of Z holds the j'th component of the latent state at the time  t_0 + i*dt. If 
        it is a 3d tensor, then it has shape (Np, Nt, dim). In this case, we assume there are 
        Np different combinations of parameter values. The i, j, k entry of Z in this case holds 
        the k'th component of the latent encoding at time t_0 + j*dt when we use the i'th 
        combination of parameter values. 

        dt: The time step between time steps. See the description of the "Z" argument. 
        

        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        We return three variables. 
        
        The first holds the coefficients. It is a matrix of shape (n_train, n_coef), where n_train 
        is the number of parameter combinations in the training set and n_coef is the number of 
        coefficients in the latent dynamics. The i,j entry of this array holds the value of the 
        j'th coefficient when we use the i'th combination of parameter values.

        The second holds the total SINDy loss. It is a single element tensor whose lone entry holds
        the sum of the SINDy losses across the set of combinations of parameters in the training 
        set. 

        The third is a single element tensor whose lone element holds the sum of the L1 norms of 
        the coefficients across the set of combinations of parameters in the training set.
        """

        # Run checks.
        assert(len(Latent_States) == 1);

        # Fetch Z.
        Z : torch.Tensor = Latent_States[0];


        # -----------------------------------------------------------------------------------------
        # If Z has three dimensions, loop over all train cases.
        if (Z.dim() == 3):
            # Fetch the number of training cases.
            n_train : int = Z.size(0)

            # Prepare an array to house the flattened coefficient matrices for each combination of
            # parameter values.
            coefs = torch.empty([n_train, self.ncoefs], dtype = torch.float32)

            # Initialize the losses. Note that these are floats which we will replace with 
            # tensors.
            loss_sindy, loss_coef = 0.0, 0.0

            # Cycle through the combinations of parameter values.
            for i in range(n_train):
                """"
                Get the optimal SINDy coefficients for the i'th combination of parameter values. 
                Remember that Z is 3d tensor of shape (Np, Nt, dim) whose (i, j, k) entry holds 
                the k'th component of the j'th frame of the latent trajectory for the i'th 
                combination of parameter values. Note that Result a 3 element tuple.
                """
                result : tuple[torch.Tensor] = self.calibrate([Z[i]], dt)

                # Package the results from this combination of parameter values.
                coefs[i, :] = result[0]
                loss_sindy += result[1]
                loss_coef  += result[2]
            
            # Package everything to return!
            return coefs, loss_sindy, loss_coef
            

        # -----------------------------------------------------------------------------------------
        # evaluate for one training case.
        assert(Z.dim() == 2)

        # First, compute the time derivatives. This yields a torch.Tensor object whose i,j entry 
        # holds an approximation of (d/dt) Z_j(t_0 + i*dt)
        dZdt = self.compute_time_derivative(Z, dt)
        time_dim, space_dim = dZdt.shape

        # Concatenate a column of ones. This will correspond to a constant term in the latent 
        # dynamics.
        Z_i     : torch.Tensor  = torch.cat([torch.ones(time_dim, 1), Z], dim = 1)
        
        # For each j, solve the least squares problem 
        #   min{ || dZdt[:, j] - Z_i c_j|| : C_j \in \mathbb{R}ˆNl }
        # where Nl is the number of library terms (in this case, just dim + 1, since we only allow
        # constant and linear terms). We store the resulting solutions in a matrix, coefs, whose 
        # j'th column holds the results for the j'th column of dZdt. Thus, coefs is a 2d tensor
        # with shape (Nl, dim).
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

        Z: A 2d tensor of shape (Nt, dim) whose i, j entry holds the j'th component of the i'th 
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



    def simulate(self, 
                 coefs  : numpy.ndarray, 
                 IC     : list[numpy.ndarray], 
                 times  : numpy.ndarray) -> list[numpy.ndarray]:
        """
        Time integrates the latent dynamics when it uses the coefficients specified in coefs and 
        starts from the (single) initial condition in z0.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------
        
        coefs: A one dimensional numpy.ndarray object representing the flattened copy of the array 
        of latent dynamics coefficients that calibrate returns.

        IC: A single element list of numpy.ndarray objects of shape dim. The j'th element of this 
        list should hold the j'th component of the initial position for the latent dynamics.

        times: A 1d numpy ndarray object whose i'th entry holds the value of the i'th time value 
        where we want to compute the latent solution. The elements of this array should be in 
        ascending order.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------        
        
        A single element list whose lone element is a 2d numpy.ndarray object holding the solution 
        to the latent dynamics at the time values specified in times when we use the coefficients 
        in coefs to characterize the latent dynamics model. 
        
        Specifically, it is a 2d array of shape (nt, dim), where nt is the number of time steps 
        (size of times) and dim is the latent space dimension (self.dim). Thus, the i,j element of 
        this matrix holds the j'th component of the latent solution at the time stored in the i'th 
        element of times. 
        """

        # Run checks.
        assert(len(IC)          == 1);
        assert(IC[0].size       == self.dim);
        assert(len(times.shape) == 1);

        # Extract IC.
        z0  : numpy.ndarray = IC[0];

        # First, reshape coefs as a matrix. Since we only allow for linear terms, there are dim + 1
        # library terms and dim equations, where dim = self.dim.
        # Note: copy is inevitable for numpy==1.26. removed copy=False temporarily.
        c_i : numpy.ndarray = coefs.reshape([self.dim + 1, self.dim]).T

        # Set up a lambda function to approximate dz_dt. In SINDy, we learn a coefficient matrix 
        # C such that the latent state evolves according to the dynamical system 
        #   z'(t) = C \Phi(z(t)), 
        # where \Phi(z(t)) is the library of terms. Note that the zero column of C corresponds 
        # to the constant library term, 1. 
        dz_dt               = lambda z, t : c_i[:, 1:] @ z + c_i[:, 0]

        # Solve the ODE forward in time.
        Z_i : numpy.ndarray = odeint(dz_dt, z0, times)

        # All done!
        return [Z_i];
    


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