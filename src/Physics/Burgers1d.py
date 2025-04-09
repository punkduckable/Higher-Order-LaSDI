# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

# Add the main directory to the search path.
import  os;
import  sys;
src_Path        : str   = os.path.dirname(os.path.dirname(__file__));
LD_Path         : str   = os.path.join(src_Path, "LatentDynamics");
util_Path       : str   = os.path.join(src_Path, "Utilities");
sys.path.append(src_Path);
sys.path.append(LD_Path);
sys.path.append(util_Path);

import  numpy;
from    scipy.sparse.linalg import  spsolve;
from    scipy.sparse        import  spdiags;
import  torch;

from    Physics             import  Physics;
from    Stencils            import  FDdict;


######## REMOVE ME   ||
######## REMOVE ME   ||
######## REMOVE ME   ||
######## REMOVE ME  \  /
######## REMOVE ME   \/

import  sys;
import  os;
Utilities_Path  : str   = os.path.abspath(os.path.join(os.path.dirname(__file__), "Utilities"));
sys.path.append(Utilities_Path);
from    FiniteDifference    import  Derivative1_Order4;

######## REMOVE ME   /\
######## REMOVE ME  /  \
######## REMOVE ME   ||
######## REMOVE ME   ||
######## REMOVE ME   ||


# -------------------------------------------------------------------------------------------------
# Burgers 1D class
# -------------------------------------------------------------------------------------------------

class Burgers1D(Physics):
    # Class variables
    a_idx = None; # parameter index for a
    w_idx = None; # parameter index for w


    
    def __init__(self, config : dict, param_names : list[str] = None) -> None:
        """
        This is the initializer for the Burgers Physics class. This class essentially acts as a 
        wrapper around a 1D Burgers solver.

        
        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        config: A dictionary housing the settings for the Burgers object. This should be the 
        "physics" sub-dictionary of the configuration file. 

        param_names: A list of strings. There should be one list item for each parameter. The i'tj
        element of this list should be a string housing the name of the i'th parameter. For the 
        Burgers class, this should have two elements: a and w. 

        
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Nothing!
        """

        # Checks
        assert(len(param_names) == 2);
        assert('a' in param_names);
        assert('w' in param_names);

        # Call the super class initializer.
        super().__init__(config, param_names);

        # The solution to Burgers' equation is scalar valued, so the qdim is 1. Likewise, since 
        # there is only one spatial dimension in the 1D burgers example, dim is also 1.
        self.qdim           : int   = 1;
        self.spatial_dim    : int   = 1;
        
        # Make sure the config dictionary is actually for Burgers' equation.
        assert('burgers1d' in config);

        # Fetch variables from config. 
        self.n_x                    : int       = config['burgers1d']['n_x'];
        self.Frame_Shape            : list[int] = [self.n_x];                       # number of grid points along each spatial axis

        # Fetch more variables from the config.
        self.x_min  = config['burgers1d']['x_min'];   # Minimum value of the spatial variable in the problem domain
        self.x_max  = config['burgers1d']['x_max'];   # Maximum value of the spatial variable in the problem domain
        self.dx     = (self.x_max - self.x_min) / (self.n_x - 1);    # Spacing between grid points along the spatial axis.
        assert(self.dx > 0.);

        # Set up the x grid. 
        self.x_grid : numpy.ndarray = numpy.linspace(self.x_min, self.x_max, self.n_x);

        # ???
        self.maxk                   : int   = config['burgers1d']['maxk'];                  # TODO: ??? What is this ???
        self.convergence_threshold  : float = config['burgers1d']['convergence_threshold'];

        # Determine which index corresponds to 'a' and 'w' (we pass an array of parameter values, 
        # we need this information to figure out which element corresponds to which variable).
        self.a_idx = self.param_names.index('a');
        self.w_idx = self.param_names.index('w');
        
        # All done!
        return;
    


    def initial_condition(self, param : numpy.ndarray) -> list[numpy.ndarray]:
        """
        Evaluates the initial condition along the spatial grid. For this class, we use the 
        following initial condition:
            u(0, x) = a*exp(-x^2 / (2*w^2))
        where a and w are the corresponding parameter values.

        We also compute the velocity IC by solving forward a few time steps and the computing the 
        time derivative using finite differences.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        param: A 1d numpy.ndarray object with two elements corresponding to the values of the w 
        and a parameters. self.a_idx and self.w_idx tell us which index corresponds to which 
        variable.
        

        -------------------------------------------------------------------------------------------
        Returns 
        -------------------------------------------------------------------------------------------

        A two element list of 1d numpy.ndarray objects, each of shape length self.n_x (the number 
        of grid points along the spatial axis). The i'th element holds the initial state of the 
        i'th time derivative of the FOM state.
        """

        # Fetch the parameter values.
        a   : float     = param[self.a_idx];
        w   : float     = param[self.w_idx];  

        # Get the initial displacement.
        u0  : numpy.ndarray     = a * numpy.exp(- self.x_grid ** 2 / 2 / w / w);

        # return [u0];

        #"""
        ######## REMOVE ME   ||
        ######## REMOVE ME   ||
        ######## REMOVE ME   ||
        ######## REMOVE ME  \  /
        ######## REMOVE ME   \/

        # Calculate dt.
        n_t     : int           = self.config['burgers1d']['n_t'];
        t_max   : float         = self.config['burgers1d']['t_max']; 
        dt      : float         = t_max/(n_t - 1);

        # Solve forward a few time steps.
        D       : numpy.ndarray         = solver(u0, self.maxk, self.convergence_threshold, 5, self.n_x, dt, self.dx);
        V       : numpy.ndarray         = Derivative1_Order4(torch.Tensor(D), h = dt);
        
        # Get the ICs from the solution.
        u0                              = D[0, :];
        v0                              = V[0, :];
            
        # All done!
        return [u0, v0];
    
        ######## REMOVE ME   /\
        ######## REMOVE ME  /  \
        ######## REMOVE ME   ||
        ######## REMOVE ME   ||
        ######## REMOVE ME   ||
        #"""


    def solve(self, param : numpy.ndarray) -> tuple[list[torch.Tensor], numpy.ndarray]:
        """
        Solves the 1d burgers equation when the IC uses the parameters in the param array.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        param: A 1d numpy.ndarray object with two elements corresponding to the values of the w 
        and a parameters. self.a_idx and self.w_idx tell us which index corresponds to which 
        variable.
        

        -------------------------------------------------------------------------------------------
        Returns 
        -------------------------------------------------------------------------------------------
        
        A two element tuple: X, t_Grid.

        X is an n_param element list whose i'th element is an n_IC element list whose j'th element
        is a torch.Tensor object of shape (n_t(i), n_x[0], ... , n_x[ns- 1]) holding the j'th 
        derivative of the FOM solution for the i'th combination of parameter values. Here, n_IC is 
        the number of initial conditions needed to specify the IC, n_param is the number of rows 
        in param, n_t(i) is the number of time steps we used to generate the solution with the 
        i'th combination of parameter values (the length of the i'th element of t_Grid).

        t_Grid is a list whose i'th element is a 1d numpy array housing the time steps from the 
        solution to the underlying equation when we use the i'th combination of parameter values.
        """
        
        # Fetch the initial condition.
        u0 : numpy.ndarray = self.initial_condition(param)[0];
        
        # Compute dt. 
        n_t     : int           = self.config['burgers1d']['n_t'];
        t_max   : float         = self.config['burgers1d']['t_max']; 
        dt      : float         = t_max/(n_t - 1);

        """
        # Solve the PDE and then reshape the result to be a 3d tensor with a leading dimension of 
        # size 1.
        X       : torch.Tensor          = torch.Tensor(solver(u0, self.maxk, self.convergence_threshold, n_t - 1, self.n_x, dt, self.dx));        
        new_X   : list[torch.Tensor]    = [X.reshape(1, n_t, self.n_x)];
        """

        #"""
        ######## REMOVE ME   ||
        ######## REMOVE ME   ||
        ######## REMOVE ME   ||
        ######## REMOVE ME  \  /
        ######## REMOVE ME   \/
    
        # Solve the PDE and then reshape the result to be a 3d tensor with a leading dimension of 
        # size 1.
        X       : torch.Tensor  = torch.Tensor(solver(u0, self.maxk, self.convergence_threshold, n_t - 1, self.n_x, dt, self.dx));
        V       : torch.Tensor  = Derivative1_Order4(X, h = dt);
        
        X       : torch.Tensor  = X.reshape(1, n_t, self.n_x);
        V       : torch.Tensor  = V.reshape(1, n_t, self.n_x);

        new_X   : list[torch.Tensor]    = [X, V];

        ######## REMOVE ME   /\
        ######## REMOVE ME  /  \
        ######## REMOVE ME   ||
        ######## REMOVE ME   ||
        ######## REMOVE ME   ||
        #"""

        # All done!
        return new_X;
    

    
    def residual(self, X_hist : list[numpy.ndarray]) -> tuple[numpy.ndarray, float]:
        """
        This function computes the PDE residual (difference between the left and right hand side
        of Burgers' equation when we substitute in the solution in X_hist).


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        X_hist: A single element list of 2d numpy.ndarray object of shape (n_t, n_x), where n_t is 
        the number of points along the temporal axis (this is specified by the configuration file) 
        and n_x is the number of points along the spatial axis. The i,j element of the d'th array 
        should have the j'th component of the d'th derivative of the fom solution at the i'th time 
        step.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        A two element tuple. The first is a numpy.ndarray object of shape (n_t - 2, n_x - 2) whose 
        i, j element holds the residual at the i + 1'th temporal grid point and the j + 1'th 
        spatial grid point. 
        """

        # Run checks.
        assert(len(X_hist.shape)     == 2);
        assert(X_hist.shape[1]       == self.n_x);

        # Extract only the position data.
        X_hist = X_hist[0];

        # Compute dt. 
        n_t     : int           = self.config['burgers1d']['n_t'];
        t_max   : float         = self.config['burgers1d']['t_max']; 
        dt      : float         = t_max/(n_t - 1);

        # First, approximate the spatial and temporal derivatives.
        # first axis is time index, and second index is spatial index.
        dUdx    : numpy.ndarray     = numpy.empty_like(X_hist);
        dUdt    : numpy.ndarray     = numpy.empty_like(X_hist);

        dUdx[:, :-1]    = (X_hist[:, 1:] - X_hist[:, :-1]) / self.dx;   # Use forward difference for all but the last time value.
        dUdx[:, -1]     = dUdx[:, -2];                                  # Use backwards difference for the last time value
        
        dUdt[:-1, :]    = (X_hist[1:, :] - X_hist[:-1, :]) / dt;        # Use forward difference for all but the last position
        dUdt[-1, :]     = dUdt[-2, :];                                  # Use backwards difference for the last time value.

        # compute the residual + the norm of the residual.
        r   : numpy.ndarray = dUdt - X_hist * dUdx;
        e   : float         = numpy.linalg.norm(r);

        # All done!
        return r, e;



# -------------------------------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------------------------------

def residual_burgers(un, uw, c, idxn1):

    '''

    Compute 1D Burgers equation residual for generating the data
    from https://github.com/LLNL/gLaSDI and https://github.com/LLNL/LaSDI

    '''

    f = c * (uw ** 2 - uw * uw[idxn1]);
    r = -un + uw + f;

    return r;



def jacobian(u, c, idxn1, n_x):

    '''

    Compute 1D Burgers equation jacobian for generating the data
    from https://github.com/LLNL/gLaSDI and https://github.com/LLNL/LaSDI

    '''

    diag_comp           = 1.0 + c * (2 * u - u[idxn1]);
    subdiag_comp        = numpy.ones(n_x - 1);
    subdiag_comp[:-1]   = -c * u[1:];
    data                = numpy.array([diag_comp, subdiag_comp]);
    J                   = spdiags(data, [0, -1], n_x - 1, n_x - 1, format = 'csr');
    J[0, -1]            = -c * u[0];

    return J;



def solver(u0, maxk, convergence_threshold, n_t, n_x, Dt, Dx):
    '''

    Solves 1D Burgers equation for generating the data
    from https://github.com/LLNL/gLaSDI and https://github.com/LLNL/LaSDI

    '''

    c = Dt / Dx;

    idxn1       = numpy.zeros(n_x - 1, dtype = 'int');
    idxn1[1:]   = numpy.arange(n_x - 2);
    idxn1[0]    = n_x - 2;

    u           = numpy.zeros((n_t + 1, n_x));
    u[0]        = u0;

    for n in range(n_t):
        uw = u[n, :-1].copy();
        r = residual_burgers(u[n, :-1], uw, c, idxn1);

        for k in range(maxk):
            J = jacobian(uw, c, idxn1, n_x);
            duw = spsolve(J, -r);
            uw = uw + duw;
            r = residual_burgers(u[n, :-1], uw, c, idxn1);

            rel_residual = numpy.linalg.norm(r) / numpy.linalg.norm(u[n, :-1]);
            if rel_residual < convergence_threshold:
                u[n + 1, :-1] = uw.copy();
                u[n + 1, -1] = u[n + 1, 0];
                break;

    return u;
