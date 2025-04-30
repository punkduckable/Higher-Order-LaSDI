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
from    FiniteDifference    import  Derivative1_Order4, Derivative1_Order2_NonUniform;
from    Burgers             import  solver;



# -------------------------------------------------------------------------------------------------
# Burgers class
# -------------------------------------------------------------------------------------------------

class Burgers(Physics):
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

        config: dict
            A dictionary housing the settings for the Burgers object. This should be the 
            "physics" sub-dictionary of the configuration file. 

        param_names: list[str]
            There should be one list item for each parameter. The i'tj
            element of this list should be a string housing the name of the i'th parameter. For the 
            Burgers class, this should have two elements: a and w. 

        
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Nothing!
        """

        # Checks
        assert(isinstance(param_names, list));
        assert(len(param_names) == 2);
        assert('a' in param_names);
        assert('w' in param_names);

        # Call the super class initializer.
        super().__init__(config         = config, 
                         param_names    = param_names, 
                         Uniform_t_Grid = True);

        # The solution to Burgers' equation is scalar valued, so the qdim is 1. Likewise, since 
        # there is only one spatial dimension in the 1D burgers example, dim is also 1.
        self.qdim           : int   = 1;
        self.spatial_dim    : int   = 1;
        
        # Make sure the config dictionary is actually for Burgers' equation.
        assert('Burgers' in config);

        # Other setup
        self.n_IC           : int       = 2;
        self.n_x            : int       = config['Burgers']['n_x'];
        self.Frame_Shape    : list[int] = [self.n_x];                                   # number of grid points along each spatial axis
        self.x_min          : float     = config['Burgers']['x_min'];                   # Minimum value of the spatial variable in the problem domain
        self.x_max          : float     = config['Burgers']['x_max'];                   # Maximum value of the spatial variable in the problem domain
        self.dx             : float     = (self.x_max - self.x_min) / (self.n_x - 1);   # Spacing between grid points along the spatial axis.
        assert(self.dx > 0.);

        # Set up X_Positions. For the Burgers class, X_Positions is 1D and has shape (n_x).
        self.X_Positions : numpy.ndarray = numpy.linspace(self.x_min, self.x_max, self.n_x, dtype = numpy.float32);

        # ???
        self.maxk                   : int   = config['Burgers']['maxk'];                  # TODO: ??? What is this ???
        self.convergence_threshold  : float = config['Burgers']['convergence_threshold'];

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
            v(0, x) = (d/dt)u(t, x)|_{t = 0}

        where a and w are the corresponding parameter values. We compute v(0, x) by solving forward
        a few time steps and the computing the time derivative using finite differences.

        
        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        param: numpy.ndarray, shape (2)
            The two elements correspond to the values of the w and a parameters. self.a_idx and 
            self.w_idx tell us which index corresponds to which variable.
        

        -------------------------------------------------------------------------------------------
        Returns 
        -------------------------------------------------------------------------------------------

        u0 : list[numpy.ndarray], len = 2
            i'th element is a ndarray with the same shape as self.X_Positions whose j'th element 
            holds the i'th derivative of the initial state at the position self.X_Positions[j] when 
            we use param to define the initial condition functions.
        """

        # Checks.
        assert(isinstance(param, numpy.ndarray));
        assert(len(param.shape) == 1);
        assert(param.shape[0]   == self.n_p);
        
    
        # Fetch the parameter values.
        a   : float     = param[self.a_idx];
        w   : float     = param[self.w_idx];  

        # Get the initial displacement.
        u0  : numpy.ndarray     = a * numpy.exp( -((self.X_Positions) ** 2) / 2 / w / w);

        # Calculate dt.
        n_t     : int           = self.config['Burgers']['n_t'];
        t_max   : float         = self.config['Burgers']['t_max']; 
        dt      : float         = t_max/(n_t - 1);

        # Solve forward a few time steps, use that to compute the derivative.
        t_Grid  : numpy.ndarray         = numpy.linspace(start = 0, stop = 4*dt, num = 5);  # shape (5)
        D       : numpy.ndarray         = solver(   u0                      = u0, 
                                                    t_Grid                  = t_Grid, 
                                                    Dx                      = self.dx, 
                                                    maxk                    = self.maxk, 
                                                    convergence_threshold   = self.convergence_threshold);
        V       : numpy.ndarray         = Derivative1_Order4(torch.Tensor(D), h = dt);
        
        # Get the ICs from the solution.
        u0                              = D[0, :];
        v0                              = V[0, :];
            
        # All done!
        return [u0, v0];



    def solve(self, param : numpy.ndarray) -> tuple[list[torch.Tensor], torch.Tensor]:
        """
        Solves the 1d burgers equation when the IC uses the parameters in the param array.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        param: numpy.ndarray, shape = (2)
            Holds the values of the w and a parameters. self.a_idx and self.w_idx tell us which 
            index corresponds to which variable.
        

        -------------------------------------------------------------------------------------------
        Returns 
        -------------------------------------------------------------------------------------------
        
        X, t_Grid

        X: list[torch.Tensor], len = 2
            i'th element has shape = (n_t, self.Frame_Shape), holds the i'th derivative of the FOM 
            solution when we use param to define the initial condition function. Specifically, the 
            [j, ...] sub-array of the returned array holds the i'th derivative of the FOM solution 
            at t_Grid[j].

        t_Grid: torch.Tensor, shape = (n_t)
            i'th element holds the i'th time value at which we have an approximation to the FOM 
            solution (the time value associated with X[i, ...]).
        """

        assert(isinstance(param, numpy.ndarray));
        assert(len(param.shape) == 1);
        assert(param.shape[0]   == self.n_p);
        

        # Fetch the initial condition.
        u0 : numpy.ndarray = self.initial_condition(param)[0];
        
        # Compute dt. Set up the t_Grid.
        n_t     : int           = self.config['Burgers']['n_t'];
        t_max   : float         = self.config['Burgers']['t_max']; 
        dt      : float         = t_max/(n_t - 1);
        t_Grid  : torch.Tensor  = torch.linspace(0, t_max, n_t, dtype = torch.float32);
        if(self.Uniform_t_Grid == False):
            r               : float = 0.2*(t_Grid[1] - t_Grid[0]);
            t_adjustments           = numpy.random.uniform(low = -r, high = r, size = (n_t - 2));
            t_Grid[1:-1]            = t_Grid[1:-1] + t_adjustments;



        # Solve the PDE!
        X       : torch.Tensor  = torch.Tensor(solver(u0 = u0, t_Grid = t_Grid, Dx = self.dx, maxk = self.maxk, convergence_threshold = self.convergence_threshold));
        if(self.Uniform_t_Grid  == True):
            V   : torch.Tensor  = Derivative1_Order4(X, h = dt);
        else:
            V   : torch.Tensor  = Derivative1_Order2_NonUniform(X, t_Grid = t_Grid);

        X       : torch.Tensor  = X.reshape(n_t, self.n_x);
        V       : torch.Tensor  = V.reshape(n_t, self.n_x);

        new_X   : list[torch.Tensor]    = [X, V];

        # All done!
        return new_X, t_Grid;
    

    
    def residual(self, X_hist : list[numpy.ndarray]) -> tuple[numpy.ndarray, float]:
        """
        This function computes the PDE residual (difference between the left and right hand side
        of Burgers' equation when we substitute in the solution in X_hist).


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        X_hist : list[numpy.ndarray], len = 2
            d'th element has shape (n_t, n_x), where n_t is the number of points along the 
            temporal axis (this is specified by the configuration file) and n_x is the number of 
            points along the spatial axis. The i,j element of the d'th array should hold the j'th 
            component of the d'th derivative of the FOM solution at the i'th time step.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        r, e

        r : numpy.ndarray, shape = (n_t - 2, n_x - 2)
            i, j element holds the residual at the i + 1'th temporal grid point and the j + 1'th 
            spatial grid point. 

        e : float 
            The norm of r. 
        """

        # Run checks.
        assert(len(X_hist.shape)     == 2);
        assert(X_hist.shape[1]       == self.n_x);

        # Extract only the position data.
        X_hist = X_hist[0];

        # Compute dt. 
        n_t     : int           = self.config['Burgers']['n_t'];
        t_max   : float         = self.config['Burgers']['t_max']; 
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
