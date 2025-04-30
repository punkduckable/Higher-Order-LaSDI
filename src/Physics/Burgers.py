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
import  scipy;
from    scipy.sparse.linalg import  spsolve;
from    scipy.sparse        import  spdiags;
import  torch;

from    Physics             import  Physics;



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
        # there is only one spatial dimension in the 1D Burgers example, dim is also 1.
        self.qdim           : int   = 1;
        self.spatial_dim    : int   = 1;
        
        # Make sure the config dictionary is actually for Burgers' equation.
        assert('Burgers' in config);

        # Other setup
        self.n_IC           : int       = 1;
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

        where a and w are the corresponding parameter values.

        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        param: numpy.ndarray, shape (2)
            The two elements correspond to the values of the w and a parameters. self.a_idx and 
            self.w_idx tell us which index corresponds to which variable.
        

        -------------------------------------------------------------------------------------------
        Returns 
        -------------------------------------------------------------------------------------------

        u0 : list[numpy.ndarray], len = 1
            Lone element has same shape as self.X_Positions. The i'th component of this array 
            holds the value of the initial state at x = self.X_Positions[i].
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

        return [u0];



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

        X: list[torch.Tensor], len = 1
            Lone element has shape = (n_t, self.Frame_Shape), holds the FOM solution when we use
            param to define the FOM. Specifically, the [i, ...] sub-array of the returned array 
            holds the FOM solution at t_Grid[i].

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
        t_Grid  : torch.Tensor  = torch.linspace(0, t_max, n_t, dtype = torch.float32);
        if(self.Uniform_t_Grid == False):
            r               : float = 0.2*(t_Grid[1] - t_Grid[0]);
            t_adjustments           = numpy.random.uniform(low = -r, high = r, size = (n_t - 2));
            t_Grid[1:-1]            = t_Grid[1:-1] + t_adjustments;

        # Solve the PDE!
        new_X   : torch.Tensor  = [torch.Tensor(solver(u0                       = u0, 
                                                       t_Grid                   = t_Grid, 
                                                       Dx                       = self.dx, 
                                                       maxk                     = self.maxk, 
                                                       convergence_threshold    = self.convergence_threshold))];        

        # All done!
        return new_X, t_Grid;
    

    
    def residual(self, X_hist : list[numpy.ndarray]) -> tuple[numpy.ndarray, float]:
        """
        This function computes the PDE residual (difference between the left and right hand side
        of Burgers' equation when we substitute in the solution in X_hist).


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        X_hist: list[numpy.ndarray], len = 1
            A single element list of 2d numpy.ndarray object of shape (n_t, n_x), where n_t is 
            the number of points along the temporal axis (this is specified by the configuration 
            file) and n_x is the number of points along the spatial axis. The i,j element of the 
            d'th array should have the j'th component of the d'th derivative of the FOM solution 
            at the i'th time step.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        r, e
         
        r : numpy.ndarray, shape = (n_t - 2, n_x - 2)
            i, j element holds the residual at the i + 1'th temporal grid point and the j + 1'th 
            spatial grid point. 

        e : float
            the norm of r.
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
        dUdx    : numpy.ndarray = numpy.empty_like(X_hist);
        dUdt    : numpy.ndarray = numpy.empty_like(X_hist);

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
# Solve Burgers' equation
# -------------------------------------------------------------------------------------------------

def residual_burgers(   un          : numpy.ndarray, 
                        un1_guess   : numpy.ndarray, 
                        c           : float, 
                        idx_m1      : numpy.ndarray) -> numpy.ndarray:
    """
    Compute the nonlinear residual for the Burgers update.

    The residual r_i at the i'th interior point is defined by
    
    r_i = -u^n_i + u^{n + 1, guess}_i + Dt u^{n + 1, guess}*(u^{n + 1, guess}_i - u^{n + 1, guess}_{i - 1})/Dx
    
    Here, u^n_i denotes the solution to Burgers equation at the n'th time step and i'th spatial
    grid point. Likewise, u^{n + 1, guess}_i denotes our current guess for the solution at the 
    n+1'th time step. If u^{n + 1, guess} is a good approximation to u^{n + 1}, then we would 
    expect
    
        -u^n_i + u^{n + 1, guess}_i                        \approx     Dt (d/dt)u^{n + 1}_i
        (u^{n + 1, guess}_i - u^{n + 1, guess}_{i - 1})/Dx \approx     (d/dx)u^{n + 1}_i

    meaning that

        r_i \approx Dt (d/dt)u^{n + 1}_i + Dt u^{n + 1}_i (d/dx)u^{n + 1}_i
        \approx 0

    
    
    -----------------------------------------------------------------------------------------------
    Parameters
    -----------------------------------------------------------------------------------------------

    un : ndarray, shape (n_x - 1)
        Solution at previous time level (u^n), excluding the final periodic point.
    
    un1_guess : ndarray, shape (n_x - 1)
        Current Newton iterate for the next time level (u^{n + 1,guess}), excluding final point.
    
    c : float
        A CFL-like number c = Dt / Dx.

    idx_m1 : ndarray of int, shape (n_x - 1)
        Array of indices such that un1_guess[idx_m1[i]] == un1_guess[i + 1] (with periodic wrap at 
        the end).

    
    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    r : ndarray, shape (n_x-1)
        The residual vector for the Newton update.
    """

    # Checks.
    assert(isinstance(un,           numpy.ndarray));
    assert(isinstance(un1_guess,    numpy.ndarray));
    assert(isinstance(idx_m1,       numpy.ndarray));
    assert(len(un.shape)            == 1);
    assert(un.shape                 == un1_guess.shape);
    assert(un.shape                 == idx_m1.shape);

    # Compute flux difference term: f[i]    = c * (un1_guess[i]^2 - un1_guess[i] * un1_guess[i - 1])
    #                                       = c * un1_guess[i]( un1_guess[i] - un1_guess[i - 1])
    #                                       = Dt * un1_guess[i] * ( un1_guess[i] - un1_guess[i - 1])/Dx
    #                                       \approx Dt * un1_guess[i] * (d/dx)un1_guess[i]
    f : numpy.ndarray = c * (un1_guess**2 - un1_guess * un1_guess[idx_m1]);
    
    # Residual is u^n - u^{n + 1, guess} - f
    r : numpy.ndarray = -un + un1_guess + f;
    return r;



def jacobian(   u       : numpy.ndarray, 
                c       : float, 
                idx_m1  : numpy.ndarray, 
                n_x     : int) -> scipy.sparse.csr_matrix:
    """
    Assemble the Jacobian matrix of the Burgers residual w.r.t. the new-time iterate.

    The Jacobian J is a (n_x - 1) x (n_x - 1) sparse matrix with
      - diagonal entries        J_{i,i}         = 1 + c*(2*u_i - u_{i + 1})
      - sub-diagonal entries    J_{i,i - 1}     = -c * u_i   (for i > 0)
    plus a periodic coupling    J_{0, n_x - 2}  = -c * u_0.


    -----------------------------------------------------------------------------------------------
    Parameters
    -----------------------------------------------------------------------------------------------

    u : numpy.ndarray, shape (n_x-1)
        Current Newton iterate for u^{n+1}, excluding the final periodic point.
    
    c : float
        CFL-like number Dt / Dx.
    
    idx_m1 : numpy.ndarray of int, shape (n_x-1)
        Next-index mapping for periodic wrap.
    
    n_x : int
        Total number of spatial grid points (including the periodic endpoint).


    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    J : scipy.sparse.csr_matrix, shape (n_x-1, n_x-1)
        Sparse Jacobian matrix of the residual.
    """

    # Checks.
    assert(isinstance(u,           numpy.ndarray));
    assert(isinstance(idx_m1,      numpy.ndarray));
    assert(len(u.shape)            == 1);
    assert(u.shape                 == idx_m1.shape);

    # Diagonal: dr_i / du_i = 1 + c*(2*u_i - u_{i + 1})
    diag_comp           = 1.0 + c * (2 * u - u[idx_m1]);

    # Sub-diagonal (i,i-1): ∂r_i / ∂u_{i-1} = -c * u_i (for interior points)
    subdiag_comp        = numpy.ones(n_x - 1);
    subdiag_comp[:-1]   = -c * u[1:];

    # Stack bands: band 0 (diag), band -1 (sub-diag)
    data                = numpy.vstack([diag_comp, subdiag_comp]);
    J                   = spdiags(data, [0, -1], n_x - 1, n_x - 1, format = 'csr');

    # Enforce periodic wrap: coupling from u_{n_x - 2} to r_0
    J[0, -1] = -c * u[0];
    return J;


    
def solver(u0                       : numpy.ndarray,
           t_Grid                   : numpy.ndarray,
           Dx                       : float,
           maxk                     : int,
           convergence_threshold    : float):
    """
    Solves 1D Burgers equation on a uniform spatial-temporal grid.

    -----------------------------------------------------------------------------------------------
    Arguments:
    -----------------------------------------------------------------------------------------------

    u0: numpy.ndarray, shape (n_x)
        j'th component holds the value of the initial condition to Burgers equation at the j'th 
        position along the spatial grid. 

    t_Gird: numpy.ndarray, shape (n_t)
        i'th value holds the position of the i'th temporal gridline. 

    Dx: int
        The grid spacing between spatial gridlines.

    maxk: int 
        The maximum number of corrections we are willing to make at each time step.
        If the relative residual at a time step fails to drop below the convergence_threshold in 
        maxk steps, then we move onto the next time step.

    convergence_threshold: float 
        Specifying the maximum allowed relative residual at each time step. Once the residual drops 
        below this (or the iteration number passes maxk), we move onto the next time step.

    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    r : numpy.ndarray, shape (n_t, n_x)
        i, j entry holds the solution to Burgers' equation at (t_Grid[i], j*Dx).
    """

    # Checks.
    

    # Extract n_x, n_t.
    n_x         : int   = u0.shape[0];
    n_t         : int   = t_Grid.shape[0];

    # Build index array for idx_m1[idx_m1][i + 1] == idx_m1[i] (and periodic)
    idx_m1              = numpy.zeros(n_x - 1, dtype = numpy.int32);
    idx_m1[1:]          = numpy.arange(n_x - 2);    # 1 -> 0, 2 -> 1, ..., n_x - 2 -> n_x - 3
    idx_m1[0]           = n_x - 2;                  # wrap around for periodic BC

    # Allocate solution array: rows time levels 0 ... n_t, cols spatial points 0 ... n_x - 1
    u                   = numpy.zeros((n_t, n_x), dtype = numpy.float32);
    u[0, :]             = u0;

    # Time-stepping loop
    for n in range(n_t - 1):
        # Compute c = Dt/Dx for this time step.
        Dt_n        : float = (t_Grid[n + 1] - t_Grid[n]).item();
        c_n         : float = Dt_n / Dx;

        # Initialize the Newton guess un1_guess for time level n + 1 (exclude the last periodic point)
        un1_guess = u[n, :-1].copy();

        # Compute initial residual
        r : numpy.ndarray = residual_burgers(un = u[n, :-1], un1_guess = un1_guess, c = c_n, idx_m1 = idx_m1);

        # Newton iterations
        for k in range(maxk):
            # Build Jacobian at current guess
            J       : numpy.ndarray = jacobian(u = un1_guess, c = c_n, idx_m1 = idx_m1, n_x = n_x);

            # Solve Newton correction J * dun1 = -r
            dun1    : numpy.ndarray = spsolve(J, -r);

            # Update guess
            un1_guess = un1_guess + dun1;

            # Recompute residual
            r : numpy.ndarray = residual_burgers(un = u[n, :-1], un1_guess = un1_guess, c = c_n, idx_m1 = idx_m1);

            # Check convergence: relative residual drop
            rel_residual : numpy.ndarray = numpy.linalg.norm(r) / numpy.linalg.norm(u[n, :-1]);
            if rel_residual < convergence_threshold:
                # Accept update and enforce periodic BC
                u[n + 1, :-1] = un1_guess;
                u[n + 1, -1]  = un1_guess[0];
                break

        else:
            # If maxk reached without convergence, still write out the last iterate
            u[n + 1, :-1] = un1_guess;
            u[n + 1, -1]  = un1_guess[0];

    return u;
