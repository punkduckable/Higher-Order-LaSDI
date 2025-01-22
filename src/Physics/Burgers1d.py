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

from    InputParser         import  InputParser;
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


    
    def __init__(self, cfg : dict, param_name_list : list[str] = None) -> None:
        """
        This is the initializer for the Burgers Physics class. This class essentially acts as a 
        wrapper around a 1D Burgers solver.

        
        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        cfg: A dictionary housing the settings for the Burgers object. This should be the "physics"
        sub-dictionary of the configuration file. 

        param_name_list: A list of strings. There should be one list item for each parameter. The 
        i'th element of this list should be a string housing the name of the i'th parameter.

        
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Nothing!
        """

        # Call the super class initializer.
        super().__init__(cfg, param_name_list);

        # The solution to Burgers' equation is scalar valued, so the qdim is 1. Likewise, since 
        # there is only one spatial dimension in the 1D burgers example, dim is also 1.
        self.qdim   : int   = 1;
        self.dim    : int   = 1;

        # Make sure the configuration dictionary is actually for Burgers' equation.
        assert('burgers1d' in cfg);
        
        # Now, get a parser for cfg.
        input_parser : InputParser = InputParser(cfg['burgers1d'], name = "burgers1d_input");

        # Fetch variables from the configuration. 
        self.nt                     : int       = input_parser.getInput(['number_of_timesteps'],  datatype = int);      # number of time steps when solving 
        self.spatial_grid_shape     : list[int] = input_parser.getInput(['grid_shape'],            datatype = list);    # number of grid points along each spatial axis
        self.spatial_qgrid_shape    : list[int] = self.spatial_grid_shape;
        
        # If there are n spatial dimensions, then the grid needs to have n axes (one for each 
        # dimension). Make sure this is the case.
        assert(self.dim == len(self.spatial_grid_shape));

        # Fetch more variables from the 
        self.xmin   = input_parser.getInput(['xmin'], datatype = float);    # Minimum value of the spatial variable in the problem domain
        self.xmax   = input_parser.getInput(['xmax'], datatype = float);    # Maximum value of the spatial variable in the problem domain
        self.dx     = (self.xmax - self.xmin) / (self.spatial_grid_shape[0] - 1);    # Spacing between grid points along the spatial axis.
        assert(self.dx > 0.)

        self.tmax   : float     = input_parser.getInput(['simulation_time']);  # Final simulation time. We solve form t = 0 to t = tmax
        self.dt     : float     = self.tmax / (self.nt - 1);                # step size between successive time steps/the time step we use when solving.

        # Set up the spatial, temporal grid.
        self.x_grid : numpy.ndarray = numpy.linspace(self.xmin, self.xmax, self.spatial_grid_shape[0]);
        self.t_grid : numpy.ndarray = numpy.linspace(0, self.tmax, self.nt);

        self.maxk                   : int   = input_parser.getInput(['maxk'],                   fallback = 10);     # TODO: ??? What is this ???
        self.convergence_threshold  : float = input_parser.getInput(['convergence_threshold'],  fallback = 1.e-8);

        # Determine which index corresponds to 'a' and 'w' (we pass an array of parameter values, 
        # we need this information to figure out which element corresponds to which variable).
        if (self.param_name_list is not None):
            if 'a' in self.param_name_list:
                self.a_idx = self.param_name_list.index('a');
            if 'w' in self.param_name_list:
                self.w_idx = self.param_name_list.index('w');
        
        # All done!
        return;
    


    def initial_condition(self, param : numpy.ndarray) -> list[numpy.ndarray]:
        """
        Evaluates the initial condition along the spatial grid. For this class, we use the 
        following initial condition:
            u(x, 0) = a*exp(-x^2 / (2*w^2))
        where a and w are the corresponding parameter values.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        param: A 1d numpy.ndarray object with two elements corresponding to the values of the w 
        and a parameters. self.a_idx and self.w_idx tell us which index corresponds to which 
        variable.
        

        -------------------------------------------------------------------------------------------
        Returns 
        -------------------------------------------------------------------------------------------

        A list of 1d numpy.ndarray objects, each of shape length self.spatial_grid_shape[0] (the 
        number of grid points along the spatial axis). The i'th element holds the initial state of 
        the i'th time derivative of the FOM state.
        """

        # Fetch the parameter values.
        a, w = 1.0, 1.0

        if 'a' in self.param_name_list:
            a = param[self.a_idx];
        if 'w' in self.param_name_list:
            w = param[self.w_idx];  

        # Compute the initial condition and return!
        return [a * numpy.exp(- self.x_grid ** 2 / 2 / w / w)];
    


    def solve(self, param : numpy.ndarray) -> list[torch.Tensor]:
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

        A single element list holding a 3d torch.Tensor object of shape (1, nt, nx), where nt is 
        the number of points along the temporal grid and nx is the number along the spatial grid.
        """
        
        # Fetch the initial condition.
        u0 : numpy.ndarray = self.initial_condition(param)[0];

        """
        # Solve the PDE and then reshape the result to be a 3d tensor with a leading dimension of 
        # size 1.
        X       : torch.Tensor          = torch.Tensor(solver(u0, self.maxk, self.convergence_threshold, self.nt - 1, self.spatial_grid_shape[0], self.dt, self.dx));        
        new_X   : list[torch.Tensor]    = [X.reshape(1, self.nt, self.spatial_grid_shape[0])];
        """

        ######## REMOVE ME   ||
        ######## REMOVE ME   ||
        ######## REMOVE ME   ||
        ######## REMOVE ME  \  /
        ######## REMOVE ME   \/
    
        # Solve the PDE and then reshape the result to be a 3d tensor with a leading dimension of 
        # size 1.
        X       : torch.Tensor  = torch.Tensor(solver(u0, self.maxk, self.convergence_threshold, self.nt - 1, self.spatial_grid_shape[0], self.dt, self.dx));
        V       : torch.Tensor  = Derivative1_Order4(X, h = self.dt);
        
        X       : torch.Tensor  = X.reshape(1, self.nt, self.spatial_grid_shape[0]);
        V       : torch.Tensor  = V.reshape(1, self.nt, self.spatial_grid_shape[0]);

        new_X   : list[torch.Tensor]    = [X, V];

        ######## REMOVE ME   /\
        ######## REMOVE ME  /  \
        ######## REMOVE ME   ||
        ######## REMOVE ME   ||
        ######## REMOVE ME   ||

        # All done!
        return new_X;
    


    def export(self) -> dict:
        """
        Returns a dictionary housing self's internal state. You can use this dictionary to 
        effectively serialize self.
        """

        dict_ : dict = {'t_grid' : self.t_grid, 'x_grid' : self.x_grid, 'dt' : self.dt, 'dx' : self.dx};
        return dict_;
    

    
    def residual(self, Xhist : numpy.ndarray) -> tuple[numpy.ndarray, float]:
        """
        This function computes the PDE residual (difference between the left and right hand side
        of Burgers' equation when we substitute in the solution in Xhist).


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        Xhist: A 2d numpy.ndarray object of shape (nt, nx), where nt is the number of points along
        the temporal axis and nx is the number of points along the spatial axis. The i,j element of
        this array should have the j'th component of the solution at the i'th time step.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        A two element tuple. The first is a numpy.ndarray object of shape (nt - 2, nx - 2) whose 
        i, j element holds the residual at the i + 1'th temporal grid point and the j + 1'th 
        spatial grid point. 
        """
        
        # First, approximate the spatial and teporal derivatives.
        # first axis is time index, and second index is spatial index.
        dUdx = (Xhist[:, 1:] - Xhist[:, :-1]) / self.dx;
        dUdt = (Xhist[1:, :] - Xhist[:-1, :]) / self.dt;

        # compute the residual + the norm of the residual.
        r   : numpy.ndarray = dUdt[:, :-1] - Xhist[:-1, :-1] * dUdx[:-1, :];
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



def jacobian(u, c, idxn1, nx):

    '''

    Compute 1D Burgers equation jacobian for generating the data
    from https://github.com/LLNL/gLaSDI and https://github.com/LLNL/LaSDI

    '''

    diag_comp           = 1.0 + c * (2 * u - u[idxn1]);
    subdiag_comp        = numpy.ones(nx - 1);
    subdiag_comp[:-1]   = -c * u[1:];
    data                = numpy.array([diag_comp, subdiag_comp]);
    J                   = spdiags(data, [0, -1], nx - 1, nx - 1, format = 'csr');
    J[0, -1]            = -c * u[0];

    return J;



def solver(u0, maxk, convergence_threshold, nt, nx, Dt, Dx):
    '''

    Solves 1D Burgers equation for generating the data
    from https://github.com/LLNL/gLaSDI and https://github.com/LLNL/LaSDI

    '''

    c = Dt / Dx;

    idxn1       = numpy.zeros(nx - 1, dtype = 'int');
    idxn1[1:]   = numpy.arange(nx - 2);
    idxn1[0]    = nx - 2;

    u           = numpy.zeros((nt + 1, nx));
    u[0]        = u0;

    for n in range(nt):
        uw = u[n, :-1].copy();
        r = residual_burgers(u[n, :-1], uw, c, idxn1);

        for k in range(maxk):
            J = jacobian(uw, c, idxn1, nx);
            duw = spsolve(J, -r);
            uw = uw + duw;
            r = residual_burgers(u[n, :-1], uw, c, idxn1);

            rel_residual = numpy.linalg.norm(r) / numpy.linalg.norm(u[n, :-1]);
            if rel_residual < convergence_threshold:
                u[n + 1, :-1] = uw.copy();
                u[n + 1, -1] = u[n + 1, 0];
                break;

    return u;
