# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

# Add the main directory to the search path.
import  os;
import  sys;
src_Path        : str   = os.path.dirname(os.path.dirname(__file__));
LD_Path         : str   = os.path.join(src_Path, "LatentDynamics");
util_Path       : str   = os.path.join(src_Path, "Utilities");
libROM_Path     : str   = os.path.join(os.path.curdir, "pylibROM");
sys.path.append(src_Path);
sys.path.append(LD_Path);
sys.path.append(util_Path);
sys.path.append(libROM_Path);

import  numpy;
import  torch;

from    Physics                         import  Physics;
import  nonlinear_elasticity_global_rom;



# -------------------------------------------------------------------------------------------------
# Elasticity class
# -------------------------------------------------------------------------------------------------

class Elasticity(Physics):    
    def __init__(self, config : dict, param_names : list[str] = None) -> None:
        """
        This is the initializer for the Elasticity class. This class essentially acts as a 
        wrapper around a non-linear elasticity solver in pylibROM.

        
        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        config: A dictionary housing the settings for the Elasticity object. This should be the 
        "physics" sub-dictionary of the configuration file. 

        param_names: A list of strings. There should be one list item for each parameter. The 
        i'th element of this list should be a string housing the name of the i'th parameter. For 
        the elasticity class, this should have one element: s.

        
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Nothing!
        """

        # Checks.
        assert(len(param_names) == 1);

        # Call the super class initializer.
        super().__init__(config, param_names);

        # The solution to the nonlinear elasticity equation is scalar valued, so the qdim is 1. 
        # Likewise, since there is only one spatial dimension in the nonlinear elasticity example, 
        # dim is also 1.
        self.qdim   : int   = 1;
        self.dim    : int   = 1;

        # Make sure the config dictionary is actually for Burgers' equation.
        assert('elasticity' in config);
        
        # Fetch variables from config.
        self.n_t                     : int       = config['elasticity']['number_of_timesteps'];  # number of time steps when solving 
        self.tmax                   : float     = config['elasticity']['final time'];           # We solve from t = 0 to t = tmax. 
        self.dt                     : float     = self.tmax / (self.n_t - 1);                # step size between successive time steps/the time step we use when solving.

        # Set up the spatial, temporal grid.
        self.x_grid : numpy.ndarray = numpy.linspace(self.xmin, self.xmax, self.spatial_grid_shape[0]);
        self.t_grid : numpy.ndarray = numpy.linspace(0, self.tmax, self.n_t);

        self.maxk                   : int   = config['burgers1d']['maxk'];                  # TODO: ??? What is this ???
        self.convergence_threshold  : float = config['burgers1d']['convergence_threshold'];

        # Determine which index corresponds to 'a' and 'w' (we pass an array of parameter values, 
        # we need this information to figure out which element corresponds to which variable).
        if (self.param_names is not None):
            if 'a' in self.param_names:
                self.a_idx = self.param_names.index('a');
            if 'w' in self.param_names:
                self.w_idx = self.param_names.index('w');
        
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

        if 'a' in self.param_names:
            a = param[self.a_idx];
        if 'w' in self.param_names:
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

        A single element list holding a 3d torch.Tensor object of shape (1, n_t, n_x), where n_t is 
        the number of points along the temporal grid and n_x is the number along the spatial grid.
        """
        
        # Fetch the initial condition.
        u0 : numpy.ndarray = self.initial_condition(param)[0];

        """
        # Solve the PDE and then reshape the result to be a 3d tensor with a leading dimension of 
        # size 1.
        X       : torch.Tensor          = torch.Tensor(solver(u0, self.maxk, self.convergence_threshold, self.n_t - 1, self.spatial_grid_shape[0], self.dt, self.dx));        
        new_X   : list[torch.Tensor]    = [X.reshape(1, self.n_t, self.spatial_grid_shape[0])];
        """

        ######## REMOVE ME   ||
        ######## REMOVE ME   ||
        ######## REMOVE ME   ||
        ######## REMOVE ME  \  /
        ######## REMOVE ME   \/
    
        # Solve the PDE and then reshape the result to be a 3d tensor with a leading dimension of 
        # size 1.
        X       : torch.Tensor  = torch.Tensor(solver(u0, self.maxk, self.convergence_threshold, self.n_t - 1, self.spatial_grid_shape[0], self.dt, self.dx));
        V       : torch.Tensor  = Derivative1_Order4(X, h = self.dt);
        
        X       : torch.Tensor  = X.reshape(1, self.n_t, self.spatial_grid_shape[0]);
        V       : torch.Tensor  = V.reshape(1, self.n_t, self.spatial_grid_shape[0]);

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

        Xhist: A 2d numpy.ndarray object of shape (n_t, n_x), where n_t is the number of points along
        the temporal axis and n_x is the number of points along the spatial axis. The i,j element of
        this array should have the j'th component of the solution at the i'th time step.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        A two element tuple. The first is a numpy.ndarray object of shape (n_t - 2, n_x - 2) whose 
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
