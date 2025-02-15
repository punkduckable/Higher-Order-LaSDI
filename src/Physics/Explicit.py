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
import  torch;

from    Physics                         import  Physics;



# -------------------------------------------------------------------------------------------------
# Explicit class
# -------------------------------------------------------------------------------------------------

class Explicit(Physics):    
    def __init__(self, config : dict, param_names : list[str] = None) -> None:
        """
        This is the initializer for the Explicit class. This class essentially acts as a wrapper
        around the following function of t and x:
            u(t, x) = [sin(2x-t) + 0.1 sin(w t) cos( 40x + 2t)] exp(-a x^2)

        
        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        config: A dictionary housing the settings for the Explicit object. This should be the 
        "physics" sub-dictionary of the configuration file. 

        param_names: A list of strings. There should be one list item for each parameter. The 
        i'th element of this list should be a string housing the name of the i'th parameter. 

        
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Nothing!
        """

        # Checks.
        assert(len(param_names) == 2);
        assert('a' in param_names);
        assert('w' in param_names);

        # Call the super class initializer.
        super().__init__(config, param_names);

        # The functions we deal with are scalar valued. Likewise, since there is only one spatial 
        # dimension, dim is also 1. 
        self.qdim   : int   = 1;
        self.dim    : int   = 1;

        # Make sure the config dictionary is actually for the explicit physics model.
        assert('explicit' in config);
        
        # Set up time variables.
        self.n_t                    : int       = config['explicit']['n_t'];
        self.t_max                  : float     = config['explicit']['t_max'];      # We solve from t = 0 to t = t_max. 
        self.dt                     : float     = self.t_max / (self.n_t - 1);      # step size between successive time steps/the time step we use when solving.
        
        # Set up spatial variables
        self.n_x                    : int       = config['explicit']['n_x'];    
        self.x_min                  : float     = config['explicit']['x_min'];
        self.x_max                  : float     = config['explicit']['x_max'];
        self.dx                     : float     = (self.x_max - self.x_min)/(self.n_x - 1);
        self.spatial_grid_shape     : list[int] = [self.n_x];                       # number of grid points along each spatial axis
        self.spatial_qgrid_shape    : list[int] = self.spatial_grid_shape;

        # If there are n spatial dimensions, then the grid needs to have n axes (one for each 
        # dimension). Make sure this is the case.
        assert(self.dim == len(self.spatial_grid_shape));

        # Set up the x, t grids.
        self.x_grid : numpy.ndarray = numpy.linspace(self.x_min, self.x_max, self.n_x, dtype = numpy.float32);
        self.t_grid : numpy.ndarray = numpy.linspace(0, self.t_max, self.n_t, dtype = numpy.float32);

        # Determine which index corresponds to 'a' and 'w' (we pass an array of parameter values, 
        # we need this information to figure out which element corresponds to which variable).
        self.a_idx = self.param_names.index('a');
        self.w_idx = self.param_names.index('w');
        
        # All done!
        return;
    


    def initial_condition(self, param : numpy.ndarray) -> list[numpy.ndarray]:
        """
        Evaluates the initial condition along the spatial grid. In this case,
            u(t, x) = [sin(2x-t) + 0.1 sin(w t) cos( 40x + 2t)] exp(-a x^2)
        Thus,
            v(t, x) = (d/dt)u(t, x)
                    = [-cos(2x - t) + 0.1 w cos(w t) cos( 40 x + 2t) - 0.2 sin(w t)sin( 40x + 2t) ] exp(-a x^2)
        Which means that
            u(0, x) = [sin(2x)]exp(-a x^2)
            v(0, x) = [-cos(2x) + 0.1 w cos( 40 x) ]exp(-a x^2)


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
        a   : float             = param[self.a_idx];
        w   : float             = param[self.w_idx];  

        # Compute the initial condition and return!
        u0  : numpy.ndarray     = numpy.multiply(numpy.sin(2*self.x_grid), numpy.exp(-a*numpy.multiply(self.x_grid, self.x_grid)));
        v0  : numpy.ndarray     = numpy.multiply(-1*numpy.cos(2*self.x_grid) + 0.1*w*numpy.cos(40*self.x_grid), numpy.exp(-a*numpy.multiply(self.x_grid, self.x_grid)));
        return [u0, v0];
    


    def solve(self, param : numpy.ndarray) -> list[torch.Tensor]:
        """
        Evaluates the function u(t, x) (see __init__ docstring) on the t, x grids using the 
        parameters in param.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        param: A 1d numpy.ndarray object with two elements corresponding to the values of the w 
        and a parameters. self.a_idx and self.w_idx tell us which index corresponds to which 
        variable.
        

        -------------------------------------------------------------------------------------------
        Returns 
        -------------------------------------------------------------------------------------------

        A two element list. Each element is a 3d torch.Tensor object of shape (1, n_t, n_x), where 
        n_t is the number of points along the temporal grid and n_x is the number along the spatial 
        grid.
        """

        # Fetch the parameter values.
        a   : float             = param[self.a_idx];
        w   : float             = param[self.w_idx]; 

        # Make the t, x meshgrids.
        t_mesh, x_mesh          = numpy.meshgrid(self.t_grid, self.x_grid, indexing = 'ij');
        t_mesh                  = torch.tensor(t_mesh);
        x_mesh                  = torch.tensor(x_mesh);

        # We know that
        #   u(t, x) = [sin(2x-t) + 0.1 sin(w t) cos(40x + 2t)] exp(-a x^2)
        # Thus,
        #   v(t, x) = (d/dt)u(t, x)
        #            = [-cos(2x - t) + 0.1 w cos(w t) cos(40x + 2t) - 0.2 sin(w t)sin(40x + 2t) ] exp(-a x^2)
        U   : torch.Tensor  = torch.multiply(torch.sin(2*x_mesh - t_mesh) +                                                     # [ sin(2x - t)
                                             0.1*torch.multiply(torch.sin(w*t_mesh), torch.cos(40*x_mesh + 2*t_mesh)),          #   0.1*sin(w t)cos(40x + 2t) ]*
                                             torch.exp(-a*torch.multiply(x_mesh, x_mesh)));                                     # exp(-a x*2)
        
        V   : torch.Tensor  = torch.multiply(-1*torch.cos(2*x_mesh - t_mesh) +                                                  # [ -2 cos(2x - t) + 
                                             (0.1*w)*torch.multiply(torch.cos(w*t_mesh), torch.cos(40*x_mesh + 2*t_mesh)) -     #   0.1*w*cos(w t)cos(40x + 2t) - 
                                             0.2*torch.multiply(torch.sin(w*t_mesh), torch.sin(40*x_mesh + 2*t_mesh)),          #   0.2*sin(w t)sin(40x + 2t) ] *
                                             torch.exp(-a*torch.multiply(x_mesh, x_mesh)));                                     # exp(-a x^2)

        # Give U, V them the correct shape.
        U = U.reshape((1,) + U.shape);
        V = V.reshape((1,) + V.shape);

        # All done!
        return [U, V];
    


    def export(self) -> dict:
        """
        Returns a dictionary housing self's internal state. You can use this dictionary to 
        effectively serialize self.
        """

        dict_ : dict = {'t_grid'    : self.t_grid, 
                        'x_grid'    : self.x_grid, 
                        'dt'        : self.dt, 
                        'dx'        : self.dx};
        return dict_;
    

    
    def residual(self, X_hist : list[numpy.ndarray]) -> tuple[numpy.ndarray, float]:
        """
        Because there is no governing PDE for this Physics model, "residual" doesn't make a 
        whole lot of sense for this class. Thus, we return an array of zeros whose shape matches
        that of X_hist.
        

        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        X_hist: A list of 2d numpy.ndarray object of shape (n_t, n_x), where n_t is the number of 
        points along the temporal axis and n_x is the number of points along the spatial axis. The 
        i,j element of the d'th array should have the j'th component of the d'th derivative of the 
        fom solution at the i'th time step.

        
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        A two element tuple. The first is a numpy.ndarray object of shape (n_t - 2, n_x - 2) whose 
        i, j element holds the residual at the i + 1'th temporal grid point and the j + 1'th 
        spatial grid point. 
        """

        # Run checks.
        assert(len(X_hist[0].shape)     == 2);
        assert(X_hist[0].shape[0]       == self.n_t);
        assert(X_hist[0].shape[1]       == self.n_x);

        # compute the residual + the norm of the residual.
        r   : numpy.ndarray = numpy.zeros_like(X_hist[0]);
        e   : float         = numpy.linalg.norm(r);

        # All done!
        return r, e;
