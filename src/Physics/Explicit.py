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
        assert(isinstance(param_names, list));
        assert(len(param_names) == 2);
        assert('a' in param_names);
        assert('w' in param_names);

        # Call the super class initializer.
        super().__init__(config         = config, 
                         param_names    = param_names, 
                         Uniform_t_Grid = True);
        
        # The functions we deal with are scalar valued. Likewise, since there is only one spatial 
        # dimension, dim is also 1. 
        self.qdim           : int   = 1;
        self.spatial_dim    : int   = 1;

        # Make sure the config dictionary is actually for the explicit physics model.
        assert('explicit' in config);

        # Set up spatial variables
        self.n_IC                   : int       = 2;
        self.n_x                    : int       = config['explicit']['n_x'];    
        self.x_min                  : float     = config['explicit']['x_min'];
        self.x_max                  : float     = config['explicit']['x_max'];
        self.dx                     : float     = (self.x_max - self.x_min)/(self.n_x - 1);
        self.Frame_Shape            : list[int] = [self.n_x];                       # number of grid points along each spatial axis

        # Set up X_Positions. For the Explicit class, X_Positions is 1D and has shape (n_x).
        self.X_Positions : numpy.ndarray = numpy.linspace(self.x_min, self.x_max, self.n_x, dtype = numpy.float32);
     
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

        A list of 1d numpy.ndarray objects, each of shape length self.n_x (the number of grid 
        points along the spatial axis). The i'th element holds the initial state of the i'th time 
        derivative of the FOM state.
        """

        # Checks.
        assert(isinstance(param, numpy.ndarray));
        assert(len(param.shape) == 1);
        assert(param.shape[0]   == self.n_p);

        # Fetch the parameter values.
        a   : float             = param[self.a_idx];
        w   : float             = param[self.w_idx];  

        # Compute the initial condition and return!
        X   : numpy.ndarray     = self.X_Positions;
        u0  : numpy.ndarray     = numpy.multiply(numpy.sin(2*X), numpy.exp(-a*numpy.multiply(X, X)));
        v0  : numpy.ndarray     = numpy.multiply(-1*numpy.cos(2*X) + 0.1*w*numpy.cos(40*X), numpy.exp(-a*numpy.multiply(X, X)));
        return [u0, v0];
    


    def solve(self, param : numpy.ndarray) -> tuple[list[torch.Tensor], torch.Tensor]:
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
        
        A two element tuple: X, t_Grid.

        X is a 2 element list holding the displacement and velocity of the FOM solution when we 
        use param. Each element is a torch.Tensor object of shape (n_t, self.Frame_Shape), where 
        n_t is the number of time steps when we solve the FOM using param for the IC parameters.

        t_Grid is a 1d torch.Tensor object whose i'th element holds the i'th time value at which
        we have an approximation to the FOM solution (the time value associated with X[0, i, ...]).
        """
       
        assert(isinstance(param, numpy.ndarray));
        assert(len(param.shape) == 1);
        assert(param.shape[0]   == self.n_p);

        # Fetch the parameter values.
        a   : float             = param[self.a_idx];
        w   : float             = param[self.w_idx]; 

        # Make the t_grid
        n_t     : int           = self.config['explicit']['n_t'];
        t_max   : float         = self.config['explicit']['t_max']; # We solve from t = 0 to t = t_max. 
        t_Grid  : numpy.ndarray = numpy.linspace(0, t_max, n_t, dtype = numpy.float32);

        # Make the t, x meshgrids.
        t_mesh, x_mesh          = numpy.meshgrid(t_Grid, self.X_Positions, indexing = 'ij');
        t_mesh                  = torch.tensor(t_mesh);         # shape (n_t, n_x)
        x_mesh                  = torch.tensor(x_mesh);         # shape (n_t, n_x)

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

        # All done!
        return [U, V], torch.Tensor(t_Grid);
        

    
    def residual(self, X_hist : list[numpy.ndarray]) -> tuple[numpy.ndarray, float]:
        """
        Because there is no governing PDE for this Physics model, "residual" doesn't make a 
        whole lot of sense for this class. Thus, we return an array of zeros whose shape matches
        that of X_hist.
        

        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        X_hist: A list of 2d numpy.ndarray object of shape (n_t, n_x), where n_t is the number of 
        points along the temporal axis (this is specified by the configuration file) and n_x is the 
        number of points along the spatial axis. The i,j element of the d'th array should have the 
        j'th component of the d'th derivative of the fom solution at the i'th time step.

        
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        A two element tuple. The first is a numpy.ndarray object of shape (n_t - 2, n_x - 2) whose 
        i, j element holds the residual at the i + 1'th temporal grid point and the j + 1'th 
        spatial grid point. 
        """

        # Run checks.
        assert(len(X_hist[0].shape)     == 2);
        assert(X_hist[0].shape[1]       == self.n_x);

        # compute the residual + the norm of the residual.
        r   : numpy.ndarray = numpy.zeros_like(X_hist[0]);
        e   : float         = numpy.linalg.norm(r);

        # All done!
        return r, e;
