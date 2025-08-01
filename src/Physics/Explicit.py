# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

import  numpy;
import  torch;

from    Physics                         import  Physics;



# -------------------------------------------------------------------------------------------------
# Explicit class
# -------------------------------------------------------------------------------------------------

class Explicit(Physics):    
    def __init__(self, config : dict, param_names : list[str]) -> None:
        """
        This is the initializer for the Explicit class. This class essentially acts as a wrapper
        around the following function of t and x:
            
                  u(t, x)   =  A [ sin(2x - t) + 0.2 cos( (10x + t)cos(w t) ) ]                                 exp(-0.3*x^2)
            (d/dt)u(t, x)   = -A [ cos(2x - t) + 0.2 sin( (10x + t)cos(w t) )[ sin(w t) + w(10x + t)cos(w t)] ] exp(-0.3*x^2)

        
        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        config: dict 
            A dictionary housing the settings for the Explicit object. This should be the "physics" 
            sub-dictionary of the configuration file. 

        param_names: list[str], len = 2
            i'th element be a string housing the name of the i'th parameter. 

        
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Nothing!
        """

        # Checks.
        assert(isinstance(param_names, list));
        assert(len(param_names) == 2);
        assert('A' in param_names);
        assert('w' in param_names);

        # Make sure the config dictionary is actually for the Explicit physics model.
        assert('Explicit' in config);

        # Set up spatial variables
        self.n_x                    : int       = config['Explicit']['n_x'];    
        self.x_min                  : float     = config['Explicit']['x_min'];
        self.x_max                  : float     = config['Explicit']['x_max'];
        self.dx                     : float     = (self.x_max - self.x_min)/(self.n_x - 1);

        # Call the super class initializer.
        super().__init__(config         = config, 
                         spatial_dim    = 1,            # Since there is only one spatial dimension, spatial_dim is also 1.
                         X_Positions    = numpy.linspace(self.x_min, self.x_max, self.n_x, dtype = numpy.float32),
                         Frame_Shape    = [self.n_x],
                         param_names    = param_names, 
                         Uniform_t_Grid = config['Explicit']['uniform_t_grid'],
                         n_IC           = 2);
     
        # Determine which index corresponds to 'a' and 'w' (we pass an array of parameter values, 
        # we need this information to figure out which element corresponds to which variable).
        self.A_idx = self.param_names.index('A');
        self.w_idx = self.param_names.index('w');
        
        # All done!
        return;
    


    def initial_condition(self, param : numpy.ndarray) -> list[numpy.ndarray]:
        """
        Evaluates the initial condition at the points in self.X_Positions. In this case,
        
            u(t, x) =  A [ sin(2x - t) + 0.2 cos( (10x + t)cos(w t) ) ] exp(-0.3*x^2)
        
        Thus,
            
            v(t, x) = (d/dt)u(t, x)
                    = -A [ cos(2x - t) + 0.2 sin( (10x + t)cos(w t) )[ cos(w t) - w(10x + t)sin(w t)] ] exp(-0.3*x^2)
        
        Which means that
        
            u(0, x) =  A [sin(2x) + 0.2*cos(10x)]exp(-0.3*x^2)
            v(0, x) = -A [cos(2x) - 0.2*sin(10x)]exp(-0.3*x^2)


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        param : numpy.ndarray, shape = (self.n_p)
            The two elements corresponding to the values of the w and a parameters. self.a_idx and 
            self.w_idx tell us which index corresponds to which variable.
        

        -------------------------------------------------------------------------------------------
        Returns 
        -------------------------------------------------------------------------------------------

        X0 : list[numpy.ndarray], len = self.n_IC
            i'th element has shape self.n_x (the number of grid points along the spatial axis) and
            holds the i'th derivative of the initial state when we use param to define the FOM.
        """

        # Checks.
        assert(isinstance(param, numpy.ndarray));
        assert(self.X_Positions is not None);
        assert(len(param.shape) == 1);
        assert(param.shape[0]   == self.n_p);

        # Fetch the parameter values.
        A   : float             = param[self.A_idx];
        w   : float             = param[self.w_idx];  

        # Compute the initial condition and return!
        X   : numpy.ndarray     = self.X_Positions;
        u0  : numpy.ndarray     =  A*numpy.multiply(numpy.sin(2*X) + 0.2*numpy.cos(10*X), numpy.exp(-0.3*numpy.multiply(X, X)));
        v0  : numpy.ndarray     = -A*numpy.multiply(numpy.cos(2*X) - 0.2*numpy.sin(10*X), numpy.exp(-0.3*numpy.multiply(X, X)));
        return [u0, v0];
    


    def solve(self, param : numpy.ndarray) -> tuple[list[torch.Tensor], torch.Tensor]:
        """
        Evaluates u(t, x) and v(t, x) (see __init__ docstring) on the t, x grids using the 
        parameters in param.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        param : numpy.ndarray, shape = (2)
            The two elements correspond to the values of the w and a parameters. self.a_idx and 
            self.w_idx tell us which index corresponds to which variable.
        

        -------------------------------------------------------------------------------------------
        Returns 
        -------------------------------------------------------------------------------------------
        
        U, t_Grid.

        U : list[torch.Tensor], len = 2
            Holds the displacement (first element) and velocity (second element) of the FOM 
            solution when we use param to define the FOM. Each element is a torch.Tensor object of 
            shape (n_t, self.Frame_Shape), where n_t is the number of time steps when we solve the 
            FOM using param.

        t_Grid : torch.Tensor, shape = (n_t)
            i'th element holds the i'th time value at which we have an approximation to the FOM 
            solution (the time value associated with X[0, i, ...]).
        """
       
        assert(isinstance(param, numpy.ndarray));
        assert(self.X_Positions is not None);
        assert(len(param.shape) == 1);
        assert(param.shape[0]   == self.n_p);

        # Fetch the parameter values.
        A   : float             = param[self.A_idx];
        w   : float             = param[self.w_idx]; 

        # Make the t_grid. If we are not using uniform t spacing, then add a random perturbation to 
        # the intermediate time steps.
        n_t     : int           = self.config['Explicit']['n_t'];
        t_max   : float         = self.config['Explicit']['t_max']; # We solve from t = 0 to t = t_max. 
        t_Grid  : numpy.ndarray = numpy.linspace(0, t_max, n_t, dtype = numpy.float32);
        if(self.Uniform_t_Grid == False):
            r               : float = 0.2*(t_Grid[1] - t_Grid[0]);
            t_adjustments           = numpy.random.uniform(low = -r, high = r, size = (n_t - 2));
            t_Grid[1:-1]            = t_Grid[1:-1] + t_adjustments;

        # Make the t, x meshgrids.
        t_mesh, x_mesh          = numpy.meshgrid(t_Grid, self.X_Positions, indexing = 'ij');
        t_mesh                  = torch.tensor(t_mesh);         # shape (n_t, n_x)
        x_mesh                  = torch.tensor(x_mesh);         # shape (n_t, n_x)

        # We know that
        #   u(t, x) =  A [ sin(2x - t) + 0.2 cos( (10x + t)sin(w t) ) ] exp(-0.3*x^2)
        # Thus,
        #   v(t, x) = (d/dt)u(t, x)
        #           = -A [ cos(2x - t) + 0.2 sin( (10x + t)cos(w t) )[ cos(w t) - w(10x + t)sin(w t)] ] exp(-0.3*x^2)
        U   : torch.Tensor  = A*torch.multiply( torch.sin(2.*x_mesh - t_mesh) +                                                             #  A*[ sin(2x - t)
                                                0.2*torch.cos(torch.multiply(10*x_mesh + t_mesh, torch.cos(w*t_mesh))),                     #      0.2*cos( (10x + t)cos(w t) ) ]*
                                                torch.exp(-0.3*torch.multiply(x_mesh, x_mesh)));                                            # exp(-0.3*x^2)
        
        V   : torch.Tensor  = -A*torch.multiply(torch.cos(2.*x_mesh - t_mesh) +                                                             # -A*[ cos(2x - t) + 
                                                0.2*torch.multiply( torch.sin(torch.multiply(10*x_mesh + t_mesh, torch.cos(w*t_mesh))),     #      0.2*sin( (10x + t)cos(w t) )*
                                                                    torch.cos(w*t_mesh) - w*(10*x_mesh + t_mesh)*torch.sin(w*t_mesh)),      #      [ cos(w t) - w(10x + t)sin(w t)] ] *
                                                torch.exp(-0.3*torch.multiply(x_mesh, x_mesh)));                                            # exp(-0.3*x^2)

        # All done!
        return [U, V], torch.Tensor(t_Grid);
        