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
        around the following function of t and X = (x, y):
            
                  u(t, X)   =  A [ sin(2x - t)cos(2y - t) + 0.2 cos( (10(x + y) + t)cos(w t) ) ] exp(-0.3*(x^2 + y^2))

        
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
        assert isinstance(param_names, list),   "type(param_names) = %s" % str(type(param_names));
        assert len(param_names) == 2,           "len(param_names) = %d" % len(param_names);
        assert 'A' in param_names,              "param_names = %s" % str(param_names);
        assert 'w' in param_names,              "param_names = %s" % str(param_names);

        # Make sure the config dictionary is actually for the Explicit physics model.
        assert('Explicit' in config);

        # Set up spatial variables
        self.n_positions            : int           = config['Explicit']['n_positions'];    
        self.x_min                  : float         = config['Explicit']['x_min'];
        self.x_max                  : float         = config['Explicit']['x_max'];
        self.y_min                  : float         = config['Explicit']['y_min'];
        self.y_max                  : float         = config['Explicit']['y_max'];

        # Set up the spatial grid.
        x_coords                    : numpy.ndarray = numpy.random.uniform(low = self.x_min, high = self.x_max, size = self.n_positions).astype(numpy.float32);
        y_coords                    : numpy.ndarray = numpy.random.uniform(low = self.y_min, high = self.y_max, size = self.n_positions).astype(numpy.float32);
        X_Positions                 : numpy.ndarray = numpy.row_stack((x_coords, y_coords));     # shape = (2, n_positions)

        # Call the super class initializer.
        super().__init__(config         = config, 
                         spatial_dim    = 2,
                         X_Positions    = X_Positions,
                         Frame_Shape    = [1, self.n_positions],
                         param_names    = param_names, 
                         Uniform_t_Grid = config['Explicit']['uniform_t_grid'],
                         n_IC           = 1);
     
        # Determine which index corresponds to 'a' and 'w' (we pass an array of parameter values, 
        # we need this information to figure out which element corresponds to which variable).
        self.A_idx = self.param_names.index('A');
        self.w_idx = self.param_names.index('w');
        
        # All done!
        return;
    


    def initial_condition(self, param : numpy.ndarray) -> list[numpy.ndarray]:
        """
        Evaluates the initial condition at the points in self.X_Positions. In this case,
        
            u(t, x) =  A [ sin(2x - t)cos(2y - t) + 0.2 cos( (10(x + y) + t)cos(w t) ) ] exp(-0.3*(x^2 + y^2))
  
        Which means that
        
            u(0, x) =  A [sin(2x)cos(2y) + 0.2*cos(10(x + y))]exp(-0.3*(x^2 + y^2))


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
            i'th element has shape (1, self.n_positions) (the number of grid points along the 
            spatial axis) and holds the i'th derivative of the initial state when we use param to 
            define the FOM.
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
        X           : numpy.ndarray = self.X_Positions;
        x_coords    : torch.Tensor  = torch.tensor(X[0, :], dtype = torch.float32); 
        y_coords    : torch.Tensor  = torch.tensor(X[1, :], dtype = torch.float32);
        u0          : numpy.ndarray =  A*numpy.multiply(numpy.sin(2*x_coords)*numpy.cos(2*y_coords) + 0.2*numpy.cos(10*(x_coords + y_coords)), numpy.exp(-0.3*(x_coords*x_coords + y_coords*y_coords)));
        return [u0.reshape(1, -1)];
    


    def solve(self, param : numpy.ndarray) -> tuple[list[torch.Tensor], torch.Tensor]:
        """
        Evaluates u(t, X) (see __init__ docstring) on the t, x grids using the 
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

        U : list[torch.Tensor], len = 1
            A list whose lone elment is a torch.Tensor object of shape (n_t, self.n_positions), where 
            n_t is the number of time steps when we solve the FOM using param. The i, j entry of this
            array holds u(t[i], X[:, j]). 

        t_Grid : torch.Tensor, shape = (n_t)
            i'th element holds the i'th time value at which we have an approximation to the FOM 
            solution (the time value associated with U[i, :]).
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

        # We know that
        #   u(t, X) =  A [ sin(2x - t)cos(2y - t) + 0.2 cos( (10(x + y) + t)cos(w t) ) ] exp(-0.3*(x^2 + y^2))
        U           : torch.Tensor = torch.empty((n_t, 1, self.n_positions), dtype = torch.float32);
        x_coords    : torch.Tensor = torch.tensor(self.X_Positions[0, :]);
        y_coords    : torch.Tensor = torch.tensor(self.X_Positions[1, :]);
        for i in range(n_t):
            t_i : float = t_Grid[i]*torch.ones(self.n_positions, dtype = torch.float32);

            U[i, 0, :] = A*torch.multiply(  torch.sin(2.*x_coords - t_i) * torch.cos(2.*y_coords - t_i) +                               #  A*[ sin(2x - t)cos(2y - t)
                                            0.2*torch.cos(torch.multiply(10*(x_coords + y_coords) + t_i, torch.cos(w*t_i))),            #      0.2*cos( (10(x + y) + t)cos(w t) ) ]*
                                            torch.exp(-0.3*(torch.multiply(x_coords, x_coords) + torch.multiply(y_coords, y_coords)))); # exp(-0.3*(x^2 + y^2))

        # All done!
        return [U], torch.Tensor(t_Grid);
        