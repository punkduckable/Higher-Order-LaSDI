# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

# Add the main directory to the search path.
import  os;
import  sys;
PyMFEM_Path     : str   = os.path.join(os.path.curdir, "PyMFEM");
src_Path        : str   = os.path.dirname(os.path.dirname(__file__));
LD_Path         : str   = os.path.join(src_Path, "LatentDynamics");
util_Path       : str   = os.path.join(src_Path, "Utilities");
sys.path.append([PyMFEM_Path, src_Path, LD_Path, util_Path]);

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
        assert(len(param_names) == 1);
        assert('s' in param_names);

        # Call the super class initializer.
        super().__init__(config, param_names);

        # The functions we deal with are scalar valued. Likewise, since there are 2 spatial 
        # dimensions, dim is 2. 
        self.qdim           : int   = 1;
        self.spatial_dim    : int   = 2;

        # Make sure the config dictionary is actually for the explicit physics model.
        assert('NonlinearElasticity' in config);
        
        return;
    


    def initial_condition(self, param : numpy.ndarray) -> list[numpy.ndarray]:
        """
        TO DO 
        """

        return;
    


    def solve(self, param : numpy.ndarray) -> tuple[list[torch.Tensor], numpy.ndarray]:
        """
        TO DO 
        """

        return;
    

    
    def residual(self, X_hist : list[numpy.ndarray]) -> tuple[numpy.ndarray, float]:
        """
        TO DO
        """

        return;
