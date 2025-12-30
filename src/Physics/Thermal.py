# -------------------------------------------------------------------------------------------------
# Inputers and setup
# -------------------------------------------------------------------------------------------------

import  logging;
import  os;
import  sys;

import  numpy;
import  torch;

from    Physics                         import  Physics;

# Setup the logger
LOGGER : logging.Logger = logging.getLogger(__name__);



# -------------------------------------------------------------------------------------------------
# Thermal class
# -------------------------------------------------------------------------------------------------

class Thermal(Physics):
    """
    The Thermal class is a physics subclass that is designed to in the thermal history from a batch
    of ALE3D simulations. These simulations should be of directed energy deposition (DED) additive 
    manufacturing of a powder bed. The simulation results shoul be stored in a single hdf5 file. 

    We assume there are two parameters: the scan speed (of the laser) and the beam power. We 
    assume the user ran simulations over a 2d grid of these parameters. 

    For each simulation, we load the thermal history, and parameter values.

    Notably, because we load from a file, we only have IC's (and data) for parameter values in 
    the grid of parameter values. Thus, the IC function will protest if the user specifies 
    parameters outside of the grid.
    """
    def __init__(self, config : dict) -> None:
        """
        Initialize a Thermal object.

        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        config : dict
            A dictionary containing the configuration for the Thermal object. Specifically, it 
            should have a "hdf5_file" key which specifies the name of a file in the "Data" 
            directory.

        """
        super().__init__(config, param_names);

    def initial_condition(self, param : numpy.ndarray) -> list[numpy.ndarray]:
        """
        """
    
    def solve(self, param : numpy.ndarray) -> tuple[list[torch.Tensor], torch.Tensor]:
        """
        Fetches the thermal history for the given parameter values. Note that these 
        values must be within the grid of parameter values used to run the simulations.
        """