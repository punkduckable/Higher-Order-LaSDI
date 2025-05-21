# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

import  sys;
import  os;
Util_Path       : str   = os.path.abspath(os.path.join(os.path.dirname(__file__), "Utilities"));
sys.path.append(Util_Path);

import  logging;
from    typing      import  Callable;

import  numpy;

# Setup the logger
LOGGER : logging.Logger = logging.getLogger(__name__);



# -------------------------------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------------------------------

def get_1dspace_from_list(param_dict : dict) -> tuple[int, numpy.ndarray]:
    """
    This function generates the parameter range (set of possible parameter values) for a parameter 
    that uses the list type test space. That is, "test_space_type" should be a key for the 
    parameter dictionary and the corresponding value should be "list". The parameter dictionary 
    should also have a "list" key whose value is a list of the possible parameter values.

    We parse this list and turn it into a numpy.ndarray.

    
    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    param_dict : dict
        Defines the set of allowed values for a single parameter. We should fetch this from the 
        configuration yml file. It must have a "list" key whose corresponding value is a list of 
        floats. 


    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    n_values, paramRange
     
    n_values : int
      the length of paramRange (see below).
    
    paramRange : numpy.ndarray, shape = (n_values)
        a 1d numpy ndarray (whose ith value is the i'th element of param_dict["list"]).
    """

    # In this case, the parameter dictionary should have a "list" attribute which should list the 
    # parameter values we want to test. Fetch it (it's length is n_values) and use it to generate an
    # array of possible values.
    n_values    : int           = len(param_dict['list']);
    paramRange  : numpy.ndarray = numpy.array(param_dict['list']);

    # All done!
    return n_values, paramRange;



def create_uniform_1dspace(param_dict : dict) -> tuple[int, numpy.ndarray]:
    """
    This function generates the parameter range (set of possible parameter values) for a parameter 
    that uses the uniform type test space. That is, "test_space_type" should be a key for the 
    parameter dictionary and the corresponding value should be "uniform". The parameter dictionary 
    should also have the following keys:
        "min"
        "max"
        "sample_size"
        "log_scale"
    "min" and "max" specify the minimum and maximum value of the parameter, respectively. 
    "sample_size" specifies the number of parameter values we generate. Finally, log_scale, if 
    true, specifies if we should use a uniform or logarithmic spacing between samples of the 
    parameter.
    
    The values corresponding to "min" and "max" should be floats while the values corresponding to 
    "sample_size" and "log_scale" should be an int and a bool, respectively. 

    
    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    param_dict : dict 
        Defines the set of allowed values for a single parameter. We should fetch this from the 
        configuration yml file. It must have a "min", "max", "sample_size", and "log_scale" 
        keys (see above).


    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    n_values, paramRange

    n_values : int
        the length of paramRange (see below). Equivalently, n_values = param_dict["sample_size"]. 
    
    paramRange : numpy.ndarray, shape = (n_param)
        ith value is the i'th possible value of the parameter. paramRange[0] = param_dict["min"]
        and paramRange[-1] = param_dict["max"]).
    """

    # Fetch the number of samples and the min/max value for this parameter.
    n_values    : int   = param_dict['sample_size'];
    minval      : float = param_dict['min'];
    maxval      : float = param_dict['max'];

    # Generate the range of parameter values. Note that we have to generate a uniform grid in the 
    # log space, then exponentiate it to generate logarithmic spacing.
    if (param_dict['log_scale']):
        paramRange : numpy.ndarray  = numpy.exp(numpy.linspace(numpy.log(minval), numpy.log(maxval), n_values));
    else:
        paramRange : numpy.ndarray  = numpy.linspace(minval, maxval, n_values);
    
    # All done! 
    return n_values, paramRange;



# A macro that allows us to switch function we use to generate generate a parameter's range. 
getParam1DSpace : dict[str, Callable]    = {'list'       : get_1dspace_from_list,
                                            'uniform'    : create_uniform_1dspace};



# -------------------------------------------------------------------------------------------------
# ParameterSpace Class
# -------------------------------------------------------------------------------------------------

class ParameterSpace:
    # Initialize class variables
    n_p             : int                   = 0;    # The number of parameters.
    param_list      : list[dict]            = [];   # Length = n_p. I'th element holds the parameter dictionary for the i'th parameter
    param_names     : list[str]             = [];   # Length = n_p. I'th element holds the name of the i'th parameter.
    train_space     : numpy.ndarray         = None; # Shape = (n_train, n_p). i,j element is the j'th parameter value in the i'th combination of training parameters.
    test_space      : numpy.ndarray         = None; # Shape = (n_test, n_p). i,j element is the j'th parameter value in the i'th combination of testing parameters.
    n_init_train    : int                   = 0;    # The initial number of combinations of parameters in the training set.
    test_grid_sizes : list[int]             = [];   # Length = n_p. i'th element is the number of distinct values of the i'th parameter in the test instances.
    test_meshgrid   : tuple[numpy.ndarray]  = None; # Length = n_p. I'th element is an ndarray holding the meshgrid of values for the i'th parameter.



    def __init__(self, config : dict) -> None:
        """
        Initializes a ParameterSpace object using the settings passed in the conf dictionary (which 
        should have been read from a yaml file).

        
        
        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        config : dict
            houses the settings we want to use to run the code. This should have been read from a 
            yml file. We assume it contains the following keys. If one or more keys are tabbed 
            over relative to one key above them, then the one above is a dictionary and the ones 
            below should be keys within that dictionary.
                - parameter_space
                    - parameters (this should have at least one parameter defined!)
                - test_space
                    - type (should be "grid")

        
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Nothing!
        """

        # Make sure the configuration dictionary has a "parameter_space" setting. This should house 
        # information about which variables are present in the code, as well as how we want to test
        # the various possible parameter values. 
        assert('parameter_space' in config);

        # Load the parameter_space settings. Each parameters has a name, min and max, and 
        # information on how many instances we want. 
        self.param_list : list[dict]    = config['parameter_space']['parameters'];
        self.n_p        : int           = len(self.param_list);

        # Fetch the parameter names.
        self.param_names : list[str]    = [];
        for param in self.param_list:
            self.param_names += [param['name']];
        LOGGER.info("Initializing a ParameterSpace object with parameters %s" % (str(self.param_names)));

        # First, let's make a set of parameter combinations to test at.
        test_space_type : str = config['parameter_space']['test_space']['type']
        if (test_space_type == 'grid'):
            # Generate the set possible parameter combinations. See the docstring for 
            # "createTestGridSpace" for details.
            self.test_grid_sizes, self.test_meshgrid, self.test_space = self.createTestGridSpace();
        LOGGER.info("The testing set has %d parameter combinations" % (self.test_space.shape[0]));

        # Next, let's make the training set. This should be a 2^k x n_p matrix, where n_p is the 
        # number of parameters. The i,j entry of this matrix gives the value of the j'th parameter 
        # in the i'th combination of parameters in the training set.
        self.train_space    = self.createInitialTrainSpace();
        self.n_init_train   = self.train_space.shape[0];
        LOGGER.info("The training set has %d parameter combinations" % (self.n_init_train));

        # All done!
        return;
    


    def n_train(self) -> int:
        return self.train_space.shape[0];
    


    def n_test(self) -> int:
        return self.test_space.shape[0];



    def createInitialTrainSpace(self) -> numpy.ndarray:
        """
        Sets up a grid of parameter value combinations to train at. Note that we only use the min 
        and max value of each parameter when setting up this grid. You must run this AFTER running 
        the createTestGridSpace function, since we use the min/max value of each parameter in 
        the testing set to build the training set. 


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        Nothing!
        
            
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        hyper_mesh_grid : numpy.ndarray, shape = (2^n_p, n_p)
            here, n_p = the number of parameters = len(self.param_list)). The i'th column is of
            hyper_mesh_grid is a flattened i'th mesh_grid array we when we create a mesh grid using 
            the min and max value of each parameter as the argument. See "createHyperMeshGrid" for 
            details. 
        
            Specifically, we return exactly what "createHyperGridSpace" returns. See the doc-string 
            for that function for further details. 
        """

        # We need to know the min and max value for each parameter to set up the grid of possible 
        # parameter value combinations.
        paramRanges : list[numpy.ndarray] = [];

        for i in range(len(self.param_list)):
            # Fetch the current parameter dictionary.
            param : dict = self.param_list[i];

            # Fetch the min, max value of the current parameter. 
            minval  : float = param['min'];
            maxval  : float = param['max'];
            
            # Store these combinations in an array and add them to the list.
            paramRanges += [numpy.array([minval, maxval])];

        # Use the ranges to set up a grid of possible parameter value combinations.
        mesh_grids : tuple[numpy.ndarray]  = self.createHyperMeshGrid(paramRanges);

        # Now combine these grids into a 2d 
        return self.createHyperGridSpace(mesh_grids);
    


    def createTestGridSpace(self) -> tuple[list[int], tuple[numpy.ndarray], numpy.ndarray]:
        """
        This function sets up a grid of parameter value combinations to test at. This uses the 
        information in the self.param_list variable. It should be run BEFORE running 
        createInitialTrainSpace


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        None!


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        gridSizes, mesh_grids, hyper_mesh_grid
        
        gridSizes : list[int]
            i'th element specifies the number of distinct values of the i'th parameter we consider 
            (this is the length of the i'th element of "paramRanges" below).

        mesh_grids : tuple[numpy.ndarray], len = n_p
            i'th element is a a n_p-dimension array of shape (N0, ... , N{k - 1}), where Ni = 
            self.param_list[i].size whose i(0), ... , i(k - 1) element specifies the value of the 
            i'th parameter in the i(0), ... , i(k - 1)'th unique combination of parameter values.
            here, n_p = len(self.param_list).

        hyper_mesh_grid : numpy.ndarray, shape = (M, n_p)
            flattened version of mesh_grid returned by createHyperGridSpace. Here, 
            M = \prod_{i = 0}^{k - 1} self.param_list[i].size
        """

        # Set up arrays to hold the parameter values + number of parameter values for each 
        # parameter.
        paramRanges : numpy.ndarray = [];
        gridSizes   : list[int]     = [];

        # Cycle through the parameters        
        for i in range(len(self.param_list)):
            # Fetch the current parameter.
            param   : dict  = self.param_list[i];

            # Fetch the set of possible parameter values (paramRange) + the size of this set (n_values)
            n_values, paramRange  = getParam1DSpace[param['test_space_type']](param);

            # Determine the min, max value of this parameter.
            self.param_list[i]['min']   = numpy.min(paramRange);
            self.param_list[i]['max']   = numpy.max(paramRange);

            # Add n_values, ParamRange to their corresponding lists
            gridSizes      += [n_values];
            paramRanges    += [paramRange];

        # Now that we have the set of parameter values for each parameter, turn it into a grid.
        mesh_grids : tuple[numpy.ndarray] = self.createHyperMeshGrid(paramRanges);

        # All done. Return a list specifying the number of possible values for each parameter, the
        # grids of parameter values, and the flattened/2d version of it. 
        return gridSizes, mesh_grids, self.createHyperGridSpace(mesh_grids);
        


    def createHyperMeshGrid(self, param_ranges : list[numpy.ndarray]) -> tuple[numpy.ndarray]:
        """
        This function generates arrays of parameter values. Specifically, if there are k 
        parameters (param_ranges has k elements), then we return k k-d arrays, the i'th one of 
        which is a k-dimensional array whose i(0), ... , i(k - 1) element specifies the value of 
        the i'th variable in the i(0), ... , i(k - 1)'th unique combination of parameter values.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        param_ranges : list[numpy.ndarray], len = n_p
            i'th element is a 2-element numpy.ndarray object housing the max and min-min value 
            for the i'th parameter value. Here, n_p is the number of parameters. 
                        

        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        paramSpaces : tuple[numpy.ndarray], len = n_p
            i'th element is a grid of parameter values for the i'th parameter. Specifically, if 
            there are n_p parameters and if param_range[i].size = Ni, then the i'th return array 
            has shape (N0, ... , N{k - 1}) and the j(0), ... , j(k - 1) element of this array 
            houses the j(i)'th value of the i'th parameter.

            Thus, if there are n_p parameters, the returned tuple has n_p elements, each one of 
            which is an ndarray of shape (N0, ... , N{k - 1}).
        """

        # Fetch the ranges, add them to a tuple (this is what the meshgrid function needs).
        args : tuple[numpy.ndarray] = ();
        for paramRange in param_ranges:
            args += (paramRange,);

        # Use numpy's meshgrid function to generate the grids of parameter values. This produces
        # a set of n_p arrays, each of shape (N(1), N(2), ... N(n_p)), where N(k) = 
        # len(param_ranges[k]) denotes the number of parameter values for the k'th parameter. 
        # The i(1), ... , i(n_p) element of the k'th array holds param_ranges[k][i(k)], the i(k)'th
        # value of the k'th parameter. 
        paramSpaces : tuple[numpy.ndarray] = numpy.meshgrid(*args, indexing = 'ij');

        # All done!
        return paramSpaces;
    


    def createHyperGridSpace(self, mesh_grids : tuple[numpy.ndarray]) -> numpy.ndarray:
        """
        Flattens the mesh_grid numpy.ndarray objects returned by createHyperMeshGrid and combines 
        them into a single 2d array of shape (grid size, number of parameters) (see below).


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        mesh_grids : tuple[numpy.ndarray], len = n_p
            i'th elenent is a numpy.ndarray of shape (N0, ... , N{n_p - 1}), where N0 is the number 
            of allowed values for the i'th parameter. mesh_grids should be the output of the 
            "CreateHyperMeshGrid" function. See that function's doc-string for details. 
    
        
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------
        
        param_grid : numpy.ndarray, shape = (M, n_p)
            i, j entry holds the value of the j'th parameter in the i'th combination of parameter 
            values. If mesh_grids has shape (N(0), ... , N(n_p - 1)), then M = N(0)*...*N(n_p - 1).
        """

        # For each parameter, we flatten its mesh_grid into a 1d array (of length (grid size)). We
        # horizontally stack these flattened grids to get the final param_grid array.
        param_grid : numpy.ndarray = None;
        for k, paramSpace in enumerate(mesh_grids):
            # Special treatment for the first parameter to initialize param_grid
            if (k == 0):
                param_grid : numpy.ndarray = paramSpace.reshape(-1, 1);
                continue;

            # Flatten the new mesh grid and append it to the grid.
            param_grid : numpy.ndarray = numpy.hstack((param_grid, paramSpace.reshape(-1, 1)));

        # All done!
        return param_grid;



    def appendTrainSpace(self, param_values : numpy.ndarray) -> None:
        """
        Adds a new parameter to self's train space attribute.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        param : numpy.ndarray, shape = self.n_p
            i'th element holds the value of a i'th parameter. We add this combination of parameter
            values to the training space.



        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Nothing!
        """

        # Make sure param has n_p components/can be appended to the set of training parameters.
        assert(self.train_space.shape[1] == param_values.size);

        # Add the new parameter to the training space by appending it as a new row to 
        # self.train_space
        self.train_space    : numpy.ndarray = numpy.vstack((self.train_space, param_values));
        
        # All done!
        return;
    


    def export(self) -> dict:
        """
        This function packages the testing/training examples into a dictionary, which it returns.

        
        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        None!

        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        dict_ : dict
            Below is a list of the keys with a short description of each corresponding value. 

            n_p : int 
                the number of parameters

            param_names : list[str], len = n_p
                i'th element specifies the name of the i'th parameter.
            
            param_list = self.param_list : list[dict]
                i'th element holds the dictionary used to define the i'th parameter.

            train_space = self.train_space : numpy.ndarray, shape = (n_train, n_p)
                i, j element holds the value of the j'th parameter in the i'th combination of 
                parameters in the training set.

            test_space = self.test_space : numpy.ndarray, shape = (n_test, n_p)
                i,j element holds the value of the j'th parameter in the i'th testing case.

            test_grid_sizes : list[int], len = n_p
                i'th element specifies how many distinct parameter values we use for the i'th 
                parameter. 

            test_meshgrid : tuple[numpy.ndarray], len = n_p
                i'th element is a n_p-dimensional array whose i(1), i(2), ... , i(n_p) element 
                holds the value of the i'th parameter in the i(1), ... , i(n_p) combination of 
                parameter values in the testing test. 

            n_init_train : int
                The number of combinations of training parameters in the training set.     
        """

        # Build the dictionary
        dict_ = {'n_p'              : self.n_p, 
                 'param_list'       : self.param_list, 
                 'param_names'      : self.param_names,
                 'train_space'      : self.train_space,
                 'test_space'       : self.test_space,
                 'test_grid_sizes'  : self.test_grid_sizes,
                 'test_meshgrid'    : self.test_meshgrid,
                 'n_init_train'     : self.n_init_train};
        
        # All done!
        return dict_;
    


    def load(self, dict_ : dict) -> None:
        """
        This function builds a parameter space object from a dictionary. This dictionary should 
        be one that was returned by the export method, or a loaded copy of a dictionary that was 
        returned by the export method. 


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        dict_ : dictionary
            This should be a dictionary with the following keys: 
                - n_p
                - param_list
                - param_names
                - train_space
                - test_space
                - test_grid_sizes
                - test_meshgrid
                - n_init_train
            This dictionary should have been returned by the export method. We use the values in this 
            dictionary to set up self.

        
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Nothing!
        """

        # Extract information from the dictionary.
        self.n_p                : int                   = dict_['n_p'];         
        self.param_list         : list                  = dict_['param_list'];      # length = n_p. I'th element holds dictionary for i'th parameter
        self.param_names        : list[str]             = dict_['param_names'];     # length = n_p. I'th element holds the name of the i'th parameter. 
        self.train_space        : numpy.ndarray         = dict_['train_space'];     #
        self.test_space         : numpy.ndarray         = dict_['test_space'];      #
        self.test_grid_sizes    : list[int]             = dict_['test_grid_sizes']; #
        self.test_meshgrid      : tuple[numpy.ndarray]  = dict_['test_meshgrid'];   #
        self.n_init_train       : int                   = dict_['n_init_train'];

        # Run checks
        assert(self.n_init_train            == dict_['n_init_train']);
        assert(self.train_space.shape[1]    == self.n_p);
        assert(self.test_space.shape[1]     == self.n_p);

        # All done!
        return;
