# -------------------------------------------------------------------------------------------------
# Imports and setup
# -------------------------------------------------------------------------------------------------

import  logging;
import  os;
import  glob;

import  numpy;
import  torch;
import  h5py;

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
    manufacturing of a powder bed. The simulation results should be stored in a single hdf5 file. 
        
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
            This should be the "Physics" sub-dictionary of the main configuration file. It should 
            have a "hdf5_dir" key which specifies the path to the directory housing a metadata
            file and a 2D grid of simulation results (each stored in a separate hdf5 file).
        """

        # First, let's fetch the hdf5 directory.
        self.hdf5_dir : str = config['Thermal']['hdf5_dir'];

        # Set things we know.
        n_IC            : int   = 1;        # The heat equation has one time derivative.
        Uniform_t_Grid  : bool  = False;    # ALE3D uses adaptive time stepping.
        spatial_dim     : int   = 3;

        # To get the frame shape and X_Positions, we need to load one of the simulation results. To
        # do this, we can just load any file in the hdf5 directory that ends with .h5.
        h5_files : list[str] = glob.glob(os.path.join(self.hdf5_dir, "*.h5"));
        with h5py.File(h5_files[0], "r") as f:
            # Fetch the nodet dataset and its shape.
            nodet_ds = f.get("nodet");
            if nodet_ds is None:
                raise RuntimeError("Nodet dataset not found in file %s" % h5_files[0]);
            nodet_shape = nodet_ds.shape;



            print(type(nodet_shape))
            print(nodet_shape)




            # the shape should be (n_time_steps, n_nodes). We want n_nodes.
            assert(len(nodet_shape) == 2);
            assert(nodet_shape[1] > 0);
            frame_shape : list[int] = [nodet_shape[1]];

            # Now we can fetch the node coordinates, which should have shape (n_nodes, 3).
            nodes_coords_ds     = f.get("nodes_coords");
            if nodes_coords_ds is None:
                raise RuntimeError("Nodes coordinates dataset not found in file %s" % h5_files[0]);
            nodes_coords_shape  = nodes_coords_ds.shape;
            assert(len(nodes_coords_shape) == 2);
            assert(nodes_coords_shape[0] == frame_shape[0]);
            assert(nodes_coords_shape[1] == 3);

            # Convert to numpy array.
            X_Positions : numpy.ndarray = nodes_coords_ds.value;



            print(type(X_Positions))
            print(X_Positions.shape)
            print(X_Positions[:10]);

        # We are now ready to initialize the super class.
        super().__init__(   spatial_dim     = spatial_dim,
                            Frame_Shape     = frame_shape,
                            X_Positions     = X_Positions,
                            config          = config, 
                            param_names     = ['laser_power', 'scan_speed'], 
                            Uniform_t_Grid  = Uniform_t_Grid,
                            n_IC            = n_IC);
        
        # To make the solve method work, we need to fetch the set of parameter values. 
        # We can do this by extracting the "laser powers" and "scan speed" lists from the 
        # metadata header. 
        with open(os.path.join(self.hdf5_dir, "metadata.txt"), 'r') as metadata_file:
            # The first line of the metadata_file should be "PARAMETERS". The second should be
            # something like "laser powers: <p1>, <p2>, ..." where <p1>, <p2>, ... are the power
            # values. Let's extract these and store them in a list. The third line should be 
            # something like "scan speeds: <s1>, <s2>, ..." where <s1>, <s2>, ... are the scan 
            # speed values. Let's extract these and store them in a list.
            _                                   = metadata_file.readline();                             # skip the first line
            
            laser_powers_line   : str           = metadata_file.readline();                             # read the line housing the laser powers.
            laser_powers        : list[str]     = laser_powers_line.split(":")[1].strip().split(",");   # discard the label (split(":")), then parse the rest as a comma separated list
            self.laser_powers   : list[float]   = [float(speed) for speed in laser_powers];             # convert to floats, store as an attribute.
            
            scan_speeds_line    : str           = metadata_file.readline();                             # do the same for scan speeds 
            scan_speeds         : list[str]     = scan_speeds_line.split(":")[1].strip().split(",");   
            self.scan_speeds    : list[float]   = [float(speed) for speed in scan_speeds];

        """
        Next, we need to fetch and store the initial conditions and file names. We will do this by 
        determining the name of the file storing the data for each parameter combination. We 
        will store these names, then store the first frame (IC) from each file's nodet dataset 
        in a (n_laser_powers, n_scan_speeds, n_nodes) array whose i, j, k element holds the initial
        temperature at the k'th node for the i'th laser power and j'th scan speed. 
        
        To do this, we  need to know which file corresponds to each parameter value. We can do 
        this by reading the metadata file, whcih consist sof a header and a body. The body begins 
        after a line of all `=` characters. It begins with a blank line, then has a line describing 
        the contents of each line in the body. After this, each line of the body consists of a 
        comma separated list of the form:
            <p>, <s>, <file name>
        Where <p> and <s> specify the value of the laser power and scan speed in the simulation
        that generated the data stored in <file name>. Thus, we can run line-by-line, store the 
        file name, then open the corresponding files, then store the first entry of that file's 
        nodet dataset in the corresponding entry of the initial conditions array. 
        """
        n_powers : int = len(self.laser_powers);
        n_speeds : int = len(self.scan_speeds);
        n_nodes  : int = self.X_Positions.shape[0];
        self.IC_array   : numpy.ndarray = numpy.empty((n_powers, n_speeds, n_nodes), dtype = numpy.float32);
        self.file_names : numpy.ndarray = numpy.empty((n_powers, n_speeds), dtype = object)

        with open(os.path.join(self.hdf5_dir, "metadata.txt"), 'r') as metadata_file:
            # Cycle through the header. 
            while(True):
                line : str = metadata_file.readline();
                if(line[0] == '='):
                    break;
                else:
                    continue;
                    
            # After the header, we need to parse two lines, then we can begin reading files.
            _ = metadata_file.readline();       # blank line
            _ = metadata_file.readline();       # line describing table entries.
            lines : list[str] = metadata_file.readlines();
            n_lines : int = len(lines);

            # Cycle through the lines.
            for i in range(n_lines):
                line : str = lines[i].strip();
                parts : list[str] = line.split(",");   

                # Extract the power,  scan speed, and file name.
                power       : float = float(parts[0]);
                speed       : float = float(parts[1]);    
                file_name   : str = parts[2].strip();
            
                # Determine which power and scan speed this corresponds to.
                power_idx : int = self.laser_powers.index(power);
                speed_idx : int = self.scan_speeds.index(speed);
                
                # store the file name.
                self.file_names[power_idx, speed_idx] = file_name;

                # Open the file and store the first entry of the nodet dataset.
                with h5py.File(os.path.join(self.hdf5_dir, file_name), 'r') as f:
                    nodet_ds = f.get("nodet");
                    if nodet_ds is None:
                        raise RuntimeError("Nodet dataset not found in file %s" % file_name);
                     
                    # Make sure nodet_ds has shape (n_time_steps, n_nodes).
                    nodet_shape = nodet_ds.shape;
                    assert(len(nodet_shape) == 2);
                    assert(nodet_shape[1] == self.X_Positions.shape[0]);

                    # Store the first entry of the nodet dataset.
                    self.IC_array[power_idx, speed_idx, :] = nodet_ds[0, :];
            
        # Everything is now set up!
        return;
                


    def initial_condition(self, param : numpy.ndarray) -> list[numpy.ndarray]:
        """
        Evaluates the initial condition at the points in self.X_Positions. In this case, the initial 
        condition is the temperature at the nodes of the mesh.



        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        param : numpy.ndarray, shape = (self.n_p)
            A two element array holding the values of the laser power and scan speed, respectively.
            Note that these values must be in self.laser_powers and self.scan_speeds, respectively.

        

        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        A single element list whose lone element holds a numpy array of shape (n_nodes) holding 
        the initial condition corresponding to the requested laser power and scan speed.
        """

        assert isinstance(param, numpy.ndarray), "type(param) = %s" % str(type(param));
        assert param.size == 2, "param shape = %s" % str(param.shape);
        
        # Make sure the laser powers and scan speeds are in self.laser_powers and self.scan_speeds,
        # respectively.
        requested_laser_power   : float = param[0];
        requested_scan_speed    : float = param[1];
        assert requested_laser_power in self.laser_powers, "requested laser power = %f, self.laser_powers = %s" % (requested_laser_power, str(self.laser_powers))
        assert requested_scan_speed in self.scan_speeds, "requested scan speed = %f, self.scan_speeds = %s" % (requested_scan_speed, str(self.scan_speeds)) 

        # If so, fetch the corresponding initial condition.
        power_index : int = self.laser_powers.index(requested_laser_power);
        speed_index : int = self.scan_speeds.index(requested_scan_speed);
        return [self.IC_array[power_index, speed_index, :]];


    
    def solve(self, param : numpy.ndarray) -> tuple[list[torch.Tensor], torch.Tensor]:
        """
        Fetches the thermal history for the given parameter values. Note that these 
        values must be within the grid of parameter values used to run the simulations.

        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        param : numpy.ndarray, shape = (self.n_p)
            A two element array holding the values of the laser power and scan speed, respectively.
            Note that these values must be in self.laser_powers and self.scan_speeds, respectively.

        

        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        X, t_Grid.
         
        X : list[torch.Tensor], len = 1
            A single element list whose lone element is a numpy.ndarray of shape (n_t, n_nodes) 
            whose i, j entry holds the value of the nodal temperature at the i'th time step 
            at the j'th node.

        t_Grid : torch.Tensor, shape = (n_t)
            i'th element holds the i'th time value at which we have an approximation to the FOM 
            solution (the time value associated with X[0][i, ...]).
        """

        assert isinstance(param, numpy.ndarray), "type(param) = %s" % str(type(param));
        assert param.size == 2, "param shape = %s" % str(param.shape);
        
        # Make sure the laser powers and scan speeds are in self.laser_powers and self.scan_speeds,
        # respectively.
        requested_laser_power   : float = param[0];
        requested_scan_speed    : float = param[1];
        assert requested_laser_power in self.laser_powers, "requested laser power = %f, self.laser_powers = %s" % (requested_laser_power, str(self.laser_powers))
        assert requested_scan_speed in self.scan_speeds, "requested scan speed = %f, self.scan_speeds = %s" % (requested_scan_speed, str(self.scan_speeds)) 

        # If so, fetch the corresponding initial condition.
        power_index : int = self.laser_powers.index(requested_laser_power);
        speed_index : int = self.scan_speeds.index(requested_scan_speed);

        # Fetch the file name
        requested_file : str = self.file_names[power_index, speed_index];
        with open(os.path.join(self.hdf5_dir, requested_file), 'r') as f:
            # Fetch the nodet dataset and its shape, make sure it has n_nodes nodes.
            nodet_ds = f.get("nodet");
            if nodet_ds is None:
                raise RuntimeError("Nodet dataset not found in file %s" % requested_file);
            nodet_shape = nodet_ds.shape;       # should be (n_time_steps, n_nodes)
            assert(len(nodet_shape) == 2);
            assert(nodet_shape[1] == self.X_Positions.shape[0]);

            # Fetch the time values. 
            time_ds = f.get("time");
            if time_ds is None:
                raise RuntimeError("Time dataset not found in file %s" % requested_file);
            time_shape = time_ds.shape;       # should be (n_time_steps,)
            assert(len(time_shape) == 1);
            assert(time_shape[0] == nodet_shape[0]);
            time_values : numpy.ndarray = time_ds.value;

            # Convert the nodet dataset to a torch.Tensor.
            n_time_steps : int          = nodet_shape[0];
            n_nodes      : int          = nodet_shape[1];
            X            : torch.Tensor = torch.Tensor(nodet_ds);   # shape = (n_time_steps, n_nodes)
            t_Grid       : torch.Tensor = torch.Tensor(time_values);
        
        # All done!
        return [X], t_Grid;
