# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

import  sys;
import  os;
Physics_Path    : str   = os.path.abspath(os.path.join(os.path.dirname(__file__), "Physics"));
LD_Path         : str   = os.path.abspath(os.path.join(os.path.dirname(__file__), "LatentDynamics"));
Model_Path      : str   = os.path.abspath(os.path.join(os.path.dirname(__file__), "Models"));
Utilities_Path  : str   = os.path.abspath(os.path.join(os.path.dirname(__file__), "Utilities"));
sys.path.append(Physics_Path);
sys.path.append(LD_Path);
sys.path.append(Model_Path);
sys.path.append(Utilities_Path);

import  torch;
import  numpy;
from    sklearn.gaussian_process    import  GaussianProcessRegressor;

from    GaussianProcess             import  eval_gp, sample_coefs, fit_gps;
from    Physics                     import  Physics;
from    LatentDynamics              import  LatentDynamics;
from    Autoencoder                 import  Autoencoder;
from    Autoencoder_Pair            import  Autoencoder_Pair;
from    CNN_3D_Autoencoder          import  CNN_3D_Autoencoder;
from    ParameterSpace              import  ParameterSpace;
from    Trainer                     import  Trainer;

import  logging;
LOGGER : logging.Logger = logging.getLogger(__name__);



# -------------------------------------------------------------------------------------------------
# Simulate latent dynamics
# -------------------------------------------------------------------------------------------------

def average_rom(model           : torch.nn.Module, 
                physics         : Physics, 
                latent_dynamics : LatentDynamics, 
                gp_list         : list[GaussianProcessRegressor], 
                param_grid      : numpy.ndarray,
                t_Grid          : list[numpy.ndarray | torch.Tensor],
                trainer         : Trainer) -> list[list[numpy.ndarray]]:
    """
    This function simulates the latent dynamics for a set of parameter values by using the mean of
    the posterior distribution for each coefficient's posterior distribution. Specifically, for 
    each parameter combination, we determine the mean of the posterior distribution for each 
    coefficient. We then use this mean to simulate the latent dynamics forward in time (starting 
    from the latent encoding of the FOM initial condition for that combination of coefficients).

    
    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    model : torch.nn.Module
        The actual model object that we use to map the ICs into the latent space. physics, 
        latent_dynamics, and model should have the same number of initial conditions.

    physics : Physics
        Allows us to get the latent IC solution for each combination of parameter values. physics, 
        latent_dynamics, and model should have the same number of initial conditions.
    
    latent_dynamics : LatentDynamics
        describes how we specify the dynamics in the model's latent space. We assume that 
        physics, latent_dynamics, and model all have the same number of initial conditions.

    gp_list : list[], len = n_coef
        An n_coef element list of trained GP regressor objects. The i'th element of this list is 
        a GP regressor object that predicts the i'th coefficient. 

    param_grid : numpy.ndarray, shape = (n_param, n_p)
        i,j element holds the value of the j'th parameter in the i'th combination of parameter 
        values. Here, n_p is the number of parameters and n_param is the number of combinations
        of parameter values.

    t_Grid : list[torch.Tensor], len = n_param
        i'th element is a 2d numpy.ndarray or torch.Tensor object of shape (n_t(i)) or (1, n_t(i)) 
        whose k'th or (0, k)'th entry specifies the k'th time value we want to find the latent 
        states when we use the j'th initial conditions and the i'th set of coefficients.

    trainer : Trainer
        The trainer object. We use this to get normalization stats if they are enabled.
        If normalization is enabled, we use the stats to normalize initial conditions and
        de-normalize predictions for reporting/plots.
        before encoding them.
    
    
    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------
    
    Zis : list[list[numpy.ndarray]], len = n_param
        i'th element is an n_IC element list whoe j'th element is a 2d numpy.ndarray object of 
        shape (n_t_i, n_z) whose p, q element holds the q'th component of the j'th derivative of 
        the latent solution at the p'th time step when we the means of the posterior distribution 
        for the i'th combination of parameter values to define the latent dynamics.
    """

    # Checks. 
    assert isinstance(param_grid, numpy.ndarray),   "type(param_grid) = %s, expected numpy.ndarray" % (type(param_grid) == numpy.ndarray);
    assert param_grid.ndim    == 2,                 "param_grid.ndim = %d, expected 2" % (param_grid.ndim);
    n_param : int   = param_grid.shape[0];
    n_p     : int   = param_grid.shape[1];

    assert isinstance(gp_list, list),               "type(gp_list) = %s, expected list" % (type(gp_list) == list);
    assert isinstance(t_Grid, list),                "type(t_Grid) = %s, expected list" % (type(t_Grid) == list);
    assert len(t_Grid)  == n_param,                 "len(t_Grid) = %d, n_param %d" % (len(t_Grid), n_param);

    n_IC    : int   = latent_dynamics.n_IC;
    n_z     : int   = latent_dynamics.n_z;
    assert model.n_IC       == n_IC,                "model.n_IC = %d, n_IC %d" % (model.n_IC, n_IC);
    assert physics.n_IC     == n_IC,                "physics.n_IC = %d, n_IC %d" % (physics.n_IC, n_IC);


    # For each parameter in param_grid, fetch the corresponding initial condition and then encode
    # it. This gives us a list whose i'th element holds the encoding of the i'th initial condition.
    LOGGER.debug("Fetching latent space initial conditions for %d combinations of parameters." % n_param);
    Z0      : list[list[numpy.ndarray]] = model.latent_initial_conditions(param_grid, physics, trainer = trainer);

    # Evaluate each GP at each combination of parameter values. This returns two arrays, the 
    # first of which is a 2d array of shape (n_param, n_coef) whose i,j element specifies the mean 
    # of the posterior distribution for the j'th coefficient at the i'th combination of parameter 
    # values.
    LOGGER.debug("Finding the mean of each GP's posterior distribution");
    post_mean, _ = eval_gp(gp_list, param_grid);

    # Make each element of t_Grid into a numpy.ndarray of shape (1, n_t(i)). This is what 
    # simulate expects.
    t_Grid_np : list[numpy.ndarray] = [];
    for i in range(n_param):
        if(isinstance(t_Grid[i], torch.Tensor)):
            t_Grid_np.append(t_Grid[i].detach().numpy());
        else:
            t_Grid_np.append(t_Grid[i]);
        t_Grid_np[i] = t_Grid_np[i].reshape(1, -1);

    # Simulate the laten dynamics! For each testing parameter, use the mean value of each posterior 
    # distribution to define the coefficients. 
    LOGGER.info("simulating initial conditions for %d combinations of parameters forward in time" % n_param);
    Zis : list[list[numpy.ndarray]] = latent_dynamics.simulate( coefs   = post_mean, 
                                                                IC      = Z0, 
                                                                t_Grid  = t_Grid,
                                                                params  = param_grid);
    
    # At this point, Zis[i][j] has shape (n_t_i, 1, n_z). We remove the extra dimension.
    for i in range(n_param):
        n_t_i   : int   = t_Grid_np[i].shape[1];
        for j in range(n_IC):
            Zis[i][j] = Zis[i][j].reshape(n_t_i, n_z);
    
    # All done!
    return Zis;



def sample_roms(model           : torch.nn.Module, 
                physics         : Physics, 
                latent_dynamics : LatentDynamics, 
                gp_list         : list[GaussianProcessRegressor], 
                param_grid      : numpy.ndarray, 
                t_Grid          : list[numpy.ndarray | torch.Tensor],
                n_samples       : int,
                trainer         : Trainer) ->           list[list[numpy.ndarray]]:
    """
    This function samples the latent coefficients, solves the corresponding latent dynamics, and 
    then returns the resulting latent solutions. 

    
    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    model : torch.nn.Module
        A model (i.e., autoencoder). We use this to map the FOM IC's (which we can get from 
        physics) to the latent space using the model's encoder. physics, latent_dynamics, and 
        model should have the same number of initial conditions.

    physics : Physics
        allows us to find the IC for a particular combination of parameter values. physics, 
        latent_dynamics, and model should have the same number of initial conditions.
    
    latent_dynamics : LatentDynamics
        describes how we specify the dynamics in the model's latent space. We use this to simulate 
        the latent dynamics forward in time. physics, latent_dynamics, and model should have the
        same number of initial conditions.

    gp_list : list[GaussianProcessRegressor], len = n_coef
        i'th element is a trained GP regressor object that predicts the i'th coefficient. 

    param_grid : numpy.ndarray, shape = (n_param, n_p)
        i,j element of holds the value of the j'th parameter in the i'th combination of parameter 
        values. Here, n_p is the number of parameters and n_param is the number of combinations 
        of parameter values. 

    n_samples : int
        The number of samples we want to draw from each posterior distribution for each coefficient
        evaluated at each combination of parameter values.

    t_Grid : list[numpy.ndarray] or list[torch.Tensor], len = n_param
        i'th entry is an numpy.ndarray or torch.Tensor of shape (n_t(i)) or (1, n_t(i)) whose k'th 
        element specifies the k'th time value we want to find the latent states when we use the 
        j'th initial conditions and the i'th set of coefficients.    

    
    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------
    
    LatentStates : list[list[numpy.ndarray]], len = n_param
        i'th element is an n_IC element list whose j'th element is a 3d numpy ndarray of shape 
        (n_t(i), n_samples, n_z) whose p, q, r element holds the r'th component of the j'th 
        derivative of the q,i latent solution at t_Grid[i][p]. The q,i latent solution is the 
        solution the latent dynamics when the coefficients are the q'th sample of the posterior 
        distribution for the i'th combination of parameter values (which are stored in 
        param_grid[i, :]).
    """
    
    # Checks
    assert isinstance(gp_list, list), "type(gp_list) = %s, expected list" % (type(gp_list) == list);
    assert isinstance(t_Grid, list), "type(t_Grid) = %s, expected list" % (type(t_Grid) == list);
    assert isinstance(n_samples, int), "type(n_samples) = %s, expected int" % (type(n_samples) == int);

    assert isinstance(param_grid, numpy.ndarray), "type(param_grid) = %s, expected numpy.ndarray" % (type(param_grid));
    assert len(param_grid.shape)    == 2, "len(param_grid.shape) = %d, expected 2" % (len(param_grid.shape));
    n_param     : int               = param_grid.shape[0];
    n_p         : int               = param_grid.shape[1];

    assert len(t_Grid)              == n_param, "len(t_Grid) = %d, n_param %d" % (len(t_Grid), n_param);
    for i in range(n_param):
        assert isinstance(t_Grid[i], numpy.ndarray) or isinstance(t_Grid[i], torch.Tensor), "type(t_Grid[%d]) = %s, expected numpy.ndarray or torch.Tensor" % (i, type(t_Grid[i]));

    n_coef      : int               = len(gp_list);
    n_IC        : int               = latent_dynamics.n_IC;
    n_z         : int               = model.n_z;
    assert physics.n_IC             == n_IC, "physics.n_IC = %d, n_IC %d" % (physics.n_IC, n_IC);
    assert model.n_IC               == n_IC, "model.n_IC = %d, n_IC %d" % (model.n_IC, n_IC);


    # Reshape t_Grid so that the i'th element is a numpy.ndarray of shape (1, n_t(i)). This is what 
    # simulate expects.
    LOGGER.debug("reshaping t_Grid so that the i'th element has shape (1, n_t(i)).");
    t_Grid_np : list[numpy.ndarray] = [];
    for i in range(n_param):
        if(isinstance(t_Grid[i], torch.Tensor)):
            t_Grid_np.append(t_Grid[i].detach().numpy());
        else:
            t_Grid_np.append(t_Grid[i]);
        
        t_Grid_np[i] = t_Grid_np[i].reshape(1, -1);
    
    # For each combination of parameter values in param_grid, fetch the corresponding initial 
    # condition and then encode it. This gives us a list whose i'th element is an n_IC element
    # list whose j'th element is an array of shape (1, n_z) holding the IC for the j'th derivative
    # of the latent state when we use the i'th combination of parameter values. 
    LOGGER.debug("Fetching latent space initial conditions for %d combinations of parameters." % n_param);
    Z0      : list[list[numpy.ndarray]] = model.latent_initial_conditions(param_grid, physics, trainer = trainer);


    # Setup a list to hold the simulated dynamics. There are n_param parameters. For each 
    # combination of parameter values, we have n_IC initial conditions. For each IC, we 
    # have n_samples simulations, each of which has n_t_i frames, each of which has n_z components
    # Thus, we need a n_param element list whose i'th element is a n_IC element list whose 
    # j'th element is a 3d array of shape (n_t_i, n_samples, n_z).
    LatentStates : list[list[numpy.ndarray]] = [];
    for i in range(n_param):
        LatentStates_i  : list[numpy.ndarray]   = [];
        n_t_i           : int                   = t_Grid_np[i].shape[1];

        for j in range(n_IC):
            LatentStates_i.append(numpy.empty((n_t_i, n_samples, n_z), dtype = numpy.float32));
        LatentStates.append(LatentStates_i);


    # Process each parameter combination independently. For each parameter, we sample n_samples
    # coefficient sets, simulate the dynamics, and keep only non-divergent samples. If any diverge,
    # we resample only those that diverged. This is more efficient than the old approach which
    # resampled all parameters if any single parameter diverged.
    #
    # NOTE: We check for divergent samples (latent states that grow unreasonably large) and 
    # resample if needed. This prevents a single divergent sample from breaking the STD calculation.
    LOGGER.info("Generating latent trajectories for %d samples across %d parameter combinations." % (n_samples, n_param));
    divergence_threshold  : float = 1e4;   # If any latent component exceeds this, the sample is divergent
    max_resample_attempts : int   = 100;   # Maximum times to resample a single sample before giving up
    
    for i in range(n_param):
        LOGGER.debug(f"Processing parameter combination {i+1}/{n_param}");
        
        # Track which samples are valid (non-divergent) for this parameter
        valid_samples       : list[int]                 = [];  # Indices of samples that are non-divergent
        sample_trajectories : list[list[numpy.ndarray]] = [];  # Store trajectories for valid samples
        
        total_resample_attempts : int = 0;
        
        # Keep sampling until we have n_samples valid (non-divergent) trajectories
        while len(valid_samples) < n_samples:
            n_needed = n_samples - len(valid_samples);
            
            # Check if we've exceeded total resampling budget
            if total_resample_attempts > max_resample_attempts:
                LOGGER.error(
                    f"Parameter {i}: Failed to generate {n_needed} non-divergent samples after "
                    f"{max_resample_attempts} total resampling attempts. Using divergent samples. "
                    f"This suggests:\n"
                    f"  - Model hasn't converged yet (train longer)\n"
                    f"  - GP variance is too high (increase alpha in GaussianProcess.py)\n"
                    f"  - Latent dynamics are fundamentally unstable at this parameter value"
                );

                # Generate the remaining samples even if they might diverge
                n_needed_coefs = sample_coefs(gp_list = gp_list, Input = param_grid[i, :], n_samples = n_needed);
                for sample_idx in range(n_needed):
                    coef_sample = n_needed_coefs[sample_idx, :].reshape(1, -1);
                    Z0_i = [Z0[i][j] for j in range(n_IC)];
                    traj = latent_dynamics.simulate( 
                                            coefs   = coef_sample, 
                                            IC      = [Z0_i], 
                                            t_Grid  = [t_Grid_np[i]], 
                                            params  = param_grid[i, :].reshape(1, -1));
                    sample_trajectories.append(traj[0]);  # traj[0] is the trajectory for the i-th parameter
                break;
            
            # Sample n_needed coefficient sets for this parameter
            coef_samples = sample_coefs(gp_list = gp_list, Input = param_grid[i, :], n_samples = n_needed);
            
            # Simulate each coefficient set individually and check for divergence
            for sample_idx in range(n_needed):
                total_resample_attempts += 1;
                
                # Extract this sample's coefficients (reshape to (1, n_coef) for simulate())
                coef_sample = coef_samples[sample_idx, :].reshape(1, -1);
                
                # Get IC for this parameter (list of n_IC arrays, each shape (1, n_z))
                Z0_i = [Z0[i][j] for j in range(n_IC)];
                
                # Simulate: returns list[list[array]], outer list has 1 element (1 param), 
                # inner list has n_IC elements
                traj = latent_dynamics.simulate( 
                                        coefs   = coef_sample, 
                                        IC      = [Z0_i], 
                                        t_Grid  = [t_Grid_np[i]], 
                                        params  = param_grid[i, :].reshape(1, -1));
                
                # Check if this trajectory diverged (check all ICs for this parameter)
                is_divergent = False;
                for j in range(n_IC):
                    max_magnitude = numpy.max(numpy.abs(traj[0][j]));  # traj[0] = first (only) param
                    if max_magnitude > divergence_threshold or not numpy.isfinite(max_magnitude):
                        is_divergent = True;
                        LOGGER.warning(
                            f"Parameter {i}, sample {len(valid_samples)+1}/{n_samples}: diverged "
                            f"(max magnitude: {max_magnitude:.2e} at IC {j}). Resampling."
                        );
                        break;
                
                # If non-divergent, keep it
                if not is_divergent:
                    valid_samples.append(len(valid_samples));
                    sample_trajectories.append(traj[0]);  # traj[0] is list of n_IC trajectories
                    
        # Now store all valid samples for this parameter in LatentStates
        for sample_idx in range(n_samples):
            for j in range(n_IC):
                # sample_trajectories[sample_idx][j] has shape (n_t_i, 1, n_z)
                LatentStates[i][j][:, sample_idx, :] = sample_trajectories[sample_idx][j][:, 0, :];

    # All done!
    return LatentStates;



def Rollout_Error_and_STD(  model           : torch.nn.Module,
                            physics         : Physics,
                            param_space     : ParameterSpace,
                            latent_dynamics : LatentDynamics,
                            gp_list         : list[GaussianProcessRegressor],
                            t_Test          : list[torch.Tensor],
                            U_Test          : list[list[torch.Tensor]],
                            n_samples       : int,
                            trainer         : Trainer) -> tuple[numpy.ndarray, numpy.ndarray, list[list[numpy.ndarray]], list[list[numpy.ndarray]]]:
    r"""
    This function computes the relative error and STD between the FOM solution and its 
    prediction when we rollout the FOM solution using the the ICs and mean of the posterior 
    distribution of the coefficients for each combination of parameter values.
    
    To do this, we first sample the posterior distribution of the coefficients for each combination 
    of parameter values and solve the latent dynamics forward in time using each sample (as well as
    the mean of the posterior distribution). We then decode the latent trajectories to get a set of 
    FOM solutions. We then compute the relative error between the mean predicted solution and the 
    true solution for each frame of each derivative of the FOM solution for each combination of 
    parameter values. We then find the maximum relative error (across the frames and components) 
    for each derivative for each combination of parameter values. 
    
    We also compute the STD (across the samples) of each frame of each derivative of the FOM 
    solution for each combination of the parameter values. We then find the maximum STD (across 
    the frames and components) for each derivative for each combination of parameter values.

    Note: If X_1, ... , X_M \in \mathbb{R}^N are vectors then the STD of this collection is the 
    vector whose i'th component holds the (sample) STD of {X_1[i], ... , X_M[i]}.
    
    Note: If X, Y in \mathbb{R}^N are vectors then we define the relative error of X relative to 
    Y as the vector whose i'th component is given by (x_i - y_i)/||y||_{\inf}. 


    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    model : torch.nn.Module
        For each combinations of parameters, we find the model's latent dynamics for that 
        combination and solve them forward in time. 

    physics : Physics
        A Physics object that we use to fetch the initial condition for each combination of 
        parameter values.

    param_space : ParameterSpace
        A ParameterSpace object which holds the testing parameters.
    
    latent_dynamics : LatentDynamics
        The LatentDynamics object we use to generate the latent space data. For each combination 
        of parameter values, we fetch the corresponding coefficients to define the latent dynamics.
    
    gp_list : list, len = c_coefs
        A set of trained gaussian project objects. The i'th one represents a gaussian process that
        maps a combination of parameter values to a sample for the i'th coefficient in the latent
        dynamics. For each combination of parameter values, we sample the posterior distribution of
        each GP; we use these samples to build samples of the latent dynamics, which we can use 
        to sample the predicted dynamics produced by that combination of parameter values.

    t_Test : list[torch.Tensor], len = n_test
        i'th element is a 1d numpy.ndarray object of length n_t(i) whose j'th element holds the 
        value of the j'th time value at which we solve the latent dynamics for the i'th combination
        of parameter values.

    U_Test : list[list[torch.Tensor]], len = n_test
        i'th element is an n_IC element list whose j'th element is a torch.Tensor of shape 
        (n_t(i), ...) whose k, ... slice holds the k'th frame of the j'th time derivative of the
        FOM model when we use the i'th combination of parameter values to define the FOM model.

    n_samples : int
        The number of samples we draw from each GP's posterior distribution. Each sample gives us 
        a set of coefficients which we can use to define the latent dynamics that we then solve 
        forward in time. 


    
    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    max_Rel_Error, max_STD, Rel_Error, STD

    max_Rel_Error : numpy.ndarray, shape = (n_Test, n_IC)
        i, j element holds the maximum of rel_error[i][j] (see below).
    
    max_STD : numpy.ndarray, shape = (n_Test, n_IC)
        i, j element holds the maximum of STD[i][j] (see below).

    Rel_Error : list[list[numpy.ndarray]], len = n_Test
        i'th element is an n_IC element list whose j'th element is an numpy.ndarray of shape 
        n_t_i, where n_t_i is the number of time steps in the time series for the i'th combination
        of testing parameters. The k'th element of this array holds
            mean(u_Rollout[i][j][k, ...] - u_True[i][j][k, ...]) / std(u_True[i][j])
    
    STD : list[list[numpy.ndarray]], len = n_Test
        i'th element is an n_IC element list whose j'th element is an numpy.ndarray whose shape
        matches that of U_Test[i][j]. The [k, ...] element of this array holds the std (across 
        the samples) of the k'th frame of the reconstruction of the j'th derivative of the FOM 
        solution when we use the i'th combination of testing parameters.
        
    """ 

    # Run checks
    assert isinstance(gp_list,          list),      "type(gp_list) = %s, expected list" % (type(gp_list));
    assert isinstance(t_Test,           list),      "type(t_Test) = %s, expected list" % (type(t_Test));
    assert isinstance(U_Test,           list),      "type(U_Test) = %s, expected list" % (type(U_Test));
    assert isinstance(n_samples,        int),       "type(n_samples) = %s, expected int" % (type(n_samples));
    assert len(t_Test)  == len(U_Test),             "len(t_Test) = %d, len(U_Test) %d" % (len(t_Test), len(U_Test));
    assert len(gp_list) == latent_dynamics.n_coefs, "len(gp_list) = %d, expected %d" % (len(gp_list), latent_dynamics.n_coefs);

    
    # Fetch the number of testing parameter combinations.
    n_Test  : int   = len(U_Test);   
    
    # Run additional checks.
    param_test  : numpy.ndarray         = param_space.test_space;
    assert isinstance(param_test,       numpy.ndarray),     "type(param_test) = %s, expected numpy.ndarray" % (type(param_test));
    assert len(param_test.shape)        == 2,               "len(param_test.shape) = %d, expected 2" % (len(param_test.shape));
    assert param_test.shape[0]          == n_Test,          "param_test.shape = %s, n_Test %d" % (str(param_test.shape), n_Test);

    n_IC    : int                       = len(U_Test[0]);
    for i in range(n_Test):
        assert isinstance(U_Test[i],    list),              "type(U_Test[%d]) = %s, expected list" % (i, type(U_Test[i]));
        assert len(U_Test[i])           == n_IC,            "len(U_Test[%d]) = %d, n_IC %d" % (i, len(U_Test[i]), n_IC);
    
        assert isinstance(t_Test[i],    torch.Tensor),      "type(t_Test[%d]) = %s, expected torch.Tensor" % (i, type(t_Test[i]));
        assert len(t_Test[i].shape)     == 1,               "len(t_Test[%d].shape) = %d, expected 1" % (i, len(t_Test[i].shape));
        n_t_i   : int = t_Test[i].shape[0];

        for j in range(n_IC):
            assert isinstance(U_Test[i][j], torch.Tensor),  "type(U_Test[%d][%d]) = %s, expected torch.Tensor" % (i, j, type(U_Test[i][j]));
            assert U_Test[i][j].shape[0]    == n_t_i,       "U_Test[%d][%d].shape = %s, n_t_i = %d" % (i, j, str(U_Test[i][j].shape), n_t_i);
    

    # ---------------------------------------------------------------------------------------------
    # Draw n_samples samples of the posterior distribution.

    # For each combination of parameter values in the testing set, sample the latent coefficients 
    # and solve the latent dynamics forward in time. 
    LOGGER.info("Generating latent dynamics trajectories for %d samples of the coefficients for %d combinations of testing parameter" % (n_samples, n_Test));
    Zis_samples     : list[list[numpy.ndarray]] = sample_roms(model, physics, latent_dynamics, gp_list, param_test, t_Test, n_samples, trainer = trainer);    # len = n_test. i'th element is an n_IC element list whose j'th element has shape (n_t(i), n_samples, n_z)

    LOGGER.info("Generating latent dynamics trajectories using posterior distribution means for %d combinations of testing parameter" % (n_Test));
    Zis_mean        : list[list[numpy.ndarray]] = average_rom(model, physics, latent_dynamics, gp_list, param_test, t_Test, trainer = trainer);               # len = n_test. i'th element is an n_IC element list whose j'th element has shape (n_t(i), n_z)
        

    # ---------------------------------------------------------------------------------------------
    # Set up Rel_Error, STD, max_Rel_Error, and max_STD.

    STD         : list[list[numpy.ndarray]] = [];           # (n_Test)
    Rel_Error   : list[list[numpy.ndarray]] = [];           # (n_Test)

    for i in range(n_Test):
        # Initialize lists for the i'th combination of parameter values
        STD_i       : list[numpy.ndarray]   = [];
        Rel_Error_i : list[numpy.ndarray]   = [];

        # Fetch n_t_i.
        n_t_i : int = t_Test[i].shape[0];

        # Build an array for each derivative of the FOM solution.
        for j in range(n_IC):
            STD_i.append(numpy.zeros_like(U_Test[i][j].numpy()));
            Rel_Error_i.append(numpy.zeros(n_t_i, dtype = numpy.float32));

        # Append the lists for the i'th combination to the overall lists.
        STD.append(STD_i);
        Rel_Error.append(Rel_Error_i);
    
    max_Rel_Error   = numpy.empty((n_Test, n_IC), dtype = numpy.float32);
    max_STD         = numpy.empty((n_Test, n_IC), dtype = numpy.float32);



    # ---------------------------------------------------------------------------------------------
    # Compute std, max_std. 

    # If the workflow uses normalization, U_Test and decoded predictions are in normalized
    # units. De-normalize here for meaningful physical errors/plots using the trainer.
    use_denorm : bool = hasattr(trainer, "has_normalization") and trainer.has_normalization();

    if(isinstance(model, Autoencoder) or isinstance(model, CNN_3D_Autoencoder)):
        for i in range(n_Test):
            # -------------------------------------------------------------------------------------
            # Relative Error

            # Decode the mean latent trajectories for each combination of parameter values.
            U_Pred_Mean_i       : numpy.ndarray = model.Decode(torch.Tensor(Zis_mean[i][0]))[0].detach().numpy();
            if use_denorm:
                U_Pred_Mean_i = trainer.denormalize_np(U_Pred_Mean_i, 0);

            # Fetch the corresponding test predictions.
            U_Test_i            : numpy.ndarray = U_Test[i][0].detach().numpy();                # (n_t_i, physics.Frame_Shape)
            if use_denorm:
                U_Test_i = trainer.denormalize_np(U_Test_i, 0);

            # Compute the std of the components of the FOM solution.
            U_Test_i_std   : float = numpy.std(U_Test_i);

            # For each frame, compute the relative error between the true and predicted FOM solutions.
            # We normalize the error by the std of the true solution.
            n_t_i : int = U_Test_i.shape[0];
            for k in range(n_t_i):
                Rel_Error[i][0][k] = numpy.mean(numpy.abs(U_Pred_Mean_i[k, ...] - U_Test_i[k, ...]))/U_Test_i_std;
            
            # Now compute the corresponding element of max_Rel_Error
            max_Rel_Error[i, 0] = Rel_Error[i][0].max();
        

            # -------------------------------------------------------------------------------------
            # Standard Deviation

            # Set up an array to hold the decoding of latent trajectory.
            FOM_Frame_Shape : list[int]         = physics.Frame_Shape;
            U_Pred_i        : numpy.ndarray     = numpy.empty([n_t_i, n_samples] + FOM_Frame_Shape, dtype = numpy.float32);

            # Decode the latent trajectory for each sample.
            for j in range(n_samples):
                U_Pred_ij   : numpy.ndarray     = model.Decode(torch.Tensor(Zis_samples[i][0][:, j, :]))[0].detach().numpy();
                U_Pred_i[:, j, ...]             = U_Pred_ij;
        
            # Compute the STD across the sample axis.
            STD_i0          = numpy.std(U_Pred_i, axis = 1);
            STD[i][0]       = trainer.scale_std_np(STD_i0, 0) if use_denorm else STD_i0;
            
            # Compute max STD using robust metric: average across spatial dimensions, then max over time
            # This prevents single outlier nodes from dominating the metric.
            STD_i0_spatial_avg : numpy.ndarray = STD[i][0].mean(axis=tuple(range(1, STD[i][0].ndim)));  # Average over spatial dims
            max_STD[i, 0]      : numpy.float32 = STD_i0_spatial_avg.max();  # Max over time only
        
    

    elif(isinstance(model, Autoencoder_Pair)):
        for i in range(n_Test):
            # -------------------------------------------------------------------------------------
            # Relative Error

            # Decode the mean latent trajectories for each combination of parameter values.
            U_Pred_Mean_i       : list[torch.Tensor]    = model.Decode(torch.Tensor(Zis_mean[i][0]), torch.Tensor(Zis_mean[i][1]));
            D_Pred_Mean_i       : numpy.ndarray         = U_Pred_Mean_i[0].detach().numpy();  # (n_t_i, physics.Frame_Shape)
            V_Pred_Mean_i       : numpy.ndarray         = U_Pred_Mean_i[1].detach().numpy();  # (n_t_i, physics.Frame_Shape)
            if use_denorm:
                D_Pred_Mean_i = trainer.denormalize_np(D_Pred_Mean_i, 0);
                V_Pred_Mean_i = trainer.denormalize_np(V_Pred_Mean_i, 1);

            # Fetch the corresponding test predictions.
            D_Test_i            : numpy.ndarray         = U_Test[i][0].detach().numpy();       # (n_t_i, physics.Frame_Shape)
            V_Test_i            : numpy.ndarray         = U_Test[i][1].detach().numpy();       # (n_t_i, physics.Frame_Shape)
            if use_denorm:
                D_Test_i = trainer.denormalize_np(D_Test_i, 0);
                V_Test_i = trainer.denormalize_np(V_Test_i, 1);
            
            # Compute the std of the components of the FOM solution.
            D_Test_i_std        : float                 = numpy.std(D_Test_i);
            V_Test_i_std        : float                 = numpy.std(V_Test_i);

            # For each frame, compute the relative error between the true and predicted FOM solutions.
            # We normalize the error by the std of the true solution.
            n_t_i : int = D_Test_i.shape[0];
            for k in range(n_t_i):
                Rel_Error[i][0][k] = numpy.mean(numpy.abs(D_Pred_Mean_i[k, ...] - D_Test_i[k, ...]))/D_Test_i_std;
                Rel_Error[i][1][k] = numpy.mean(numpy.abs(V_Pred_Mean_i[k, ...] - V_Test_i[k, ...]))/V_Test_i_std;

            # Now compute the corresponding element of max_Rel_Error
            max_Rel_Error[i, 0] = Rel_Error[i][0].max();
            max_Rel_Error[i, 1] = Rel_Error[i][1].max();


            # -------------------------------------------------------------------------------------
            # Standard Deviation

            # Set up an array to hold the decoding of latent trajectory.
            FOM_Frame_Shape : list[int]         = physics.Frame_Shape;
            D_Pred_i        : numpy.ndarray     = numpy.empty([n_t_i, n_samples] + FOM_Frame_Shape, dtype = numpy.float32);
            V_Pred_i        : numpy.ndarray     = numpy.empty([n_t_i, n_samples] + FOM_Frame_Shape, dtype = numpy.float32);

            # Decode the latent trajectory for each sample.
            for j in range(n_samples):
                U_Pred_ij   : list[torch.Tensor]    = model.Decode(torch.Tensor(Zis_samples[i][0][:, j, :]), torch.Tensor(Zis_samples[i][1][:, j, :]));
                D_Pred_i[:, j, ...]                 = U_Pred_ij[0].detach().numpy();
                V_Pred_i[:, j, ...]                 = U_Pred_ij[1].detach().numpy();

            # Compute the STD across the sample axis.
            STD_D = numpy.std(D_Pred_i, axis = 1);
            STD_V = numpy.std(V_Pred_i, axis = 1);
            STD[i][0]       = trainer.scale_std_np(STD_D, 0) if use_denorm else STD_D;
            STD[i][1]       = trainer.scale_std_np(STD_V, 1) if use_denorm else STD_V;

            # Compute max STD using robust metric: average across spatial dimensions, then max over time
            # This prevents single outlier nodes from dominating the metric.
            STD_D_spatial_avg : numpy.ndarray = STD[i][0].mean(axis=tuple(range(1, STD[i][0].ndim)));  # Average over spatial dims
            STD_V_spatial_avg : numpy.ndarray = STD[i][1].mean(axis=tuple(range(1, STD[i][1].ndim)));  # Average over spatial dims
            max_STD[i, 0]     : numpy.float32 = STD_D_spatial_avg.max();  # Max over time only
            max_STD[i, 1]     : numpy.float32 = STD_V_spatial_avg.max();  # Max over time only  


    # All done!
    return max_Rel_Error, max_STD, Rel_Error, STD;