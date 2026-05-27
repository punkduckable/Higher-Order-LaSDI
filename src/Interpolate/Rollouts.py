# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

import  sys;
import  os;
Physics_Path        : str   = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "Physics"));
LD_Path             : str   = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "LatentDynamics"));
EncoderDecoder_Path : str   = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "EncoderDecoder"));
Interpolate_Path    : str   = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "Interpolate"));
Utilities_Path      : str   = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "Utilities"));
sys.path.append(Physics_Path);
sys.path.append(LD_Path);
sys.path.append(EncoderDecoder_Path);
sys.path.append(Interpolate_Path);
sys.path.append(Utilities_Path);

import  torch;
import  numpy;
from    Interpolate                 import  Interpolate;
from    Physics                     import  Physics;
from    LatentDynamics              import  LatentDynamics;
from    ParameterSpace              import  ParameterSpace;
from    EncoderDecoder              import  EncoderDecoder;
from    Trainer                     import  Trainer;

import  logging;
LOGGER : logging.Logger = logging.getLogger(__name__);





# -------------------------------------------------------------------------------------------------
# Rollout using mean LD coefficients
# -------------------------------------------------------------------------------------------------

def Mean_Rollout(   encoder_decoder : EncoderDecoder, 
                    physics         : Physics, 
                    latent_dynamics : LatentDynamics, 
                    interpolator    : Interpolate, 
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

    encoder_decoder : EncoderDecoder
        The actual EncoderDecoder object that we use to map the ICs into the latent space. physics, 
        latent_dynamics, and EncoderDecoder should have the same number of initial conditions.

    physics : Physics
        Allows us to get the latent IC solution for each combination of parameter values. physics, 
        latent_dynamics, and EncoderDecoder should have the same number of initial conditions.
    
    latent_dynamics : LatentDynamics
        describes how we specify the dynamics in the EncoderDecoder's latent space. We assume that 
        physics, latent_dynamics, and EncoderDecoder all have the same number of initial 
        conditions.

    interpolator : Interpolate
        Interpolator object that returns native latent-dynamics coefficient dictionaries from the
        posterior mean, posterior standard deviation, or posterior samples.

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

    assert isinstance(t_Grid, list),                "type(t_Grid) = %s, expected list" % (type(t_Grid) == list);
    assert len(t_Grid)  == n_param,                 "len(t_Grid) = %d, n_param %d" % (len(t_Grid), n_param);

    n_IC    : int   = latent_dynamics.n_IC;
    n_z     : int   = latent_dynamics.n_z;
    assert encoder_decoder.n_IC == n_IC,            "encoder_decoder.n_IC = %d, n_IC %d" % (encoder_decoder.n_IC, n_IC);
    assert physics.n_IC         == n_IC,            "physics.n_IC = %d, n_IC %d" % (physics.n_IC, n_IC);


    # For each parameter in param_grid, fetch the corresponding initial condition and then encode
    # it. This gives us a list whose i'th element holds the encoding of the i'th initial condition.
    LOGGER.debug("Fetching latent space initial conditions for %d combinations of parameters." % n_param);
    Z0      : list[list[numpy.ndarray]] = encoder_decoder.latent_initial_conditions(param_grid, physics, trainer = trainer);

    LOGGER.debug("Finding native coefficient means from the interpolator");
    coef_means = [interpolator.mean(param_grid[i, :]) for i in range(n_param)];

    # Make each element of t_Grid into a numpy.ndarray of shape (1, n_t(i)). This is what 
    # simulate expects.
    t_Grid_np : list[numpy.ndarray] = [];
    for i in range(n_param):
        if(isinstance(t_Grid[i], torch.Tensor)):
            t_Grid_np.append(t_Grid[i].detach().cpu().numpy());
        else:
            t_Grid_np.append(t_Grid[i]);
        t_Grid_np[i] = t_Grid_np[i].reshape(1, -1);

    # Simulate the laten dynamics! For each testing parameter, use the mean value of each posterior 
    # distribution to define the coefficients. 
    LOGGER.info("simulating initial conditions for %d combinations of parameters forward in time" % n_param);
    Zis : list[list[numpy.ndarray]] = latent_dynamics.simulate( coefs   = coef_means, 
                                                                IC      = Z0, 
                                                                t_Grid  = t_Grid_np,
                                                                params  = param_grid);
    
    # At this point, Zis[i][j] has shape (n_t_i, 1, n_z). We remove the extra dimension.
    for i in range(n_param):
        n_t_i   : int   = t_Grid_np[i].shape[1];
        for j in range(n_IC):
            Zis[i][j] = Zis[i][j].reshape(n_t_i, n_z);
    
    # All done!
    return Zis;





# -------------------------------------------------------------------------------------------------
# Rollout using LD coefficient samples
# -------------------------------------------------------------------------------------------------

def Sample_Rollouts(encoder_decoder : EncoderDecoder, 
                    physics         : Physics, 
                    latent_dynamics : LatentDynamics, 
                    interpolator    : Interpolate, 
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

    encoder_decoder : EncoderDecoder
        An EncoderDecoder (i.e., autoencoder). We use this to map the FOM IC's (which we can get 
        from physics) to the latent space using the EncoderDecoder's encoder. physics, 
        latent_dynamics, and encoder_decoder should have the same number of initial conditions.

    physics : Physics
        allows us to find the IC for a particular combination of parameter values. physics, 
        latent_dynamics, and encoder_decoder should have the same number of initial conditions.
    
    latent_dynamics : LatentDynamics
        describes how we specify the dynamics in the encoder_decoder's latent space. We use this
        to simulate the latent dynamics forward in time. physics, latent_dynamics, and 
        encoder_decoder should have the same number of initial conditions.

    interpolator : Interpolate
        Interpolator object that samples native latent-dynamics coefficient dictionaries at each
        requested parameter value.

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
    assert isinstance(t_Grid, list), "type(t_Grid) = %s, expected list" % (type(t_Grid) == list);
    assert isinstance(n_samples, int), "type(n_samples) = %s, expected int" % (type(n_samples) == int);

    assert isinstance(param_grid, numpy.ndarray), "type(param_grid) = %s, expected numpy.ndarray" % (type(param_grid));
    assert len(param_grid.shape)    == 2, "len(param_grid.shape) = %d, expected 2" % (len(param_grid.shape));
    n_param     : int               = param_grid.shape[0];
    n_p         : int               = param_grid.shape[1];

    assert len(t_Grid)              == n_param, "len(t_Grid) = %d, n_param %d" % (len(t_Grid), n_param);
    for i in range(n_param):
        assert isinstance(t_Grid[i], numpy.ndarray) or isinstance(t_Grid[i], torch.Tensor), "type(t_Grid[%d]) = %s, expected numpy.ndarray or torch.Tensor" % (i, type(t_Grid[i]));

    n_IC        : int               = latent_dynamics.n_IC;
    n_z         : int               = encoder_decoder.n_z;
    assert physics.n_IC             == n_IC, "physics.n_IC = %d, n_IC %d" % (physics.n_IC, n_IC);
    assert encoder_decoder.n_IC     == n_IC, "encoder_decoder.n_IC = %d, n_IC %d" % (encoder_decoder.n_IC, n_IC);


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
    Z0      : list[list[numpy.ndarray]] = encoder_decoder.latent_initial_conditions(param_grid, physics, trainer = trainer);


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
                    f"  - EncoderDecoder hasn't converged yet (train longer)\n"
                    f"  - GP variance is too high (increase alpha in GaussianProcess.py)\n"
                    f"  - Latent dynamics are fundamentally unstable at this parameter value"
                );

                # Generate the remaining samples even if they might diverge
                coef_samples_needed = [interpolator.sample(param_grid[i, :]) for _ in range(n_needed)];
                for sample_idx in range(n_needed):
                    coef_sample = coef_samples_needed[sample_idx];
                    Z0_i = [Z0[i][j] for j in range(n_IC)];
                    traj = latent_dynamics.simulate( 
                                            coefs   = coef_sample, 
                                            IC      = [Z0_i], 
                                            t_Grid  = [t_Grid_np[i]], 
                                            params  = param_grid[i, :].reshape(1, -1));
                    sample_trajectories.append(traj[0]);  # traj[0] is the trajectory for the i-th parameter
                break;
            
            # Sample n_needed coefficient sets for this parameter
            coef_samples = [interpolator.sample(param_grid[i, :]) for _ in range(n_needed)];
            
            # Simulate each coefficient set individually and check for divergence
            for sample_idx in range(n_needed):
                total_resample_attempts += 1;
                
                # Extract this sample's native coefficient dictionary.
                coef_sample = coef_samples[sample_idx];
                
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