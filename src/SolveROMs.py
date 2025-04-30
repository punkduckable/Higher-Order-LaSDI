# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

import  sys;
import  os;
Physics_Path    : str   = os.path.abspath(os.path.join(os.path.dirname(__file__), "Physics"));
LD_Path         : str   = os.path.abspath(os.path.join(os.path.dirname(__file__), "LatentDynamics"));
Utilities_Path  : str   = os.path.abspath(os.path.join(os.path.dirname(__file__), "Utilities"));
sys.path.append(Physics_Path);
sys.path.append(LD_Path);
sys.path.append(Utilities_Path);

import  torch;
import  numpy;
from    sklearn.gaussian_process    import  GaussianProcessRegressor;

from    GaussianProcess             import  eval_gp, sample_coefs;
from    Physics                     import  Physics;
from    LatentDynamics              import  LatentDynamics;
from    Model                       import  Autoencoder, Autoencoder_Pair;



# -------------------------------------------------------------------------------------------------
# Simulate latent dynamics
# -------------------------------------------------------------------------------------------------

def average_rom(model           : torch.nn.Module, 
                physics         : Physics, 
                latent_dynamics : LatentDynamics, 
                gp_list         : list[GaussianProcessRegressor], 
                param_grid      : numpy.ndarray,
                t_Grid          : list[numpy.ndarray] | list[torch.Tensor]) -> list[numpy.ndarray]:
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

    
    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------
    
    Zis : list[numpy.ndarray], len = n_param
        i'th element is a 2d numpy.ndarray object of shape (n_t_i, n_z) whose j, k element holds 
        the k'th component of the latent solution at the j'th time step when we the means of the 
        posterior distribution for the i'th combination of parameter values to define the latent 
        dynamics.
    """

    # Checks. 
    assert(isinstance(param_grid, numpy.ndarray));
    assert(param_grid.ndim    == 2);
    n_param : int   = param_grid.shape[0];
    n_p     : int   = param_grid.shape[1];

    assert(isinstance(gp_list, list));
    assert(isinstance(t_Grid, list));
    assert(len(t_Grid)  == n_param);

    n_IC    : int   = latent_dynamics.n_IC;
    n_z     : int   = latent_dynamics.n_z;
    assert(model.n_IC       == n_IC);
    assert(physics.n_IC     == n_IC);


    # For each parameter in param_grid, fetch the corresponding initial condition and then encode
    # it. This gives us a list whose i'th element holds the encoding of the i'th initial condition.
    Z0      : list[list[numpy.ndarray]] = model.latent_initial_conditions(param_grid, physics);

    # Evaluate each GP at each combination of parameter values. This returns two arrays, the 
    # first of which is a 2d array of shape (n_param, n_coef) whose i,j element specifies the mean 
    # of the posterior distribution for the j'th coefficient at the i'th combination of parameter 
    # values.
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
    Zis : list[list[numpy.ndarray]] = latent_dynamics.simulate( coefs   = post_mean, 
                                                                IC      = Z0, 
                                                                t_Grid  = t_Grid);
    
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
                t_Grid          : list[numpy.ndarray] | list[torch.Tensor],
                n_samples       : int) ->           list[list[numpy.ndarray]]:
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
    assert(isinstance(gp_list, list));
    assert(isinstance(t_Grid, list));
    assert(isinstance(n_samples, int));

    assert(isinstance(param_grid, numpy.ndarray));
    assert(len(param_grid.shape)    == 2);
    n_param     : int               = param_grid.shape[0];
    n_p         : int               = param_grid.shape[1];

    assert(len(t_Grid)              == n_param);
    for i in range(n_param):
        assert(isinstance(t_Grid[i], numpy.ndarray) or isinstance(t_Grid[i], torch.Tensor));

    n_coef      : int               = len(gp_list);
    n_IC        : int               = latent_dynamics.n_IC;
    n_z         : int               = model.n_z;
    assert(physics.n_IC             == n_IC);
    assert(model.n_IC               == n_IC);


    # Make each element of t_Grid into a numpy.ndarray of shape (1, n_t(i)). This is what 
    # simulate expects.
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
    Z0      : list[list[numpy.ndarray]] = model.latent_initial_conditions(param_grid, physics);


    # Now, for each combination of parameters, draw n_samples samples from the posterior
    # distributions for each coefficient at that combination of parameters. We store these samples 
    # in an n_param element list whose k'th element is a (n_sample, n_coef) array whose i, j 
    # element stores the i'th sample from the posterior distribution for the j'th coefficient at 
    # the k'th combination of parameter values.
    coefs_by_parameter  : list[numpy.ndarray]       = [sample_coefs(gp_list = gp_list, Input = param_grid[i, :], n_samples = n_samples) for i in range(n_param)];

    # Reorganize the coef samples into an n_samples element list whose i'th element is an 
    # array of shape (n_param, n_coef) whose j, k element holds the i'th sample of the k'th 
    # coefficient when we sample from the posterior distribution evaluated at the j'th combination
    # of parameter values.
    coefs_by_samples    : list[numpy.ndarray]   = [];
    for k in range(n_samples):
        coefs_by_samples.append(numpy.empty((n_param, n_coef), dtype = numpy.float32));
    
    for i in range(n_param):
        for k in range(n_samples):
            coefs_by_samples[k][i, :] = coefs_by_parameter[i][k, :];


    # Setup a list to hold the simulated dynamics. There are n_param parameters. For each 
    # combination of parameter values, we have n_IC initial conditions. For each IC, we 
    # have n_samples simulations, each of which has n_t_i frames, each of which has n_z components
    # Thus, we need a n_param element list whose i'th element is a n_IC element list whose 
    # j'th element is a 3d array of shape n_samples, n_t_i, n_z.
    LatentStates : list[list[numpy.ndarray]] = [];
    for i in range(n_param):
        LatentStates_i  : list[numpy.ndarray]   = [];
        n_t_i           : int                   = t_Grid_np[i].shape[1];

        for j in range(n_IC):
            LatentStates_i.append(numpy.empty((n_t_i, n_samples, n_z), dtype = numpy.float32));
        LatentStates.append(LatentStates_i);


    # Simulate each set of dynamics forward in time. We generate this one sample at a time. For 
    # each sample, we use the k'th set of coefficients. There is one set of coefficients per 
    # sample. For each sample, we use the same ICs and the same t_Grid.
    for k in range(n_samples):
        # This yields an n_param element list whose i'th element is an n_IC element list whose
        # j'th element is an numpy.ndarray of shape (n_t_i, 1, n_z). We store this in the 
        # (:, k, :) elements of LatentStates[i][j]
        LatentStates_kth_sample : list[list[numpy.ndarray]] = latent_dynamics.simulate( coefs   = coefs_by_samples[k], 
                                                                                        IC      = Z0,
                                                                                        t_Grid  = t_Grid_np);
    
        for i in range(n_param):
            for j in range(n_IC):
                LatentStates[i][j][:, k, :] = LatentStates_kth_sample[i][j][:, 0, :];

    # All done!
    return LatentStates;



def get_FOM_max_std(model : torch.nn.Module, LatentStates : list[list[numpy.ndarray]]) -> int:
    r"""
    We find the combination of parameter values which produces with FOM solution with the greatest
    variance.

    To make that more precise, consider the set of all FOM frames generated by decoding the latent 
    trajectories in LatentStates. We assume these latent trajectories were generated as follows:
    For a combination of parameter values, we sampled the posterior coefficient distribution for 
    that combination of parameter values. For each set of coefficients, we solved the corresponding
    latent dynamics forward in time. We assume the user used the same time grid for all latent 
    trajectories for that combination of parameter values.
    
    After solving, we end up with a collection of latent trajectories for that parameter value. 
    We then decoded each latent trajectory, which gives us a collection of FOM trajectories for 
    that combination of parameter values. At each value in the time grid, we have a collection of
    frames. We can compute the variance of each component of the frames at that time value for that
    combination of parameter values. We do this for each time value and for each combination of
    parameter values and then return the index for the combination of parameter values that gives
    the largest variance (among all components at all time frames).

    Stated another way, we find the following:
        argmax_{i}[ STD[ { Decoder(LatentStates[i][0][p, q, :])_k : p \in {1, 2, ... , n_samples} } ]
                    |   k \in {1, 2, ... , n_{FOM}},
                        i \in {1, 2, ... , n_param},
                        q \in {1, 2, ... , n_t(i)} ]
    

    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    model : torch.nn.Module
        The model. We assume the solved dynamics (whose frames are stored in Zis) 
        take place in the model's latent space. We use this to decode the solution frames.

    LatentStates : list[list[torch.Tensor]], len = n_param
        i'th element is an n_IC element list whose j'th element is a 3d tensor of shape 
        (n_samples, n_t(i), n_z) whose p, q, r element holds the r'th component of the j'th 
        component of the latent solution at the q'th time step when we solve the latent dynamics 
        using the p'th set of coefficients we got by sampling the posterior distribution for the 
        i'th combination of parameter values. 


    -----------------------------------------------------------------------------------------------
    Returns:
    -----------------------------------------------------------------------------------------------

    m_index : int
        The index of the testing parameter that gives the largest standard deviation. See the 
        description above for details.
    """
    
    # Run checks.
    assert(isinstance(LatentStates,         list));
    assert(isinstance(LatentStates[0],      list));
    assert(isinstance(LatentStates[0][0],   numpy.ndarray));
    assert(len(LatentStates[0][0].shape)    == 3);

    n_param : int   = len(LatentStates);
    n_IC    : int   = len(LatentStates[0]);
    n_z     : int   = LatentStates[0][0].shape[2];

    assert(n_z  == model.n_z);

    for i in range(n_param):
        assert(isinstance(LatentStates[i], list));
        assert(len(LatentStates[i]) == n_IC);

        assert(isinstance(LatentStates[i][0],   numpy.ndarray));
        assert(len(LatentStates[i][0].shape)    == 3);
        n_samples_i : int   = LatentStates[i][0].shape[0];
        n_t_i       : int   = LatentStates[i][0].shape[1];

        for j in range(1, n_IC):
            assert(isinstance(LatentStates[i][j],   numpy.ndarray));
            assert(len(LatentStates[i][j].shape)    == 3);
            assert(LatentStates[i][j].shape[0]      == n_samples_i);
            assert(LatentStates[i][j].shape[1]      == n_t_i);
            assert(LatentStates[i][j].shape[2]      == n_z);


    # Find the index that gives the largest STD!
    max_std     : float     = 0.0;
    m_index     : int       = 0;
    
    if(isinstance(model, Autoencoder)):
        assert(n_IC == 1);

        for i in range(n_param):
            # Fetch the set of latent trajectories for the i'th combination of parameter values.
            # Z_i is a 3d tensor of shape (n_samples_i, n_t_i, n_z), where n_samples_i is the 
            # number of samples of the posterior distribution for the i'th combination of parameter 
            # values, n_t_i is the number of time steps in the latent dynamics solution for the 
            # i'th combination of parameter values, nd n_z is the dimension of the latent space. 
            # The p, q, r element of Zi is the r'th component of the q'th frame of the latent 
            # solution corresponding to p'th sample of the posterior distribution for the i'th 
            # combination of parameter values.
            Z_i             : torch.Tensor  = torch.Tensor(LatentStates[i][0]);

            # Now decode the frames, one sample at a time.
            n_samples_i     : int           = Z_i.shape[0];
            n_t_i           : int           = Z_i.shape[1];
            X_Pred_i        : numpy.ndarray = numpy.empty([n_samples_i, n_t_i] + model.reshape_shape, dtype = numpy.float32);
            for j in range(n_samples_i):
                X_Pred_i[j, ...] = model.Decode(Z_i[j, :, :]).detach().numpy();

            # Compute the standard deviation across the sample axis. This gives us an array of shape 
            # (n_t, n_FOM) whose i,j element holds the (sample) standard deviation of the j'th component 
            # of the i'th frame of the FOM solution. In this case, the sample distribution consists of 
            # the set of j'th components of i'th frames of FOM solutions (one for each sample of the 
            # coefficient posterior distributions).
            X_pred_i_std    : numpy.ndarray = X_Pred_i.std(0);

            # Now compute the maximum standard deviation across frames/FOM components.
            max_std_i       : numpy.float32 = X_pred_i_std.max();

            # If this is bigger than the biggest std we have seen so far, update the maximum.
            if max_std_i > max_std:
                m_index : int   = i;
                max_std : float = max_std_i;

        # Report the index of the testing parameter that gave the largest maximum std.
        return m_index


    elif(isinstance(model, Autoencoder_Pair)):
        assert(n_IC == 2);

        for i in range(n_param):
            # Fetch the set of latent trajectories for the i'th combination of parameter values.
            # Z_D_i and Z_D_i are a 3d tensor sof shape (n_samples_i, n_t_i, n_z), where 
            # n_samples_i is the number of samples of the posterior distribution for the i'th 
            # combination of parameter values, n_t_i is the number of time steps in the latent 
            # dynamics solution for the i'th combination of parameter values, nd n_z is the 
            # dimension of the latent space. 
            # 
            # The p, q, r element of Z_D_i is the r'th component of the q'th frame of the latent 
            # displacement corresponding to p'th sample of the posterior distribution for the i'th 
            # combination of parameter values. The components of Z_V_i are analogous but for the 
            # latent velocity. 
            Z_D_i   : torch.Tensor  = torch.Tensor(LatentStates[i][0]);
            Z_V_i   : torch.Tensor  = torch.Tensor(LatentStates[i][1]);

            n_samples_i : int           = Z_D_i.shape[0];
            n_t_i       : int           = Z_D_i.shape[1];
            D_Pred_i    : numpy.ndarray = numpy.empty([n_samples_i, n_t_i] + model.reshape_shape, dtype = numpy.float32);
            for j in range(n_samples_i):
                D_Pred_ij, _ = model.Decode(Latent_Displacement   = Z_D_i[j, :, :], Latent_Velocity    = Z_V_i[j, :, :]);
                D_Pred_i[j, ...] = D_Pred_ij.detach().numpy();

            # Compute the standard deviation across the sample axis. This gives us an array of 
            # shape (n_t, n_FOM) whose i,j element holds the (sample) standard deviation of the 
            # j'th component of the i'th frame of the FOM solution. In this case, the sample 
            # distribution consists of the set of j'th components of i'th frames of FOM solutions 
            # (one for each sample of the coefficient posterior distributions).
            D_Pred_i_std    : numpy.ndarray = D_Pred_i.std(0);

            # Now compute the maximum standard deviation across frames/FOM components.
            max_std_i       : numpy.float32 = D_Pred_i_std.max();

            # If this is bigger than the biggest std we have seen so far, update the maximum.
            if max_std_i > max_std:
                m_index : int   = i;
                max_std : float = max_std_i;

        # Report the index of the testing parameter that gave the largest maximum std.
        return m_index;
    
    
    else:
        raise ValueError("Invalid model type!");