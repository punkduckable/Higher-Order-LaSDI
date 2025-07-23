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

from    GaussianProcess             import  eval_gp, sample_coefs, fit_gps;
from    Physics                     import  Physics;
from    LatentDynamics              import  LatentDynamics;
from    Model                       import  Autoencoder, Autoencoder_Pair;
from    ParameterSpace              import  ParameterSpace;

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
                t_Grid          : list[numpy.ndarray | torch.Tensor]) -> list[numpy.ndarray]:
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
    LOGGER.debug("Fetching latent space initial conditions for %d combinations of parameters." % n_param);
    Z0      : list[list[numpy.ndarray]] = model.latent_initial_conditions(param_grid, physics);

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
                t_Grid          : list[numpy.ndarray | torch.Tensor],
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
    Z0      : list[list[numpy.ndarray]] = model.latent_initial_conditions(param_grid, physics);


    # Now, for each combination of parameters, draw n_samples samples from the posterior
    # distributions for each coefficient at that combination of parameters. We store these samples 
    # in an n_param element list whose k'th element is a (n_sample, n_coef) array whose i, j 
    # element stores the i'th sample from the posterior distribution for the j'th coefficient at 
    # the k'th combination of parameter values.
    LOGGER.debug("Sampling coefficients from the GP posterior distributions");
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
    LOGGER.info("Generating latent trajectories for %d samples of the coefficients." % n_samples);
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
            U_Pred_i        : numpy.ndarray = numpy.empty([n_samples_i, n_t_i] + model.reshape_shape, dtype = numpy.float32);
            for j in range(n_samples_i):
                U_Pred_i[j, ...] = model.Decode(Z_i[j, :, :]).detach().numpy();

            # Compute the standard deviation across the sample axis. This gives us an array of shape 
            # (n_t, n_FOM) whose i,j element holds the (sample) standard deviation of the j'th component 
            # of the i'th frame of the FOM solution. In this case, the sample distribution consists of 
            # the set of j'th components of i'th frames of FOM solutions (one for each sample of the 
            # coefficient posterior distributions).
            U_pred_i_std    : numpy.ndarray = U_Pred_i.std(0);

            # Now compute the maximum standard deviation across frames/FOM components.
            max_std_i       : numpy.float32 = U_pred_i_std.max();

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




def Rollout_Error_and_STD(  model           : torch.nn.Module,
                            physics         : Physics,
                            param_space     : ParameterSpace,
                            latent_dynamics : LatentDynamics,
                            gp_list         : list[GaussianProcessRegressor],
                            t_Test          : list[torch.Tensor],
                            U_Test          : list[list[torch.Tensor]],
                            n_samples       : int,
                            skip_proportion : float) -> tuple[numpy.ndarray, numpy.ndarray, list[list[numpy.ndarray]], list[list[numpy.ndarray]]]:
    """
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

    skip_proportion : float
        The proportion of the time steps we don't compute the relative error or STD for. 
        Specifically, if this number is p, then for each combination parameters, for each time
        step, t, which has t/t_max_i < p (where t_max_i is the maximum time value for that 
        parameter combination), we set the relative error/std to zero.
    
    
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
        of testing parameters. The k'th element of this array holds the relative error between
        k'th frame of the mean prediction and for the j'th derivative of the FOM solution 
        for the i'th combination of parameter values and U_Test[i][j][k, ...]. 
    
    STD : list[list[numpy.ndarray]], len = n_Test
        i'th element is an n_IC element list whose j'th element is an numpy.ndarray whose shape
        matches that of U_Test[i][j]. The [k, ...] element of this array holds the std (across 
        the samples) of the k'th frame of the reconstruction of the j'th derivative of the FOM 
        solution when we use the i'th combination of testing parameters.
        
    """ 

    # Run checks
    assert(isinstance(gp_list,          list));
    assert(isinstance(t_Test,           list));
    assert(isinstance(U_Test,           list));

    param_test  : numpy.ndarray         = param_space.test_space;
    assert(isinstance(param_test,       numpy.ndarray));
    assert(len(param_test.shape)        == 2);
    assert(isinstance(n_samples,        int));
    assert(len(gp_list)                 == latent_dynamics.n_coefs);

    n_Test  : int   = len(U_Test);   
    assert(len(U_Test)                  == n_Test);
    assert(param_test.shape[0]          == n_Test);

    assert(isinstance(U_Test[0],        list));
    n_IC    : int                       = len(U_Test[0]);
    for i in range(n_Test):
        assert(isinstance(U_Test[i],    list));
        assert(len(U_Test[i])           == n_IC);
    
        for j in range(n_IC):
            assert(isinstance(t_Test[j],    torch.Tensor));
            assert(len(t_Test[j].shape)     == 1);
            n_t_j   : int = t_Test[j].shape[0];

            assert(isinstance(U_Test[i][j], torch.Tensor));
            assert(U_Test[i][j].shape[0]    == n_t_j);
    

    # ---------------------------------------------------------------------------------------------
    # Draw n_samples samples of the posterior distribution.

    # For each combination of parameter values in the testing set, sample the latent coefficients 
    # and solve the latent dynamics forward in time. 
    LOGGER.info("Generating latent dynamics trajectories for %d samples of the coefficients for %d combinations of testing parameter" % (n_samples, n_Test));
    Zis_samples     : list[list[numpy.ndarray]] = sample_roms(model, physics, latent_dynamics, gp_list, param_test, t_Test, n_samples);    # len = n_test. i'th element is an n_IC element list whose j'th element has shape (n_t(i), n_samples, n_z)

    LOGGER.info("Generating latent dynamics trajectories using posterior distribution means for %d combinations of testing parameter" % (n_Test));
    Zis_mean        : list[list[numpy.ndarray]] = average_rom(model, physics, latent_dynamics, gp_list, param_test, t_Test);               # len = n_test. i'th element is an n_IC element list whose j'th element has shape (n_t(i), n_z)
        

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

    if(isinstance(model, Autoencoder)):
        for i in range(n_Test):
            # -------------------------------------------------------------------------------------
            # Relative Error

            # Decode the mean latent trajectories for each combination of parameter values.
            U_Pred_Mean_i   : numpy.ndarray = model.Decode(torch.Tensor(Zis_mean[i][0])).detach().numpy();

            # Fetch the corresponding test predictions.
            U_Test_i        : numpy.ndarray = U_Test[i][0].detach().numpy();        # (n_t_i, ...)

            # Compute the Absolute Error.
            Abs_Error_i     : numpy.ndarray = numpy.linalg.norm((U_Pred_Mean_i - U_Test_i).reshape(n_t_i, -1), axis = 1);  # (n_t_i)

            # Compute the norm of each frame.
            Norms_i         : numpy.ndarray = numpy.linalg.norm(U_Test_i.reshape(n_t_i, -1), axis = 1);  # (n_t_i)

            # Compute the relative error for the i'th combination of parameter values.
            for k in range(n_t_i):
                # Replace all zero frames with a norm of 1 to prevent division by 0.
                if(Norms_i[k] == 0): 
                    LOGGER.warning("Frame %d of the true solution for U for testing parameter combination %d is all zero" % (k, i));
                    Norms_i[k] = 1;

                # Set the relative to 0 if the time step is before the skip proportion.
                if(t_Test[i][k] < skip_proportion*t_Test[i][-1]):
                    Rel_Error[i][0][k] = 0.0;
                else:
                    Rel_Error[i][0][k] = Abs_Error_i[k]/Norms_i[k];
            
            # Now compute the corresponding element of max_Rel_Error
            max_Rel_Error[i, 0] = Rel_Error[i][0].max();
        

            # -------------------------------------------------------------------------------------
            # Standard Deviation

            # Set up an array to hold the decoding of latent trajectory.
            FOM_Frame_Shape : list[int]         = physics.Frame_Shape;
            U_Pred_i        : numpy.ndarray     = numpy.empty([n_t_i, n_samples] + FOM_Frame_Shape, dtype = numpy.float32);

            # Decode the latent trajectory for each sample.
            for j in range(n_samples):
                U_Pred_ij   : numpy.ndarray     = model.Decode(torch.Tensor(Zis_samples[i][0][:, j, :])).detach().numpy();
                U_Pred_i[:, j, ...]             = U_Pred_ij;
        
            # Compute the STD across the sample axis.
            STD[i][0]       = numpy.std(U_Pred_i, axis = 1);
            max_STD[i, 0]   = STD[i][0].max();
        
    elif(isinstance(model, Autoencoder_Pair)):
        for i in range(n_Test):
            # -------------------------------------------------------------------------------------
            # Relative Error

            # Decode the mean latent trajectories for each combination of parameter values.
            U_Pred_Mean_i   : list[torch.Tensor]    = model.Decode(torch.Tensor(Zis_mean[i][0]), torch.Tensor(Zis_mean[i][1]));
            D_Pred_Mean_i   : numpy.ndarray         = U_Pred_Mean_i[0].detach().numpy();                # (n_t_i, ...)
            V_Pred_Mean_i   : numpy.ndarray         = U_Pred_Mean_i[1].detach().numpy();                # (n_t_i, ...)

            # Fetch the corresponding test predictions.
            D_Test_i        : numpy.ndarray = U_Test[i][0].detach().numpy();                            # (n_t_i, ...)
            V_Test_i        : numpy.ndarray = U_Test[i][1].detach().numpy();                            # (n_t_i, ...)

            # Compute the Absolute Error.
            n_t_i           : int           = t_Test[i].shape[0];
            D_Abs_Error_i   : numpy.ndarray = numpy.linalg.norm((D_Pred_Mean_i - D_Test_i).reshape(n_t_i, -1), axis = 1);   # (n_t_i)
            V_Abs_Error_i   : numpy.ndarray = numpy.linalg.norm((V_Pred_Mean_i - V_Test_i).reshape(n_t_i, -1), axis = 1);   # (n_t_i)

            # Compute the norm of each frame.
            D_Norms_i       : numpy.ndarray = numpy.linalg.norm(D_Test_i.reshape(n_t_i, -1), axis = 1); # (n_t_i)
            V_Norms_i       : numpy.ndarray = numpy.linalg.norm(V_Test_i.reshape(n_t_i, -1), axis = 1); # (n_t_i)

            # Compute the relative error for the i'th combination of parameter values.
            for k in range(n_t_i):
                # Replace all zero frames with a norm of 1 to prevent division by 0.
                if(D_Norms_i[k] == 0):
                    LOGGER.warning("Frame %d of the true solution for U for testing parameter combination %d is all zero" % (k, i));
                    D_Norms_i[k] = 1;
                elif(V_Norms_i[k] == 0):
                    LOGGER.warning("Frame %d of the true solution for D_t U for testing parameter combination %d is all zero" % (k, i));
                    V_Norms_i[k] = 1;

                # Set the relative to 0 if the time step is before the skip proportion.
                if(t_Test[i][k] < skip_proportion*t_Test[i][-1]):
                    Rel_Error[i][0][k] = 0.0;
                    Rel_Error[i][1][k] = 0.0;
                else:
                    Rel_Error[i][0][k] = D_Abs_Error_i[k]/D_Norms_i[k];
                    Rel_Error[i][1][k] = V_Abs_Error_i[k]/V_Norms_i[k];

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
            STD[i][0]       = numpy.std(D_Pred_i, axis = 1);
            STD[i][1]       = numpy.std(V_Pred_i, axis = 1);

            max_STD[i, 0]   = STD[i][0].max();  
            max_STD[i, 1]   = STD[i][1].max();  


    # All done!
    return max_Rel_Error, max_STD, Rel_Error, STD;