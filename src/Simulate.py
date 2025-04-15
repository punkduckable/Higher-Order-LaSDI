# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

import  sys;
import  os;
Physics_Path    : str   = os.path.abspath(os.path.join(os.path.dirname(__file__), "Physics"));
LD_Path         : str   = os.path.abspath(os.path.join(os.path.dirname(__file__), "LatentDynamics"));
sys.path.append(Physics_Path);
sys.path.append(LD_Path);

import  torch;
import  numpy;
from    sklearn.gaussian_process    import  GaussianProcessRegressor;

from    GPLaSDI                     import  BayesianGLaSDI;
from    GaussianProcess             import  eval_gp, sample_coefs;
from    Physics                     import  Physics;
from    LatentDynamics              import  LatentDynamics;
from    Model                       import  Autoencoder, Autoencoder_Pair;



# -------------------------------------------------------------------------------------------------
# Simulate latent dynamics
# -------------------------------------------------------------------------------------------------

def average_rom(trainer         : BayesianGLaSDI,
                model           : torch.nn.Module, 
                physics         : Physics, 
                latent_dynamics : LatentDynamics, 
                gp_list         : list[GaussianProcessRegressor], 
                param_grid      : numpy.ndarray) -> list[numpy.ndarray]:
    """
    This function simulates the latent dynamics for a set of parameter values by using the mean of
    the posterior distribution for each coefficient's posterior distribution. Specifically, for 
    each parameter combination, we determine the mean of the posterior distribution for each 
    coefficient. We then use this mean to simulate the latent dynamics forward in time (starting 
    from the latent encoding of the FOM initial condition for that combination of coefficients).

    
    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    trainer: A BayesianGLaSDI object that we use to train the model. 

    model: The actual model object that we use to map the ICs into the latent space.

    physics: A "Physics" object that stores the datasets for each parameter combination. 
    
    latent_dynamics: A LatentDynamics object which describes how we specify the dynamics in the
    model's latent space.    

    gp_list: a list of trained GP regressor objects. The number of elements in this list should 
    match the number of columns in param_grid. The i'th element of this list is a GP regressor 
    object that predicts the i'th coefficient. 

    param_grid: A 2d numpy.ndarray object of shape (n_param, n_p), where n_p is the number of 
    parameters and n_param is the number of combinations of parameter values. The i,j element of 
    this array holds the value of the j'th parameter in the i'th combination of parameter values. 


    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------
    
    An n_param element list whose i'th element is a 2d numpy ndarray object of shape (n_t, n_z) 
    whose j, k element holds the k'th component of the latent solution at the j'th time step when 
    we the means of the posterior distribution for the i'th combination of parameter values to 
    define the coefficients in the latent dynamics.
    """

    # Setup
    n_param : int   = param_grid.shape[0];
    n_IC    : int   = latent_dynamics.n_IC;
    n_z     : int   = latent_dynamics.n_z;

    # For each parameter in param_grid, fetch the corresponding initial condition and then encode
    # it. This gives us a list whose i'th element holds the encoding of the i'th initial condition.
    Z0      : list[list[numpy.ndarray]] = model.latent_initial_conditions(param_grid, physics);

    # Evaluate each GP at each combination of parameter values. This returns two arrays, the 
    # first of which is a 2d array of shape (n_param, n_coef) whose i,j element specifies the mean 
    # of the posterior distribution for the j'th coefficient at the i'th combination of parameter 
    # values.
    pred_mean, _ = eval_gp(gp_list, param_grid);

    # For each testing parameter, use the mean value of each posterior distribution to define the 
    # coefficients, solve the corresponding laten dynamics (starting from the corresponding IC 
    # value) and store the resulting solution frames in an n_IC element list whose i'th element 
    # is a 2d numpy ndarray of shape (n_t_i, n_z) whose j, k element holds the k'th component of 
    # the j'th time step of the latent solution when we use the mean of the posterior distribution 
    # for the i'th combination of parameter values to define the latent dynamics coefficients. 
    
    for i in range(n_param):
        # Reshape each element of the IC to have shape (1, n_z), which is what simulate expects
        Z0_i     = Z0[i];
        for d in range(n_IC):
            Z0_i[d] = Z0_i[d].reshape(1, -1);
        
        ith_Zis : list[numpy.ndarray] = latent_dynamics.simulate(coefs = pred_mean[i, :], IC = Z0_i, times = physics.t_grid);
        for d in range(n_IC):
            Zis[d][i, :, :] = ith_Zis[d].reshape(n_t, n_z);

    # All done!
    return Zis;



def sample_roms(model           : torch.nn.Module, 
                physics         : Physics, 
                latent_dynamics : LatentDynamics, 
                gp_list         : list[GaussianProcessRegressor], 
                param_grid      : numpy.ndarray, 
                n_samples       : int) ->           list[numpy.ndarray]:
    """
    This function samples the latent coefficients, solves the corresponding latent dynamics, and 
    then returns the resulting latent solutions. 
    
    Specifically, for each combination of parameter values in the param_grid, we draw n_samples 
    samples of the latent coefficients (from the coefficient posterior distributions evaluated at 
    that parameter value). This gives us a set of n_samples latent dynamics coefficients. For each 
    set of coefficients, we solve the corresponding latent dynamics forward in time and store the 
    resulting solution frames. We do this for each sample and each combination of parameter values,
    resulting in an (n_param, n_sample, n_t, n_z) array of solution frames, which is what we 
    return. Here, n_param is the number of combinations of parameter values.

    
    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    model: A model (i.e., autoencoder). We use this to map the FOM IC's (stored in Physics) to the 
    latent space using the model's encoder.

    physics: A "Physics" object that stores the ICs for each parameter combination. 
    
    latent_dynamics: A LatentDynamics object which describes how we specify the dynamics in the
    model's latent space. We use this to simulate the latent dynamics forward in time.

    gp_list: a list of trained GP regressor objects. The number of elements in this list should 
    match the number of columns in param_grid. The i'th element of this list is a GP regressor 
    object that predicts the i'th coefficient. 

    param_grid: A 2d numpy.ndarray object of shape (number of parameter combination, number of 
    parameters). The i,j element of this array holds the value of the j'th parameter in the i'th 
    combination of parameters. 

    n_samples: The number of samples we want to draw from each posterior distribution for each 
    coefficient evaluated at each combination of parameter values.
    

    
    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------
    
    A list of numpy.ndarrays, each of size [n_test, n_samples, physics.n_t, model.n_z]. If the 
    latent dynamics require n_ID initial conditions (latent_dynamics.n_ID = n_ID), then the 
    returned list has n_ID elements, the d'th one of which is a 4d array whose i, j, k, l element 
    holds the l'th component of the k'th frame of the solution to the d'th derivative of latent 
    dynamics when we use the j'th sample of latent coefficients drawn from the posterior 
    distribution for the i'th combination of parameter values (i'th row of param_grid).
    """

    # The param grid needs to be two dimensional, with the first axis corresponding to which 
    # instance of the parameter values we are using. If there is only one parameter, it may be 1d. 
    # We can fix that by adding on an axis with size 1. 
    if (param_grid.ndim == 1):
        param_grid = param_grid.reshape(1, -1);
    
    # Now fetch the number of combinations of parameter values (rows of param_grid).
    n_param : int = param_grid.shape[0];

    # For each parameter in param_grid, fetch the corresponding initial condition and then encode
    # it. This gives us a list whose i'th element holds the encoding of the i'th initial condition.
    Z0      : list[list[numpy.ndarray]] = model.latent_initial_conditions(param_grid, physics);

    # Now, for each combination of parameters, draw n_samples samples from the posterior
    # distributions for each coefficient at that combination of parameters. We store these samples 
    # in a list of numpy arrays. The k'th list element is a (n_sample, n_coef) array whose i, j 
    # element stores the i'th sample from the posterior distribution for the j'th coefficient at 
    # the k'th combination of parameter values.
    coef_samples : list[numpy.ndarray]  = [sample_coefs(gp_list, param_grid[i], n_samples) for i in range(n_param)];

    # Initialize a list to hold the solutions to the latent dynamics. This is a list of 4d numpy 
    # arrays of shape (n_parm, n_samples, n_t, n_z). The i, j, k, l element of the d'th array holds
    # the holds the l'th component of the  k'th frame of the d'th derivative of the solution to 
    # the latent dynamics when we use the j'th sample of latent coefficients drawn from the 
    # posterior distribution for the i'th combination of parameter values.
    n_IC    : int                   = latent_dynamics.n_IC;
    Zis     : list[numpy.ndarray]   = [];
    for i in range(n_IC):
        Zis.append(numpy.zeros([n_param, n_samples, physics.n_t, model.n_z]));

    # For each testing parameter, cycle through the samples of the coefficients for that 
    # combination of parameter values. For each set of coefficients, solve the corresponding latent 
    # dynamics forward in time and store the resulting frames in Zis.
    n_t : int = physics.n_t;
    n_z : int = latent_dynamics.n_z; 
    for i in range(n_param):
        # Fetch the initial conditions when we use the i'th combination of parameter values.
        # Reshape each element of the IC to have shape (1, n_z), which is what simulate expects
        ith_ICs                 : list[numpy.ndarray]   = Z0[i];
        for d in range(n_IC):
            ith_ICs[d] = ith_ICs[d].reshape(1, -1);
        coef_samples_ith_param  : numpy.ndarray         = coef_samples[i];

        for j in range(n_samples):
            # Fetch the j'th sample of the coefficients when we use the i'th combination of 
            # parameter values.
            jth_coef_sample_ith_param   : numpy.ndarray = coef_samples_ith_param[j, :];
        
            # Generate the latent trajectory when we use this set of coefficients.
            Zij : list[numpy.ndarray] = latent_dynamics.simulate(coefs = jth_coef_sample_ith_param, IC = ith_ICs, times = physics.t_grid);
        
            # Now store the results in Zis. 
            for d in range(n_IC):
                Zis[d][i, j, :, :]  = Zij[d].reshape(n_t, n_z);

    # All done!
    return Zis;



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
        argmax_{i}[ STD[ { Decoder(LatentStates[i][0][p, q, :])_k : p \in {1, 2, ... , n_samples(i)} } ]
                    |   k \in {1, 2, ... , n_{FOM}},
                        i \in {1, 2, ... , n_param},
                        q \in {1, 2, ... , n_t(i)} ]
    

    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    model: The model. We assume the solved dynamics (whose frames are stored in Zis) 
    take place in the model's latent space. We use this to decode the solution frames.

    LatentStates: An n_param element list whose i'th element is an n_IC element list whose j'th
    element is a 3d tensor of shape (n_samples(i), n_t(i), n_z) whose p, q, r element holds the 
    r'th component of the j'th component of the latent solution at the q'th time step when we solve 
    the latent dynamics using the p'th set of coefficients we got by sampling the posterior 
    distribution for the i'th combination of parameter values. 


    -----------------------------------------------------------------------------------------------
    Returns:
    -----------------------------------------------------------------------------------------------

    An integer. The index of the testing parameter that gives the largest standard deviation. 
    See the description above for details.
    """
    
    # Run checks.
    assert(isinstance(LatentStates,         list));
    assert(isinstance(LatentStates[0],      list));
    assert(isinstance(LatentStates[0][0],   numpy.ndarray));
    assert(len(LatentStates[0][0])          == 3);

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
            X_Pred_i        : numpy.ndarray = numpy.empty((n_samples_i, n_t_i, n_z), dtype = numpy.float32);
            for j in range(n_samples_i):
                X_Pred_i[j, :, :] = model.Decode(Z_i[j, :, :]).detach().numpy();

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
            D_Pred_i    : numpy.ndarray = numpy.empty((n_samples_i, n_t_i, n_z), dtype = numpy.float32);
            for j in range(n_samples_i):
                D_Pred_ij, _ = model.Decode(Latent_Displacement   = Z_D_i, Latent_Velocity    = Z_V_i);
                D_Pred_i[j, :, :] = D_Pred_ij.detach().numpy();

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