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
                param_grid      : numpy.ndarray):
    """
    This function simulates the latent dynamics for a collection of testing parameters by using
    the mean of the posterior distribution for each coefficient's posterior distribution. 
    Specifically, for each parameter combination, we determine the mean of the posterior 
    distribution for each coefficient. We then use this mean to simulate the latent dynamics 
    forward in time (starting from the latent encoding of the FOM initial condition for that 
    combination of coefficients).

    
    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    model: The actual model object that we use to map the ICs into the latent space.

    physics: A "Physics" object that stores the datasets for each parameter combination. 
    
    latent_dynamics: A LatentDynamics object which describes how we specify the dynamics in the
    model's latent space.    

    gp_list: a list of trained GP regressor objects. The number of elements in this list should 
    match the number of columns in param_grid. The i'th element of this list is a GP regressor 
    object that predicts the i'th coefficient. 

    param_grid: A 2d numpy.ndarray object of shape (number of parameter combination, number of 
    parameters). The i,j element of this array holds the value of the j'th parameter in the i'th 
    combination of parameters. 


    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------
    
    A 3d numpy ndarray whose i, j, k element holds the k'th component of the j'th time step of 
    the solution to the latent dynamics when we use the latent encoding of the initial condition 
    from the i'th combination of parameter values
    """

    # The param grid needs to be two dimensional, with the first axis corresponding to which 
    # instance of the parameter values we are using. If there is only one parameter, it may be 1d. 
    # We can fix that by adding on an axis with size 1. 
    if (param_grid.ndim == 1):
        param_grid = param_grid.reshape(1, -1);

    # Now fetch the number of combinations of parameter values.
    n_param : int = param_grid.shape[0];

    # For each parameter in param_grid, fetch the corresponding initial condition and then encode
    # it. This gives us a list whose i'th element holds the encoding of the i'th initial condition.
    Z0      : list[list[numpy.ndarray]] = model.latent_initial_conditions(param_grid, physics);

    # Evaluate each GP at each combination of parameter values. This returns two arrays, the 
    # first of which is a 2d array whose i,j element specifies the mean of the posterior 
    # distribution for the j'th coefficient at the i'th combination of parameter values.
    pred_mean, _ = eval_gp(gp_list, param_grid);

    # For each testing parameter, cycle through the mean value of each coefficient from each 
    # posterior distribution. For each set of coefficients (combination of parameter values), solve
    # the latent dynamics forward in time (starting from the corresponding IC value) and store the
    # resulting solution frames in Zis, a 3d array whose i, j, k element holds the k'th component 
    # of the j'th time step fo the latent solution when we use the coefficients from the posterior 
    # distribution for the i'th combination of parameter values.
    nz  : int           = model.n_z;
    Zis : numpy.ndarray = numpy.zeros([n_param, physics.nt, nz]);

    for i in range(n_param):
        Zis[i] = latent_dynamics.simulate(pred_mean[i], Z0[i], physics.t_grid);

    # All done!
    return Zis;



def sample_roms(model           : torch.nn.Module, 
                physics         : Physics, 
                latent_dynamics : LatentDynamics, 
                gp_list         : list[GaussianProcessRegressor], 
                param_grid      : numpy.ndarray, 
                n_samples       : int) ->           numpy.ndarray:
    """
    This function samples the latent coefficients, solves the corresponding latent dynamics, and 
    then returns the resulting latent solutions. 
    
    Specifically, for each combination of parameter values in the param_grid, we draw n_samples 
    samples of the latent coefficients (from the coefficient posterior distributions evaluated at 
    that parameter value). This gives us a set of n_samples latent dynamics coefficients. For each 
    set of coefficients, we solve the corresponding latent dynamics forward in time and store the 
    resulting solution frames. We do this for each sample and each combination of parameter values,
    resulting in an (n_param, n_sample, n_t, n_z) array of solution frames, which is what we 
    return.

    
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
    
    A numpy.ndarray of size [n_test, n_samples, physics.nt, model.n_z]. The i, j, k, l 
    element holds the l'th component of the k'th frame of the solution to the latent dynamics when 
    we use the j'th sample of latent coefficients drawn from the posterior distribution for the 
    i'th combination of parameter values (i'th row of param_grid).
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

    # For each testing parameter, cycle through the samples of the coefficients for that 
    # combination of parameter values. For each set of coefficients, solve the corresponding latent 
    # dynamics forward in time and store the resulting frames in Zis. This is a 4d array whose i, 
    # j, k, l element holds the l'th component of the k'th frame of the solution to the latent 
    # dynamics when we use the j'th sample of latent coefficients drawn from the posterior 
    # distribution for the i'th combination of parameter values.
    Zis = numpy.zeros([n_param, n_samples, physics.nt, model.n_z]);
    for i, Zi in enumerate(Zis):
        z_ic = Z0[i];
        for j, coef_sample in enumerate(coef_samples[i]):
            Zi[j] = latent_dynamics.simulate(coef_sample, z_ic, physics.t_grid);

    # All done!
    return Zis;



def get_FOM_max_std(model : torch.nn.Module, LatentStates : list[numpy.ndarray]) -> int:
    r"""
    Finds which parameter combination gives the maximum standard deviation of some component 
    of the FOM solution at some time step across the corresponding samples of the coefficients
    in the latent space. 

    We assume that LatentStates is a list of 4d tensors of shape (n_test, n_samples, n_t, n_z). 
    Here, n_test is the number of testing combinations, n_samples is the number of samples of the 
    latent coefficients we draw per parameter combination, n_t is the number of time steps we 
    generate in the latent space per set of coefficients, and n_z is the dimension of the latent 
    space.

    For each time step and parameter combination, we get a set of latent frames. We map that 
    set to a set of FOM frames and then find the STD of each component of those FOM frames 
    across the samples. This give us a number. We find the corresponding number for each time 
    step and combination of parameter values and then return the parameter combination that 
    gives the biggest number (for some time step).

    Let i \in {1, 2, ... , n_test} and k \in {1, 2, ... , n_t}. For each j, we map the k'th frame
    of the j'th solution trajectory for the i'th parameter combination 
    (LatentStates[:][i, j, k, :]) to a FOM frame. We do this for each j (the set of samples), which 
    gives us a collection of n_sample FOM frames, representing samples of the distribution of FOM 
    frames at the k'th time step when we use the posterior distribution for the i'th set of 
    parameters. For each l \in {1, 2, ... , n_FOM}, we compute the STD of the set of l'th 
    components of these n_sample FOM frames. We do this for each i and k and then figure out which 
    i, k, l combination gives the largest STD. We return the corresponding i index. 
    

    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    model: The model. We assume the solved dynamics (whose frames are stored in Zis) 
    take place in the model's latent space. We use this to decode the solution frames.

    LatentStates: A list of 4d numpy array of shape (n_test, n_samples, n_t, n_z). The i, j, k, l
    element of the d'th list item holds the l'th component of the k'th frame of the d'th time 
    derivative of the solution to the latent dynamics when we use the j'th sample of latent 
    coefficients drawn from the posterior distribution for the i'th testing parameter.

    

    -----------------------------------------------------------------------------------------------
    Returns:
    -----------------------------------------------------------------------------------------------

    An integer. The index of the testing parameter that gives the largest standard deviation. 
    Specifically, for each testing parameter, we compute the STD of each component of the FOM 
    solution at each frame generated by samples from the posterior coefficient distribution for 
    that parameter. We compute the maximum of these STDs and pair that number with the parameter. 
    We then return the index of the parameter whose corresponding maximum std (the number we pair
    with it) is greatest.
    """

    max_std     : float     = 0.0;
    m_index     : int       = 0;
    
    if(isinstance(model, Autoencoder)):
        assert(len(LatentStates) == 1);
        Latent_Displacements    : numpy.ndarray = LatentStates[0];
        n_Test                  : int           = Latent_Displacements.shape[0];

        for m in range(n_Test):
            # Zi is a 3d tensor of shape (n_samples, n_t, n_z), where n_samples is the number of 
            # samples of the posterior distribution per parameter, n_t is the number of time steps 
            # in the latent dynamics solution, and n_z is the dimension of the latent space. The
            # i,j,k element of Zi is the k'th component of the j'th frame of the solution to the 
            # latent dynamics when the latent dynamics uses the i'th set of sampled parameter 
            # values.
            Z_m             : torch.Tensor  = torch.Tensor(Latent_Displacements[m, ...]);

            # Now decode the frames.
            X_pred_m        : numpy.ndarray = model.Decode(Z_m).detach().numpy();

            # Compute the standard deviation across the sample axis. This gives us an array of shape 
            # (n_t, n_FOM) whose i,j element holds the (sample) standard deviation of the j'th component 
            # of the i'th frame of the FOM solution. In this case, the sample distribution consists of 
            # the set of j'th components of i'th frames of FOM solutions (one for each sample of the 
            # coefficient posterior distributions).
            X_pred_m_std    : numpy.ndarray = X_pred_m.std(0);

            # Now compute the maximum standard deviation across frames/FOM components.
            max_std_m       : numpy.float32 = X_pred_m_std.max();

            # If this is bigger than the biggest std we have seen so far, update the maximum.
            if max_std_m > max_std:
                m_index : int   = m;
                max_std : float = max_std_m;

        # Report the index of the testing parameter that gave the largest maximum std.
        return m_index


    elif(isinstance(model, Autoencoder_Pair)):
        assert(len(LatentStates)    == 2);
        Latent_Displacements    : numpy.ndarray     = LatentStates[0];
        Latent_Velocities       : numpy.ndarray     = LatentStates[1];
        n_Test                  : int               = Latent_Displacements.shape[0];

        # Cycle through the testing parameters, components of the derivative.
        for m in range(n_Test):
            # Build the latent state for the i'th testing sample. 
            # In this case, the latent state is a 2 element list of 3d tensor of shape (n_samples, 
            # n_t, n_z), where n_samples is the number of samples of the posterior distribution 
            # per parameter, n_t is the number of time steps in the latent dynamics solution, and 
            # n_z is the dimension of the latent space. The i,j,k element of the d'th list item is 
            # the k'th component of the j'th frame of the d'th derivative of the solution to the 
            # latent dynamics when the latent dynamics uses the i'th set of sampled parameter 
            # values.
            X_pred_m, _  = model.Decode(Latent_Displacement = torch.Tensor(Latent_Displacements[m, ...]), 
                                        Latent_Velocity     = torch.Tensor(Latent_Velocities[m, ...]));
            X_pred_m            = X_pred_m.detach().numpy();

            # Compute the standard deviation across the sample axis. This gives us an array of 
            # shape (n_t, n_FOM) whose i,j element holds the (sample) standard deviation of the 
            # j'th component of the i'th frame of the FOM solution. In this case, the sample 
            # distribution consists of the set of j'th components of i'th frames of FOM solutions 
            # (one for each sample of the coefficient posterior distributions).
            X_pred_m_std    : numpy.ndarray = X_pred_m.std(0);

            # Now compute the maximum standard deviation across frames/FOM components.
            max_std_m       : numpy.float32 = X_pred_m_std.max();

            # If this is bigger than the biggest std we have seen so far, update the maximum.
            if max_std_m > max_std:
                m_index : int   = m;
                max_std : float = max_std_m;

        # Report the index of the testing parameter that gave the largest maximum std.
        return m_index;
    
    
    else:
        raise ValueError("Invalid model type!");