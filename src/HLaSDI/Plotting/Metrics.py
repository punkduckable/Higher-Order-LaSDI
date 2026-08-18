# -------------------------------------------------------------------------------------------------
# Import and Setup
# -------------------------------------------------------------------------------------------------

import  logging;

import  torch;
import  numpy;

from    HLaSDI.EncoderDecoder                  import  EncoderDecoder;
from    HLaSDI.Physics                         import  Physics;
from    HLaSDI.LatentDynamics                  import  LatentDynamics, InterpolatableLatentDynamics;
from    HLaSDI.ParameterSpace                  import  ParameterSpace;
from    HLaSDI.Trainer                         import  Trainer;
from    HLaSDI.Rollouts                        import  Sample_Rollouts, Mean_Rollout;


# Set up the logger
LOGGER : logging.Logger = logging.getLogger(__name__);


# -------------------------------------------------------------------------------------------------
# Heatmap data
# -------------------------------------------------------------------------------------------------

def Generate_Heatmap_Data(  encoder_decoder : EncoderDecoder,
                            physics         : Physics,
                            param_space     : ParameterSpace,
                            latent_dynamics : LatentDynamics,
                            t_Test          : list[torch.Tensor],
                            U_Test          : list[list[torch.Tensor]],
                            trainer         : Trainer,
                            n_samples       : int       = 20) -> tuple[numpy.ndarray, numpy.ndarray, list[list[numpy.ndarray]], list[list[numpy.ndarray]], numpy.ndarray, numpy.ndarray]:
    r"""
    This function computes the relative error and STD between the FOM solution and its 
    prediction when we rollout the FOM solution using the the ICs and mean of the posterior 
    distribution of the coefficients for each combination of parameter values.
    
    To do this, we first sample the posterior distribution of the coefficients for each combination 
    of parameter values and solve the latent dynamics forward in time using each sample (as well as
    the mean of the posterior distribution). We then decode the latent trajectories to get a set of 
    FOM solutions. We then compute a *normalized absolute error* between the mean predicted solution 
    and the true solution for each frame of each derivative of the FOM solution for each combination 
    of parameter values. The normalization is a single standard deviation per (parameter, derivative)
    computed over all time steps and spatial nodes of the true trajectory. We then find the maximum 
    relative error (across the frames and components) for each derivative for each combination of 
    parameter values. 
    
    We also compute the STD (across the samples) of each frame of each derivative of the FOM 
    solution for each combination of the parameter values. We then find the maximum STD (across 
    the frames and components) for each derivative for each combination of parameter values.

    Note: If X_1, ... , X_M \in \mathbb{R}^N are vectors then the STD of this collection is the 
    vector whose i'th component holds the (sample) STD of {X_1[i], ... , X_M[i]}.
    
    Note: The implementation below does **not** use an l^\infty normalization. Instead, for each
    (parameter, derivative) it computes
        mean_x |u_pred(t_k, x) - u_true(t_k, x)| / std(u_true)
    where std(u_true) is computed over the full true trajectory (all times and spatial nodes).


    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    encoder_decoder : EncoderDecoder
        For each combinations of parameters, we find the encoder_decoder's latent dynamics for that 
        combination and solve them forward in time. 

    physics : Physics
        A Physics object that we use to fetch the initial condition for each combination of 
        parameter values.

    param_space : ParameterSpace
        A ParameterSpace object which holds the testing parameters.
    
    latent_dynamics : LatentDynamics
        The LatentDynamics object we use to generate the latent space data. For each combination 
        of parameter values, we fetch the corresponding coefficients to define the latent dynamics.
    
    t_Test : list[torch.Tensor], len = n_test
        i'th element is a 1d numpy.ndarray object of length n_t(i) whose j'th element holds the 
        value of the j'th time value at which we solve the latent dynamics for the i'th combination
        of parameter values.

    U_Test : list[list[torch.Tensor]], len = n_test
        i'th element is an n_IC element list whose j'th element is a torch.Tensor of shape 
        (n_t(i), ...) whose k, ... slice holds the k'th frame of the j'th time derivative of the
        FOM model when we use the i'th combination of parameter values to define the FOM model.

    trainer : Trainer
        A Trainer object.

    n_samples : int
        If the `LD` is Interpolatable, this is the number of coefficient samples we draw from 
        each coefficient posterior. Otherwise, this does nothing. Each sample gives us 
        a set of coefficients which we can use to define the latent dynamics that we then solve 
        forward in time. 


    
    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    max_Rel_Error, max_STD, Rel_Error, STD, coef_means, coef_stds

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
    
    coef_means : numpy.ndarray, shape = (n_Test, n_Coef) | None
        i, j element holds the mean of the posterior distribution for the j'th coefficient 
        evaluated at the i'th combination of testing parameters. None if the LD is not 
        Interpolatable.

    coef_stds : numpy.ndarray, shape = (n_Test, n_Coef) | None
        i, j element holds the stds of the posterior distribution for the j'th coefficient 
        evaluated at the i'th combination of testing parameters. None if the LD is not 
        Interpolatable.
    """ 

    # Run checks
    assert isinstance(t_Test,           list),      "type(t_Test) = %s, expected list" % (type(t_Test));
    assert isinstance(U_Test,           list),      "type(U_Test) = %s, expected list" % (type(U_Test));
    assert isinstance(n_samples,        int),       "type(n_samples) = %s, expected int" % (type(n_samples));
    assert len(t_Test)  == len(U_Test),             "len(t_Test) = %d, len(U_Test) %d" % (len(t_Test), len(U_Test));
    
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
    
    # Evaluate posterior means/stds if the LD is interpolatable.
    if isinstance(latent_dynamics, InterpolatableLatentDynamics):
        coef_means_native : list[dict[str, torch.Tensor]] = [latent_dynamics.interpolator.mean(param_test[i, :]) for i in range(n_Test)];
        coef_stds_native  : list[dict[str, torch.Tensor]] = [latent_dynamics.interpolator.std(param_test[i, :])  for i in range(n_Test)];
        coef_means = flatten_coefficients(coef_means_native);
        coef_stds  = flatten_coefficients(coef_stds_native);
    else:
        coef_means = None
        coef_stds = None;


    # ---------------------------------------------------------------------------------------------
    # Compute ROM ICs.

    # First, fetch the FOM ICs.
    FOM_IC : list[list[numpy.ndarray]] = []; # len = n_Test; i'th element has n_IC elements
    has_norm : bool = (trainer is not None) and hasattr(trainer, "has_normalization") and trainer.has_normalization();
    for i in range(n_Test):
        # Get the ICs for the i'th combination of parameter values.
        ith_FOM_IC : list[numpy.ndarray] = physics.initial_condition(param_test[i]);
        assert isinstance(ith_FOM_IC, list), "type(ith_FOM_IC) = %s, expected list" % str(type(ith_FOM_IC));
        assert len(ith_FOM_IC) == encoder_decoder.n_IC, "len(ith_FOM_IC) = %d, expected %d (=encoder_decoder.n_IC)" % (len(ith_FOM_IC), encoder_decoder.n_IC);

        # Apply normalization if available.    
        if has_norm:
            Normalized_FOM_IC : list[numpy.ndarray] = [];
            for k in range(len(ith_FOM_IC)):
                Normalized_FOM_IC.append(trainer.normalize(ith_FOM_IC[k], k));
            ith_FOM_IC = Normalized_FOM_IC;

        # All done!
        FOM_IC.append(ith_FOM_IC);

    # Now encode them.
    ROM_IC : list[list[numpy.ndarray]] = encoder_decoder.latent_initial_conditions(FOM_IC);

    # ---------------------------------------------------------------------------------------------
    # Draw n_samples samples of the posterior distribution.

    # For each combination of parameter values in the testing set, sample the latent coefficients 
    # and solve the latent dynamics forward in time. 
    LOGGER.info("Generating latent dynamics trajectories for %d samples of the coefficients for %d combinations of testing parameter" % (n_samples, n_Test));
    Zis_samples     : list[list[numpy.ndarray]] = Sample_Rollouts(
                                                    ROM_IC          = ROM_IC, 
                                                    latent_dynamics = latent_dynamics, 
                                                    param_grid      = param_test,
                                                    t_Grid          = t_Test, 
                                                    n_samples       = n_samples);    # len = n_test. i'th element is an n_IC element list whose j'th element has shape (n_t(i), n_samples, n_z)

    LOGGER.info("Generating latent dynamics trajectories using posterior distribution means for %d combinations of testing parameter" % (n_Test));
    Zis_mean        : list[list[numpy.ndarray]] = Mean_Rollout(
                                                    ROM_IC          = ROM_IC,
                                                    latent_dynamics = latent_dynamics, 
                                                    param_grid      = param_test, 
                                                    t_Grid          = t_Test);               # len = n_test. i'th element is an n_IC element list whose j'th element has shape (n_t(i), n_z)
        

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
            STD_i.append(numpy.zeros_like(U_Test[i][j].detach().cpu().numpy()));
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

    for i in range(n_Test):
        # -------------------------------------------------------------------------------------
        # Relative Error

        # Convert latent trajectories to Tensors
        Zis_mean_i : list[torch.Tensor] = [];
        for j in range(n_IC):
            Zis_mean_i.append(torch.Tensor(Zis_mean[i][j]));

        # Decode the mean latent trajectories for each combination of parameter values.
        U_Pred_Mean_i       : list[torch.Tensor]    = list(encoder_decoder.Decode(*Zis_mean_i));

        # Fetch the corresponding test predictions.
        U_Test_i            : list[torch.Tensor]    = U_Test[i];
        
        # Set up a list to hold the STDs of the FOM solution.
        U_Test_i_std        : list[float]           = [];

        # Convert to numpy and denormalize. Also populate U_Test_i_std.
        U_Test_i_np         : list[numpy.ndarray]   = [];
        U_Pred_Mean_i_np    : list[numpy.ndarray]   = [];
        for j in range(n_IC):
            U_Pred_Mean_i_np.append(U_Pred_Mean_i[j].detach().numpy())  # (n_t_i, physics.Frame_Shape)
            U_Test_i_np.append(U_Test_i[j].detach().numpy())            # (n_t_i, physics.Frame_Shape)
            
            if use_denorm:
                U_Pred_Mean_i_np[j] = trainer.denormalize_np(U_Pred_Mean_i_np[j], j);
                U_Test_i_np[j]      = trainer.denormalize_np(U_Test_i_np[j], j);
        
            U_Test_i_std.append(numpy.std(U_Test_i_np[j]))

        # For each frame, compute the relative error between the true and predicted FOM solutions.
        # We normalize the error by the std of the true solution.
        n_t_i : int = t_Test[i].shape[0];
        for j in range(n_IC):
            for k in range(n_t_i):
                Rel_Error[i][j][k] = numpy.mean(numpy.abs(U_Pred_Mean_i_np[j][k, ...] - U_Test_i_np[j][k, ...]))/U_Test_i_std[j];
        
            # Now compute the corresponding element of max_Rel_Error
            max_Rel_Error[i, j] = Rel_Error[i][j].max();
    

        # -------------------------------------------------------------------------------------
        # Standard Deviation

        # Set up an array to hold the decoding of latent trajectory.
        FOM_Frame_Shape : list[int]             = physics.Frame_Shape;
        U_Pred_i        : list[numpy.ndarray]   = [];
        for j in range(n_IC):
            U_Pred_i.append(numpy.empty([n_t_i, n_samples] + FOM_Frame_Shape, dtype = numpy.float32));

        # Decode the latent trajectory for each sample.
        for j in range(n_samples):
            Zis_sample_ij: list[torch.Tensor] = [];
            for k in range(n_IC):
                Zis_sample_ij.append(torch.Tensor(Zis_samples[i][k][:, j, :]));
            U_Pred_ij   : tuple[torch.Tensor]     = encoder_decoder.Decode(*Zis_sample_ij);
            
            # Detach, convert to numpy, and store in U_Pred_i.
            for k in range(n_IC):
                U_Pred_ijk_np = U_Pred_ij[k].detach().numpy();
                U_Pred_i[k][:, j, ...]             = U_Pred_ijk_np;
    
        # Compute the STD across the sample axis.
        for j in range(n_IC):
            STD_ij          = numpy.std(U_Pred_i[j], axis = 1);
            STD[i][j]       = trainer.scale_std_np(STD_ij, j) if use_denorm else STD_ij;
        
            # Compute max STD using robust metric: average across spatial dimensions, then max over time
            # This prevents single outlier nodes from dominating the metric.
            STD_ij_spatial_avg : numpy.ndarray = STD[i][j].mean(axis = tuple(range(1, STD[i][j].ndim)));  # Average over spatial dims
            max_STD[i, j]      : numpy.float32 = STD_ij_spatial_avg.max();  # Max over time only
    

    # All done!
    return max_Rel_Error, max_STD, Rel_Error, STD, coef_means, coef_stds;


# -------------------------------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------------------------------

def flatten_coefficients(coefs : dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]) -> numpy.ndarray:
    r"""
    Flatten a dictionary (or list of dictionaries) of tensors into one coefficient matrix. This 
    is useful when trying to analyze coefficient dictionaries for interpolation style latent 
    dynamics classes. 

    For each coefficient dictionary, this method loops through the dictionary items in insertion
    order, flattens each tensor, and concatenates those flattened arrays into one row.


    -------------------------------------------------------------------------------------------
    Arguments
    -------------------------------------------------------------------------------------------

    coefs : dict[str, torch.Tensor] or list[dict[str, torch.Tensor]]
        One coefficient dictionary or a list of coefficient dictionaries.


    -------------------------------------------------------------------------------------------
    Returns
    -------------------------------------------------------------------------------------------

    coef_matrix : numpy.ndarray, shape = (n_param, n_coefs)
        Flattened coefficient matrix. The i'th row contains all flattened coefficient tensors
        from the i'th native coefficient dictionary, concatenated in dictionary item order.
    """

    # Normalize the input so the single-coefficient and multi-coefficient cases share the same
    # validation/flattening logic.
    coefs_list = [coefs] if isinstance(coefs, dict) else coefs;
    assert isinstance(coefs_list, list), "coefs must be a dict or a list of dicts";
    assert len(coefs_list) > 0, "coefs must be non-empty";

    rows : list[numpy.ndarray] = [];
    for coef_dict in coefs_list:
        assert isinstance(coef_dict, dict), "Each coefficient set must be a dictionary";
        assert len(coef_dict) > 0, "Coefficient dictionaries must be non-empty";

        parts : list[numpy.ndarray] = [];
        for name, tensor in coef_dict.items():
            assert isinstance(name, str), "Coefficient names must be strings";
            assert isinstance(tensor, torch.Tensor), "Coefficient %s must be a torch.Tensor" % name;
            parts.append(tensor.detach().cpu().numpy().reshape(-1));

        rows.append(numpy.concatenate(parts, axis = 0).reshape(1, -1));

    return numpy.concatenate(rows, axis = 0);
