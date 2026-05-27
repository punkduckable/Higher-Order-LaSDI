# -------------------------------------------------------------------------------------------------
# Import and Setup
# -------------------------------------------------------------------------------------------------

import os;
import sys;
from pathlib import Path;

# Resolve paths relative to the project root (Higher-Order-LaSDI/), independent of CWD.
Figures_Path        : str   = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "Figures");
Physics_Path        : str   = os.path.abspath(os.path.join(os.path.dirname(__file__), "Physics"));
LD_Path             : str   = os.path.abspath(os.path.join(os.path.dirname(__file__), "LatentDynamics"));
EncoderDecoder_Path : str   = os.path.abspath(os.path.join(os.path.dirname(__file__), "EncoderDecoder"));
Interpolate_Path    : str   = os.path.abspath(os.path.join(os.path.dirname(__file__), "Interpolate"));
Utilities_Path      : str   = os.path.abspath(os.path.join(os.path.dirname(__file__), "Utilities"));
sys.path.append(Utilities_Path);
sys.path.append(Physics_Path);
sys.path.append(LD_Path);
sys.path.append(Interpolate_Path);
sys.path.append(EncoderDecoder_Path);

import  logging;

import  torch;
import  numpy;
import  matplotlib.pyplot               as      plt;
import  matplotlib                      as      mpl;
from    matplotlib.figure               import  Figure;
from    matplotlib.backends.backend_agg import FigureCanvasAgg;

from    EncoderDecoder                  import  EncoderDecoder;
from    Physics                         import  Physics;
from    LatentDynamics                  import  LatentDynamics;
from    ParameterSpace                  import  ParameterSpace;
from    Trainer                         import  Trainer;
from    Rollouts                        import  Sample_Rollouts, Mean_Rollout;
from    Interpolate                     import  Interpolate;


# Set up the logger
LOGGER : logging.Logger = logging.getLogger(__name__);

# Set plot settings. 
mpl.rcParams['lines.linewidth'] = 2;
mpl.rcParams['axes.linewidth']  = 1.5;
mpl.rcParams['axes.edgecolor']  = "black";
mpl.rcParams['grid.color']      = "gray";
mpl.rcParams['grid.linestyle']  = "dotted";
mpl.rcParams['grid.linewidth']  = .67;
mpl.rcParams['xtick.labelsize'] = 10;
mpl.rcParams['ytick.labelsize'] = 10;
mpl.rcParams['axes.labelsize']  = 11;
mpl.rcParams['axes.titlesize']  = 11;
mpl.rcParams['xtick.direction'] = 'in';
mpl.rcParams['ytick.direction'] = 'in';



# -------------------------------------------------------------------------------------------------
# Latent Trajectory Plots
# -------------------------------------------------------------------------------------------------

def Plot_Latent_Trajectories(physics         : Physics,
                             encoder_decoder : EncoderDecoder,
                             latent_dynamics : LatentDynamics,
                             interpolator    : Interpolate,
                             param_grid      : numpy.ndarray,
                             U_True          : list[list[torch.Tensor]],
                             t_Grid          : list[torch.Tensor],
                             file_prefix     : str,
                             trainer         = None,
                             n_samples       : int           = 20,
                             figsize         : tuple[int]    = (15, 13)) -> None:
    """
    This function plots the latent trajectories of the latent dynamics model for a combination of 
    parameter values. Specifically, we fetch the FOM IC for the given parameter values, encode then, 
    and then use the Interpolate object to sample native latent-dynamics coefficient dictionaries,
    solve and plot each resulting dynamical solution, and then plot the encodings of the FOM
    trajectory. 


    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    physics : Physics
        A Physics object which acts as a wrapper for the FOM. We use this to get the FOM IC.

    encoder_decoder : EncoderDecoder
        The EncoderDecoder we use to encode the FOM IC and the FOM trajectories.

    latent_dynamics : LatentDynamics
        The LatentDynamics model we use to simulate the latent dynamics forward in time.

    interpolator : Interpolate
        An Interpolate object that returns native coefficient dictionaries via `sample(...)`,
        `mean(...)`, and `std(...)`. We use it to draw coefficient samples for latent rollouts.

    param_grid : numpy.ndarray, shape = (n_param, n_p)
        A numpy array whose rows holds the parameter values whose latent dynamics we want to plot.
        We assume that the i'th row hodls the i'th combination of parameter values.

    U_True : list[list[torch.Tensor]], len = n_param
        The i'th element is an n_IC element list whose j'th element is a torch.Tensor of shape 
        (n_t_i,) + physics.Frame_Shape whose k'th row holds the j'th time derivative of the FOM 
        solution for the i'th combination of prameter values at t_Grid[i][k].

    t_Grid : list[torch.Tensor], len = n_param
        The i'th element is a 1D torch.Tensor object which holds the time grid for the i'th 
        combination of parameter values. We assume that this tensor has shape (n_t_i,).

    file_prefix : str
        The prefix of the file name we use to save the plots. Usually the name of the FOM model.
    
    n_samples : int
        The number of coefficient samples we want to draw from `interpolator` for each combination
        of parameter values.
        
    figsize : tuple[int], len = 2
        A two element tuple specifying the size of the overall figure size. 


    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    Nothing!
    """ 

    # Checks
    assert isinstance(physics, Physics),                "type(physics) = %s" % type(physics);
    assert isinstance(encoder_decoder, EncoderDecoder), "type(encoder_decoder) = %s" % type(EncoderDecoder);
    assert isinstance(latent_dynamics, LatentDynamics), "type(latent_dynamics) = %s" % type(latent_dynamics);
    assert isinstance(interpolator, Interpolate), "type(interpolator) = %s" % type(interpolator);

    assert isinstance(param_grid, numpy.ndarray),        "type(param_grid) = %s" % type(param_grid);
    assert param_grid.ndim     == 2,                     "param_grid.ndim = %d != 2" % param_grid.ndim;
    assert param_grid.shape[1] == physics.n_p,           "param_grid.shape = %s != physics.n_p = %d" % (str(param_grid.shape), physics.n_p);
    n_param : int = param_grid.shape[0];

    assert isinstance(n_samples, int),                  "type(n_samples) = %s" % type(n_samples);
    assert isinstance(U_True, list),                    "type(U_True) = %s" % type(U_True);
    assert isinstance(t_Grid, list),                    "type(t_Grid) = %s" % type(t_Grid);
    assert len(t_Grid)      == n_param,                 "len(t_Grid) = %d != n_param = %d" % (len(t_Grid), n_param);
    assert len(U_True)      == n_param,                 "len(U_True) = %d != n_param = %d" % (len(U_True), n_param);
    for i in range(n_param):
        assert isinstance(U_True[i], list),             "type(U_True[%d]) = %s" % (i, type(U_True[i]));
        assert len(U_True[i])  == physics.n_IC,         "len(U_True[%d]) = %d != physics.n_IC = %d" % (i, len(U_True[i]), physics.n_IC);
        assert isinstance(t_Grid[i], torch.Tensor),     "type(t_Grid[%d]) = %s" % (i, type(t_Grid[i]));
        assert t_Grid[i].ndim     == 1,                 "t_Grid[%d].ndim = %d != 1" % (i, t_Grid[i].ndim);
        n_t_i : int = t_Grid[i].shape[0];   # number of time steps for the i'th combination of parameter values.
        for j in range(physics.n_IC):
            assert isinstance(U_True[i][j], torch.Tensor),  "type(U_True[%d][%d]) = %s" % (i, j, type(U_True[i][j]));
            assert U_True[i][j].ndim >= 2,                  "U_True[%d][%d].ndim = %d < 2" % (i, j, U_True[i][j].ndim);
            assert U_True[i][j].shape[0]    == n_t_i,       "U_True[%d][%d].shape[0] = %d != n_t_i = %d" % (i, j, U_True[i][j].shape[0], n_t_i);

    assert isinstance(figsize, tuple), "type(figsize) = %s" % type(figsize);
    assert len(figsize)     == 2,       "len(figsize) = %d != 2" % len(figsize);


    # ---------------------------------------------------------------------------------------------
    # Generate the Latent Trajectories.

    # First generate the latent trajectories. This is a an n_param element list whose i'th element
    # is an n_IC element list whose j'th element is a 3d array of shape (n_t(i), n_samples, n_z). 
    # Here, n_param is the number of combinations of parameter values.
    LOGGER.info("Solving the latent dynamics using %d samples of the posterior distributions for %d combinations of parameter values" % (n_samples, n_param));
    Predicted_Latent_Trajectories : list[list[numpy.ndarray]] = Sample_Rollouts( 
                                                                    encoder_decoder = encoder_decoder, 
                                                                    physics         = physics, 
                                                                    latent_dynamics = latent_dynamics, 
                                                                    interpolator    = interpolator, 
                                                                    param_grid      = param_grid,
                                                                    t_Grid          = t_Grid,
                                                                    n_samples       = n_samples,
                                                                    trainer         = trainer);
    
    # Now encode the FOM trajectories. Store these in an n_param element list whose i'th element
    # is an n_IC element list whose j'th element is a numpy array of shape (n_t(i), n_z) holding
    # the encoding of the j'th FOM trajectory for the i'th combination of parameter values.
    True_Latent_Trajectories : list[list[numpy.ndarray]] = [];          # len = n_param
    for i in range(n_param):
        ith_True_Latent_Trajectories : list[numpy.ndarray] = [];
        ith_Encoding : tuple[torch.Tensor] = encoder_decoder.Encode(*U_True[i]);
        for j in range(len(ith_Encoding)):
                ith_True_Latent_Trajectories.append(ith_Encoding[j].detach().numpy());
        
        True_Latent_Trajectories.append(ith_True_Latent_Trajectories);
        

    # ---------------------------------------------------------------------------------------------
    # Make the plots!

    # Set up the subplots.
    LOGGER.info("Making latent trajectory plots for %d combinations of parameter values" % n_param);
    for i in range(n_param):
        # Time grid for this parameter combination (used as x-axis).
        t_np: numpy.ndarray = t_Grid[i].detach().cpu().numpy();
        for j in range(physics.n_IC):
            # Set up the plot for this combination of parameter values.
            plt.figure(figsize = figsize);

            # Plot the predicted latent trajectories
            for s in range(n_samples):
                for k in range(latent_dynamics.n_z):
                    plt.plot(t_np, Predicted_Latent_Trajectories[i][j][:, s, k], 'C' + str(k), linewidth = 1, alpha = 0.3);

            # Plot each component of the latent trajectories
            for k in range(latent_dynamics.n_z):
                plt.plot(t_np, True_Latent_Trajectories[i][j][:, k], 'C' + str(k), linewidth = 3, alpha = 0.75);
            
            # Determine the title and save file name.
            if(j == 0):
                title          : str = "Z(t), param = %s" % (str(param_grid[i, :]));
                save_file_name : str = file_prefix + "_Z" + "_param" + str(param_grid[i, :]) + ".png";
            elif(j == 1):
                title          : str = "Dt Z(t), param = %s" % (str(param_grid[i, :]));
                save_file_name : str = file_prefix + "_Dt_Z" + "_param" + str(param_grid[i, :]) + ".png";
            else:
                title          : str = "Dt^%d Z(t), param = %s" % (j, str(param_grid[i, :]));
                save_file_name : str = file_prefix + ("_Dt^%d_Z" % (j)) + "_param" + str(param_grid[i, :]) + ".png";
            
            # Add plot labels and legend.
            plt.xlabel(r'$t$');
            plt.ylabel(r'$z$');
            plt.title(title);

            # Save the figure under Higher-Order-LaSDI/Figures (independent of CWD).
            figures_dir : Path = Path(Figures_Path);
            figures_dir.mkdir(parents = True, exist_ok = True);
            save_file_path: str = str(figures_dir / save_file_name);
            plt.savefig(save_file_path);

            # Show the plot for this IC and combination of parameter values.
            plt.show();

    # All done!
    return;
    


# -------------------------------------------------------------------------------------------------
# Heatmaps!
# -------------------------------------------------------------------------------------------------


def Generate_Heatmap_Data(  encoder_decoder : EncoderDecoder,
                            physics         : Physics,
                            param_space     : ParameterSpace,
                            latent_dynamics : LatentDynamics,
                            interpolator    : Interpolate,
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
    
    interpolator : Interpolate
        Interpolator object for the native latent-dynamics coefficients. For each combination of
        parameter values, we sample coefficient dictionaries from this object and use them to sample
        the predicted dynamics produced by that combination of parameter values.

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
        The number of coefficient samples we draw from the Interpolate posterior. Each sample gives us 
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
    
    coef_means : numpy.ndarray, shape = (n_Test, n_Coef)
        i, j element holds the mean of the posterior distribution for the j'th coefficient 
        evaluated at the i'th combination of testing parameters.

    coef_stds : numpy.ndarray, shape = (n_Test, n_Coef)
        i, j element holds the stds of the posterior distribution for the j'th coefficient 
        evaluated at the i'th combination of testing parameters.
    """ 

    # Run checks
    assert isinstance(interpolator,      Interpolate), "type(interpolator) = %s, expected Interpolate" % (type(interpolator));
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
    
    # Evaluate posterior means/stds through the Interpolate interface in native coefficient format,
    # then flatten to the legacy matrix shape used by heatmap plotting.
    coef_means_native : list[dict[str, torch.Tensor]] = [interpolator.mean(param_test[i, :]) for i in range(n_Test)];
    coef_stds_native  : list[dict[str, torch.Tensor]] = [interpolator.std(param_test[i, :])  for i in range(n_Test)];
    coef_means = latent_dynamics.flatten_coefficients(coef_means_native);
    coef_stds  = latent_dynamics.flatten_coefficients(coef_stds_native);


    # ---------------------------------------------------------------------------------------------
    # Draw n_samples samples of the posterior distribution.

    # For each combination of parameter values in the testing set, sample the latent coefficients 
    # and solve the latent dynamics forward in time. 
    LOGGER.info("Generating latent dynamics trajectories for %d samples of the coefficients for %d combinations of testing parameter" % (n_samples, n_Test));
    Zis_samples     : list[list[numpy.ndarray]] = Sample_Rollouts(encoder_decoder, physics, latent_dynamics, interpolator, param_test, t_Test, n_samples, trainer = trainer);    # len = n_test. i'th element is an n_IC element list whose j'th element has shape (n_t(i), n_samples, n_z)

    LOGGER.info("Generating latent dynamics trajectories using posterior distribution means for %d combinations of testing parameter" % (n_Test));
    Zis_mean        : list[list[numpy.ndarray]] = Mean_Rollout(encoder_decoder, physics, latent_dynamics, interpolator, param_test, t_Test, trainer = trainer);               # len = n_test. i'th element is an n_IC element list whose j'th element has shape (n_t(i), n_z)
        

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





def Plot_Heatmap2d( values          : numpy.ndarray, 
                    param_space     : ParameterSpace,
                    figsize         : tuple[int]    = (10, 10), 
                    title           : str           = '',
                    save_file_name  : str           = "Heatmap",
                    show_plot       : bool          = True,
                    annotate_cells  : bool          = True) -> None:
    """
    This plot makes a "heatmap". Specifically, we assume that values represents the samples of 
    a function which depends on two parameters, p1 and p2 (the two variables in the 
    ParameterSpace object). The i,j entry of values represents the value of some function when 
    p1 takes on it's i'th value and p2 takes on it's j'th. 
    
    We make an image whose i, j has a color based on values[i, j]. We also add boxes around 
    each pixel that is part of the training set (with special red boxes for elements of the 
    initial training set).

    

    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    values : numpy.ndarray, shape = (n1, n2)
        i,j element holds the value of some function (that depends on two parameters, p1 and p2) 
        when p1 = param_space.test_meshgrid[0][i, 0] and p2 = param_space.test_meshgrid[1][0, j]. 
        Here, n1 and n2 represent the number of distinct values for the p1 and p2 parameters, 
        respectively.

    param_space : ParameterSpace
        A ParameterSpace object which holds the combinations of parameters in the testing and 
        training sets. We assume that this object has two parameters (it's n_p attribute is two).

    figsize : tuple[int], len = 2
        A two element tuple specifying the size of the overall figure size. 

    title : str
        The plot title.

    save_file_name : str
        The name of the file in which we want to save the figure in the Figures directory.
    
    show_plot : bool
        If true, we will display the plot after saving it. Otherwise, we will not (save only). 
    
    annotate_cells : bool
        If true, we add labels to each cell of the plot. If not, then we do not (though you 
        can still approximate the cell's value based on its color). Disabling this can 
        considerably speed up plotting.


    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    Nothing!
    """

    # Checks
    assert isinstance(values, numpy.ndarray),       "type(values) = %s" % type(values);
    assert isinstance(param_space, ParameterSpace), "type(param_space) = %s" % type(param_space);
    assert param_space.n_p  == 2,                    "param_space.n_p = %d != 2" % param_space.n_p;
    assert values.ndim      == 2,                    "values.ndim = %d != 2" % values.ndim;

    p1_grid : numpy.ndarray     = param_space.test_meshgrid[0][:, 0];
    p2_grid : numpy.ndarray     = param_space.test_meshgrid[1][0, :];
    n1      : int               = p1_grid.shape[0];
    n2      : int               = p2_grid.shape[0];
    assert values.shape[0]  == n1, "values.shape[0] = %d != n1 = %d" % (values.shape[0], n1);
    assert values.shape[1]  == n2, "values.shape[1] = %d != n2 = %d" % (values.shape[1], n2);

    assert isinstance(figsize, tuple), "type(figsize) = %s" % type(figsize);
    assert len(figsize)     == 2,      "len(figsize) = %d != 2" % len(figsize);
    
    # Setup.
    n_train         : int           = param_space.n_train();
    n_test          : int           = param_space.n_test();
    param_names     : list[str]     = param_space.param_names;
    n_init_train    : int           = param_space.n_init_train;
    LOGGER.info("Making \"%s\" heatmap. Parameters = %s. %d training points (%d initial) and %d testing points." % (title, str(param_names), n_train, n_init_train, n_test));


    # ---------------------------------------------------------------------------------------------
    # Make the heatmap!

    # Set up the subplots.
    if(show_plot == True):
        fig, ax = plt.subplots(1, 1, figsize = figsize);
    else:
        # When not showing the plot, render via the Agg canvas to avoid GUI/X11 overhead.
        # This keeps interactive plots working elsewhere in the run, while making save-only
        # plots fast and backend-independent.
        fig = Figure(figsize = figsize);
        FigureCanvasAgg(fig);  # attach an Agg canvas so fig.savefig works as expected
        ax = fig.add_subplot(1, 1, 1);
    LOGGER.debug("Making the initial heatmap");

    # Set up the color map.
    from matplotlib.colors import LinearSegmentedColormap;
    cmap = LinearSegmentedColormap.from_list('rg', ['C0', 'w', 'C3'], N = 256);

    # Plot the figure as an image (the i,j pixel is just value[i, j], the value associated with 
    # the i'th value of p1 and j'th value of p2
    im = ax.imshow(values.T, cmap = cmap);
    fig.colorbar(im, ax = ax, fraction = 0.04);
    
    # Format tick labels with scientific notation for small values
    def format_tick_label(val):
        """Format tick labels: use scientific notation if |val| < 0.01 or |val| > 1000, else use 2 decimals."""
        if val == 0.0:
            return '0.0';
        elif abs(val) < 0.01 or abs(val) > 1000:
            return f'{val:.2e}';
        else:
            return f'{val:.2f}';
    
    ax.set_xticks(numpy.arange(0, n1, 2), labels = [format_tick_label(val) for val in p1_grid[::2]]);
    ax.set_yticks(numpy.arange(0, n2, 2), labels = [format_tick_label(val) for val in p2_grid[::2]]);

    # Add the value itself (as text) to the center of each "pixel".
    LOGGER.debug("Adding values to the center of each pixel");
    if(annotate_cells):
        for i in range(n1):
            for j in range(n2):
                label_ij : str = f"{values[i, j]:.3g}";
                ax.text(i, j, label_ij, fontsize = 10, ha = 'center', va = 'center', color = 'k');


    # ---------------------------------------------------------------------------------------------
    # Add boxes around each "pixel" corresponding to a training point. 

    # Stuff to help us plot the boxes.
    grid_square_x   : numpy.ndarray = numpy.arange(-0.5, n1, 1);
    grid_square_y   : numpy.ndarray = numpy.arange(-0.5, n2, 1);

    # Add boxes around parameter combinations in the training set.
    LOGGER.debug("Adding boxes around parameters in the training set");
    for i in range(n_train):
        p1_index : float = numpy.sum(p1_grid < param_space.train_space[i, 0]);
        p2_index : float = numpy.sum(p2_grid < param_space.train_space[i, 1]);

        # Add red boxes around the initial points and black ones around points we added to the 
        # training set in later rounds.
        if i < n_init_train:
            color : str = 'r';
        else:
            color : str = 'k';

        # Add colored lines around the pixel corresponding to the i'th training combination.
        ax.plot([grid_square_x[p1_index],       grid_square_x[p1_index]     ],  [grid_square_y[p2_index],       grid_square_y[p2_index] + 1 ],  c = color, linewidth = 2);
        ax.plot([grid_square_x[p1_index] + 1,   grid_square_x[p1_index] + 1 ],  [grid_square_y[p2_index],       grid_square_y[p2_index] + 1 ],  c = color, linewidth = 2);
        ax.plot([grid_square_x[p1_index],       grid_square_x[p1_index] + 1 ],  [grid_square_y[p2_index],       grid_square_y[p2_index]     ],  c = color, linewidth = 2);
        ax.plot([grid_square_x[p1_index],       grid_square_x[p1_index] + 1 ],  [grid_square_y[p2_index] + 1,   grid_square_y[p2_index] + 1 ],  c = color, linewidth = 2);


    # ---------------------------------------------------------------------------------------------
    # Finalize the plot!

    # Set plot labels and plot!
    # Position x-axis label at the right of the axis
    # Axis labels/ticks: keep labels inside the figure and increase readability.
    ax.set_xlabel(param_names[0], fontsize = 16, labelpad = 10);
    # Place x label slightly closer to the heatmap (and avoid being clipped).
    ax.xaxis.set_label_coords(0.5, -0.06);

    # y label at top-left (horizontal), avoiding overlap with tick labels
    ax.set_ylabel(param_names[1], fontsize = 16, rotation = 0, labelpad = 12);
    ax.yaxis.set_label_coords(-0.08, 1.02);

    ax.tick_params(axis = 'both', which = 'major', labelsize = 12);
    ax.set_title(title, fontsize = 25);

    # Save the figure under Higher-Order-LaSDI/Figures (independent of CWD).
    figures_dir: Path = Path(Figures_Path);
    figures_dir.mkdir(parents=True, exist_ok=True);
    save_file_path: str = str(figures_dir / save_file_name);
    # Ensure labels/ticks are not clipped in saved figures.
    fig.tight_layout(rect = [0.06, 0.08, 0.98, 0.95]);
    fig.savefig(save_file_path);
    
    # Show the plot and then return!
    if(show_plot == True):
        plt.show();
    plt.close(fig);

    return;




# -------------------------------------------------------------------------------------------------
# train space relative error heatmaps
# -------------------------------------------------------------------------------------------------


def trainSpace_RelativeErrors_Heatmap(
            trainer         : Trainer,
            param_space     : ParameterSpace,
            figsize         : tuple[int]    = (10, 10), 
            title           : str           = '',
            file_prefix     : str           = "") -> None:
    """
    This function creates heatmaps showing the relative errors between all pairs of training 
    trajectories. The (i,j) cell of the d'th heatmap displays the relative error of d'th 
    derivative of the i-th train trajectory relative to the d'th derivative of the j-th train 
    trajectory, computed as:
    
        relative_error[d, i,j] = 100 * ||U_Train[i][d] - U_Train[j][d]||_2 / ||U_Train[j][d]||_2
        
    Each row and column is labeled with the corresponding parameter values (displayed as tuples), 
    and the title includes the parameter names to provide context.

    
    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    trainer : Trainer
        A Trainer object that holds the training trajectories in its U_Train  attribute. U_Train 
        should be a list of length n_train, where each element is a list of torch.Tensors 
        representing different initial conditions or derivatives.

    param_space : ParameterSpace
        A ParameterSpace object which holds the training parameter combinations in its train_space 
        attribute. The parameter names are stored in param_space.param_names.

    figsize : tuple[int], len = 2
        A two-element tuple specifying the size of the overall figure. Default is (10, 10).

    title : str
        The plot title. This will be displayed at the top of the heatmap.

    file_prefix : str
        We prepend this string to "TrainSpaceRelativeErrorHeatmap" to get the name of the file 
        (without path) in which to save the figure in the Figures directory.
    

    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    Nothing!
    """

    # Checks
    assert isinstance(param_space, ParameterSpace), "type(param_space) = %s" % type(param_space);
    assert hasattr(trainer, 'U_Train'),             "trainer has no U_Train attribute";
    assert isinstance(trainer.U_Train, list),       "type(trainer.U_Train) = %s" % type(trainer.U_Train);
    assert hasattr(trainer, 't_Train'),             "trainer has no t_Train attribute (needed to interpolate)";
    assert isinstance(figsize, tuple),              "type(figsize) = %s" % type(figsize);
    assert len(figsize) == 2,                       "len(figsize) = %d" % len(figsize);

    # Get the number of train trajectories and parameter names
    n_train     : int       = len(trainer.U_Train);     # len = n_train, i'th element is a list of length n_IC
    param_names : list[str] = param_space.param_names;
    n_p         : int       = param_space.n_p;
    
    assert n_train == param_space.n_train(),        "n_train = %d != param_space.n_train() = %d" % (n_train, param_space.n_train());
    
    LOGGER.info("Making train space relative errors heatmap for %d training trajectories with parameters %s" % (n_train, str(param_names)));

    
    # ---------------------------------------------------------------------------------------------
    # Compute the relative errors between all pairs of train trajectories

    def _interp_U_to_time_grid(
        t_src: torch.Tensor,
        U_src: torch.Tensor,
        t_tgt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Linearly interpolate U_src(t_src) onto t_tgt.

        Returns:
            U_interp: Tensor of shape (n_eval, ...) corresponding to U_src interpolated at the
                subset of t_tgt that lies within [min(t_src), max(t_src)].
            mask: boolean mask of shape (t_tgt.shape[0],) indicating which t_tgt points were used.

        Notes:
            - No extrapolation: we only evaluate on the overlapping time interval.
            - Works for U_src with arbitrary spatial dimensions; time must be dim 0.
        """
        assert t_src.ndim == 1, "t_src must be 1D";
        assert t_tgt.ndim == 1, "t_tgt must be 1D";
        assert U_src.shape[0] == t_src.shape[0], "U_src.shape[0] must match t_src.shape[0]";

        # Work on CPU for predictable behavior.
        t_src_c = t_src.detach().cpu();
        U_src_c = U_src.detach().cpu();
        t_tgt_c = t_tgt.detach().cpu();

        # Ensure t_src is sorted (required for searchsorted).
        if t_src_c.shape[0] >= 2 and not bool(torch.all(t_src_c[1:] >= t_src_c[:-1])):
            order = torch.argsort(t_src_c);
            t_src_c = t_src_c[order];
            U_src_c = U_src_c[order];

        # Figure out which time points in t_tgt are within the source time interval.
        mask = (t_tgt_c >= t_src_c[0]) & (t_tgt_c <= t_src_c[-1]);
        t_eval = t_tgt_c[mask];

        # If no overlap, return an empty tensor and mask.
        if t_eval.numel() == 0:
            return U_src_c[:0], mask;

        # Flatten spatial dims so we can interpolate all components in parallel.
        U_src_flat = U_src_c.reshape(U_src_c.shape[0], -1);  # (n_src, n_feat)

        # For each time value in t_eval, find the index of the time value in t_src 
        # that is just <= the time value in t_eval. Specifically, the i'th element
        # of hi is the index j such that t_src_c[j - 1] < t_eval[i] <= t_src_c[j].
        hi_src = torch.searchsorted(t_src_c, t_eval, right = False);  # shape = (n_eval,)
        hi_src = torch.clamp(hi_src, 1, t_src_c.shape[0] - 1);            # clamp to avoid out of bounds errors.
        lo_src = hi_src - 1;

        # Find the time step sizes in t_src.
        t0_src = t_src_c[lo_src];    # i'th element holds the time value just before the i'th time value in t_eval.
        t1_src = t_src_c[hi_src];    # i'th element holds the time value just after the i'th time value in t_eval.
        dt_src = (t1_src - t0_src);
  
        # The i'th element of t_eval occurs somewhere between t0[i] and t1[i]. 
        # We compute 'w', whose i'th element is the proportion of the way from t0[i] to t1[i]
        # where t_eval[i] lives. We guard against any repeated time values (dt==0) by falling 
        # back to left value.
        dt_safe = torch.where(dt_src == 0, torch.ones_like(dt_src), dt_src);
        w = ((t_eval - t0_src) / dt_safe).unsqueeze(1);  # (n_eval, 1)

        # Use 'w' to compute the linear interpolation of U_src_flat[lo[i]] and U_src_flat[hi[i]] 
        # to get the i'th element of U_interp.
        U0_src = U_src_flat[lo_src];
        U1_src = U_src_flat[hi_src];
        U_src_interp_flat = U0_src + w * (U1_src - U0_src);  # (n_eval, n_feat)

        U_src_interp = U_src_interp_flat.reshape(t_eval.shape[0], *U_src_c.shape[1:]);
        return U_src_interp, mask;
    
    
    # Initialize the relative error matrix
    n_IC            : int           = trainer.n_IC;
    relative_errors : numpy.ndarray = numpy.zeros((n_IC, n_train, n_train));
    
    # Compute relative errors for all pairs (i, j). 
    for d in range(n_IC):
        for i in range(n_train):
            Ui : torch.Tensor = trainer.U_Train[i][d];
            ti : torch.Tensor = trainer.t_Train[i];
            
            for j in range(n_train):
                Uj : torch.Tensor = trainer.U_Train[j][d];
                tj : torch.Tensor = trainer.t_Train[j];

                # Sanity checks: spatial dimensions must match to compare trajectories.
                assert Ui.shape[1:] == Uj.shape[1:], \
                    "Shape mismatch for U_Train[%d][%d] %s vs U_Train[%d][%d] %s" % (
                        i, d, str(tuple(Ui.shape)), j, d, str(tuple(Uj.shape))
                    );

                # Interpolate Ui(t) onto tj. Only evaluate on overlapping time interval (no extrapolation).
                Ui_interp, mask = _interp_U_to_time_grid(t_src = ti, U_src = Ui, t_tgt = tj);

                # Use the same subset of tj for the "true" trajectory Uj (to match time locations).
                Uj_eval = Uj.detach().cpu()[mask];

                # If no overlap in time, mark as invalid.
                if Ui_interp.shape[0] == 0:
                    relative_errors[d, i, j] = -1.0;
                    continue;

                # Flatten and compute the relative error: ||Ui(tj) - Uj(tj)||_2 / ||Uj(tj)||_2
                diff = (Ui_interp - Uj_eval).reshape(-1);
                base = Uj_eval.reshape(-1);
                numerator   : float = torch.norm(diff, p = 2).item();
                denominator : float = torch.norm(base, p = 2).item();
                
                if denominator > 0:
                    relative_errors[d, i, j] = 100*(numerator / denominator);
                else:
                    relative_errors[d, i, j] = -1.0;  # Handle division by zero
        
        
    # ---------------------------------------------------------------------------------------------
    # Create parameter labels for rows and columns
    
    # Get the train parameter combinations
    train_params : numpy.ndarray = param_space.train_space;  # shape = (n_train, n_p)
    
    # Create labels as tuples of parameter values (scientific notation so small values don't round to 0.0)
    param_labels : list[str] = [];
    for i in range(n_train):
        parts: list[str] = [numpy.format_float_scientific(float(v), precision = 2, unique = False, trim = 'k')
                            for v in train_params[i, :]];
        param_labels.append("(" + ", ".join(parts) + ")");
    
    
    # ---------------------------------------------------------------------------------------------
    # Make the heatmaps!
    
    for d in range(n_IC):
        # Set up the figure
        fig, ax = plt.subplots(1, 1, figsize = figsize);
        LOGGER.debug("Creating the relative errors heatmap");
        
        # Set up the color map
        from matplotlib.colors import LinearSegmentedColormap;
        cmap = LinearSegmentedColormap.from_list('rg', ['C0', 'w', 'C3'], N = 256);
            
        # Plot the heatmap using imshow
        im = ax.imshow(relative_errors[d, :, :], cmap = cmap, aspect = 'auto');
        fig.colorbar(im, ax = ax, fraction = 0.04);
        
        # Set the tick labels
        ax.set_xticks(numpy.arange(n_train));
        ax.set_yticks(numpy.arange(n_train));
        ax.set_xticklabels(param_labels, rotation = 45, ha = 'right', fontsize = 8);
        ax.set_yticklabels(param_labels, fontsize = 8);
        
        # Add the relative error values as text in each cell
        LOGGER.debug("Adding relative error values to each cell");
        for i in range(n_train):
            for j in range(n_train):
                ax.text(j, i, f'{relative_errors[d, i, j]:.2f}%', 
                    fontsize = 8, ha = 'center', va = 'center', color = 'k');
        
    
        # Create a title that includes the parameter names
        param_names_tuple   : str = str(tuple(param_names));
        full_title          : str = title + '\n' + f'Parameters: {param_names_tuple}' if title else f'Train Space Relative Errors\nParameters: {param_names_tuple}';
        
        # Set plot labels and title
        ax.set_xlabel('Train trajectory (j)', fontsize = 12, loc='right');
        ax.set_ylabel('Train trajectory (i)', fontsize = 12, rotation = 0, loc='top', labelpad=10);
        ax.set_title("D^%d U Relative Errors" % d + '\n' + full_title, fontsize = 15);
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout();
        
        # Save the figure under Higher-Order-LaSDI/Figures (independent of CWD).
        figures_dir: Path = Path(Figures_Path);
        figures_dir.mkdir(parents=True, exist_ok=True);
        save_file_path: str = str(figures_dir / (file_prefix + ("_D^%d U_" % d) + "TrainSpaceRelativeErrorHeatmap.png"));
        fig.savefig(save_file_path, dpi = 150, bbox_inches = 'tight');
        LOGGER.info("Saved heatmap to %s" % save_file_path);
    
    # Show the plot and then return!
    plt.show();
    return;
