# -------------------------------------------------------------------------------------------------
# Import and Setup
# -------------------------------------------------------------------------------------------------

import  os;
import  sys;
physics_path    : str   = os.path.join(os.path.curdir, "Physics");
ld_path         : str   = os.path.join(os.path.curdir, "LatentDynamics");
sys.path.append(physics_path);
sys.path.append(ld_path);

import  logging;

import  torch;
import  numpy;
import  matplotlib.pyplot           as      plt;
import  matplotlib                  as      mpl;
from    sklearn.gaussian_process    import  GaussianProcessRegressor;

from    Physics                     import  Physics;
from    LatentDynamics              import  LatentDynamics;
from    Model                       import  Autoencoder, Autoencoder_Pair;
from    Simulate                    import  sample_roms;


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
# Plotting code.
# -------------------------------------------------------------------------------------------------

def Plot_Reconstruction(X_True  : list[torch.Tensor], 
                        model   : torch.nn.Module, 
                        t_grid  : numpy.ndarray, 
                        x_grid  : numpy.ndarray, 
                        figsize : tuple[int]        = (15, 4)) -> None:
    """
    This function plots a single fom solution, its reconstruction using model, and their 
    difference. We assume the fom solution is SCALAR VALUED. Further, if the underlying physics
    model requires n_IC initial conditions to initialize (n_IC'th order dynamics) then we produce
    n_IC plots, the d'th one of which depicts the d'th derivative of the X_True and its 
    reconstruction by model.

     

    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    X_True: A list of torch.Tensor objects. The k'th element should be a torch.Tensor object 
    of shape (n_t, n_x) whose i,j entry holds the value of the k'th time derivative of the fom 
    solution at t_grid[i], x_grid[j].
    
    model: A model (i.e., autoencoder). We use this to map the FOM IC's (stored in Physics) to the 
    latent space using the model's encoder.

    t_grid, x_value: The set of t and x values at which we have evaluated the fom solution, 
    respectively. Specifically, we assume that the k'th element of fom_frame is a torch.Tensor 
    object of shape (n_t, n_x) where n_t = t_grid.size and n_x = x_grid.size. We assume that the 
    i, j element of the k'th element of fom_frame represents the fom solution at t = t_grid[i] and 
    x = x_grid[j].

    figsize: a two element array specifying the width and height of the figure.


    
    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    Nothing!
    """
    
    # Run checks.
    n_IC : int = len(X_True);
    assert(len(t_grid.shape)    == 1);
    assert(len(x_grid.shape)    == 1);
    n_t             : int       =  t_grid.size;
    n_x             : int       =  x_grid.size;
    assert(len(figsize)         == 2);

    for d in range(n_IC):
        assert(X_True[d].ndim       == 2);
        assert(X_True[d].shape[0]   == n_t);
        assert(X_True[d].shape[1]   == n_x);


    LOGGER.info("Making a Reconstruction plot with n_t = %d and n_x = %d" % (n_t, n_x));


    # Reshape each element of X_Pred to have a leading dimension of 1 (the model expects 3d tensors
    # whose leading axis corresponds to the number of parameter values. In our case, this should be
    # one.)
    for d in range(n_IC):
        X_True[d] = X_True[d].reshape((1,) + X_True[d].shape);

    # Compute the predictions. 
    if(n_IC == 1):
        X_Pred  : list[torch.Tensor]    = [model.forward(*X_True)];
    else:
        X_Pred  : list[torch.Tensor]    = list(model.forward(*X_True));

    # Map both the true and predicted solutions to numpy arrays.
    # also set up list to hold the difference between the prediction and true solutions.
    Diff_X : list[numpy.ndarray] = [];
    for d in range(n_IC):
        X_True[d] = X_True[d].squeeze().numpy();
        X_Pred[d] = X_Pred[d].squeeze().detach().numpy();
        Diff_X.append(X_True[d] - X_Pred[d]);


    # Get bounds.
    epsilon     : float         = .0001;
    X_min       : list[float]   = [];
    X_max       : list[float]   = [];
    Diff_X_min  : list[float]   = [];
    Diff_X_max  : list[float]   = [];

    for d in range(n_IC):
        X_min.append(       min(numpy.min(X_True[d]), numpy.min(X_Pred[d])) - epsilon);
        X_max.append(       max(numpy.max(X_True[d]), numpy.max(X_Pred[d])) + epsilon);
        Diff_X_min.append(  numpy.min(Diff_X[d]) - epsilon);
        Diff_X_max.append(  numpy.max(Diff_X[d]) + epsilon);


    # Now... plot the results!
    for d in range(n_IC):
        LOGGER.debug("Generating plot for time derivative %d of the fom solution" % d);
        fig, ax  = plt.subplots(1, 5, width_ratios = [1, 0.05, 1, 1, 0.05], figsize = figsize);
        fig.tight_layout();

        im0 = ax[0].contourf(t_grid, x_grid, X_True[d].T, levels = numpy.linspace(X_min[d], X_max[d], 200));  # Note: contourf(X, Y, Z) requires Z.shape = (Y.shape, X.shape) with Z[i, j] corresponding to Y[i] and X[j]
        ax[0].set_title("True");
        ax[0].set_xlabel("t");
        ax[0].set_ylabel("x");

        fig.colorbar(im0, cax = ax[1], format = "%0.2f", location = "left");

        ax[2].contourf(t_grid, x_grid, X_Pred[d].T, levels = numpy.linspace(X_min[d], X_max[d], 200));            
        ax[2].set_title("Prediction");
        ax[2].set_xlabel("t");
        ax[2].set_ylabel("x");


        im3 = ax[3].contourf(t_grid, x_grid, Diff_X[d].T, levels = numpy.linspace(Diff_X_min[d], Diff_X_max[d], 200));
        ax[3].set_title("Difference");
        ax[3].set_xlabel("t");
        ax[3].set_ylabel("x");

        fig.colorbar(im3, cax = ax[4], format = "%0.2f", location = "left");


    # All done!
    plt.show();



def Plot_Prediction(model           : torch.nn.Module, 
                    physics         : Physics, 
                    latent_dynamics : LatentDynamics, 
                    gp_list         : list[GaussianProcessRegressor], 
                    param_grid      : numpy.ndarray, 
                    n_samples       : int, 
                    X_True          : list[numpy.ndarray], 
                    figsize         : tuple[int]        = (14, 8))            -> None:
    """
    This function plots the mean and std (as a function of t, x) prediction of each derivative of
    the fom solution. We also plot each sample of each component of the latent trajectories over 
    time.


    
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

    X_True: A list of n_IC (where n_IC is the number of IC's needed to initialize the latent 
    dynamics. This should also be latent_dynamics.n_IC). The d'th element should be a numpy ndarray 
    object of shape (n_t, n_x), where n_t is the number of points we use to discretize the spatial
    axis of the fom solution domain. The i,j element of the d'th element of X_True should hold 
    the d'th derivative of the fom solution at the i'th time value and j'th spatial position.
    
    figsize: a two element array specifying the width and height of the figure.

    

    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    Nothing!
    """

    # ---------------------------------------------------------------------------------------------
    # Find the predicted solutions
    # ---------------------------------------------------------------------------------------------

    # First generate the latent trajectories. Z is a list of n_IC arrays, each one of which is a 
    # 4d array of shape (n_params, n_samples, n_t, n_z). Here, n_param is the number of 
    # combinations of parameter values.
    Latent_Trajectories : list[torch.Tensor] = sample_roms( 
                                                model           = model, 
                                                physics         = physics, 
                                                latent_dynamics = latent_dynamics, 
                                                gp_list         = gp_list, 
                                                param_grid      = param_grid,
                                                n_samples       = n_samples);

    # Make sure Z consists of a list of n_IC element.
    n_IC : int = latent_dynamics.n_IC;
    assert(len(Latent_Trajectories) == n_IC);

    # Fetch latent dimension.
    n_z : int = model.n_z;
    LOGGER.info("Computing mean/std of predictions. The Latent Trajectories have a shape of (n_params, n_samples, n_t, n_z) = %s" % str(Latent_Trajectories[0].shape));

    # Only keep the predicted solutions when we use the first parameter value. Note that each
    # element of Latent_Trajectories has shape (n_samples, n_t, n_z). Also map everything to 
    # tensors.
    for d in range(n_IC):
        Latent_Trajectories[d] = torch.Tensor(Latent_Trajectories[d][0, :, :, :]);

    # Now generate the predictions.
    X_Pred  : list[torch.Tensor] = list(model.Decode(*Latent_Trajectories));
    assert(len(X_Pred) == n_IC);
    LOGGER.debug("Predictions have shape %s" % str(X_Pred[0].shape));

    # Compute the mean, std of the predictions across the samples.
    X_pred_mean : list[numpy.ndarray] = [];
    X_pred_std  : list[numpy.ndarray] = [];
    for d in range(n_IC):
        X_Pred[d]       = X_Pred[d].detach().numpy();       # X_Pred[i] has shape (n_samples, n_t, n_z).
        X_pred_mean.append( numpy.mean( X_Pred[d], 0));
        X_pred_std.append(  numpy.std(  X_Pred[d], 0));

    # Compute the solution residual (this will tell us how well the predicted solution satisfies 
    # the underlying equation).
    r, _ = physics.residual(X_pred_mean);

    t_grid  : numpy.ndarray = physics.t_grid;
    x_grid  : numpy.ndarray = physics.x_grid;
    if (x_grid.ndim > 1):
        raise RuntimeError('plot_prediction supports only 1D physics!');



    # ---------------------------------------------------------------------------------------------
    # Plot!!!!
    # ---------------------------------------------------------------------------------------------

    for d in range(n_IC):
        LOGGER.debug("Generating plots for derivative %d" % d);
        plt.figure(figsize = figsize);

        # Plot each component of the d'th derivative of the latent state over time (across the 
        # samples of the latent coefficients)
        plt.subplot(231);
        for s in range(n_samples):
            for i in range(n_z):
                plt.plot(t_grid, Latent_Trajectories[d][s, :, i], 'C' + str(i), alpha = 0.3);
        plt.title('Latent Space');

        # Plot the mean of the d'th derivative of the fom solution.
        plt.subplot(232);
        plt.contourf(t_grid, x_grid, X_pred_mean[d].T, 100, cmap = plt.cm.jet);   # Note: contourf(X, Y, Z) requires Z.shape = (Y.shape, X.shape) with Z[i, j] corresponding to Y[i] and X[j].
        plt.colorbar();
        plt.xlabel("t");
        plt.ylabel("x");
        plt.title('Decoder Mean Prediction');
        
        # Plot the std of the d'th derivative of the fom solution.
        plt.subplot(233);
        plt.contourf(t_grid, x_grid, X_pred_std[d].T, 100, cmap = plt.cm.jet);
        plt.colorbar();
        plt.xlabel("t");
        plt.ylabel("x");
        plt.title('Decoder Standard Deviation');

        # Plot the d'th derivative of the true fom solution.
        plt.subplot(234);
        plt.contourf(t_grid, x_grid, X_True[d].T, 100, cmap = plt.cm.jet);
        plt.colorbar();
        plt.xlabel("t");
        plt.ylabel("x");
        plt.title('Ground Truth');

        # Plot the error between the mean predicted d'th derivative and the true d'th derivative of
        # the fom solution.
        plt.subplot(235);
        error = numpy.abs(X_True[d] - X_pred_mean[d]);
        plt.contourf(t_grid, x_grid, error.T, 100, cmap = plt.cm.jet);
        plt.colorbar();
        plt.xlabel("t");
        plt.ylabel("x");
        plt.title('Absolute Error');

        # Finally, plot the residual.
        plt.subplot(236);
        plt.contourf(t_grid[:-1], x_grid[:-1], r.T, 100, cmap = plt.cm.jet);
        plt.colorbar();
        plt.xlabel("t");
        plt.ylabel("x");
        plt.title('Residual');

        plt.tight_layout();

    # All done!
    plt.show();



def Plot_GP2d(  p1_mesh         : numpy.ndarray, 
                p2_mesh         : numpy.ndarray, 
                gp_mean         : numpy.ndarray, 
                gp_std          : numpy.ndarray, 
                param_train     : numpy.ndarray, 
                param_names     : list[str]     = ['p1', 'p2'], 
                n_cols          : int           = 5, 
                figsize         : tuple[int]    = (15, 13), 
                color_levels    : int           = 100, 
                cm                              = plt.cm.jet) -> None:
    """
    This function plots the mean and standard deviation of the posterior distributions of each 
    latent dynamics coefficient as a function the (2) parameters. We assume there are just two 
    parameters, p1 and p2, which condition the coefficient distributions.


    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    p1_mesh: A 2d ndarray object of shape (N(1), N(2)) where N(1), N(2) denote the number of 
    distinct values for the first and second parameters in the training set, respectively. The i,j 
    element of this array holds the i'th value of the first parameter.

    p2_mesh: A 2d ndarray object of shape (N(1), N(2)) whose i,j element holds the j'th value of 
    the second parameter.

    gp_mean: A 3d numpy array of shape (N(1), N(2), n_coef), where n_coef denotes the number of 
    coefficients in the latent model. The i, j, k element of this model holds the mean of the 
    posterior distribution for the k'th parameter when the parameters consist of the  i'th value
    of the first parameter and the j'th of the second.

    gp_std: A 3d numpy array of shape (N(1), N(2), n_coef), where n_coef denotes the number of 
    coefficients in the latent model. The i, j, k element of this model holds the std of the 
    posterior distribution for the k'th parameter when the parameters consist of the  i'th value
    of the first parameter and the j'th of the second.

    param_train: A 2d array of shape (n_train, 2) whose i, j element holds the value of the 
    j'th parameter when we use the i'th combination of testing parameters.

    param_names: A two element list housing the names for the two parameters. 

    n_cols: The number of columns in our subplots.

    figsize: A two element tuple specifying the size of the overall figure size. 
    
    color_levels: The number of color levels to put in our plot.

    cm: The color map we use for the plots.


    
    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    Nothing!
    """
    
    # Checks
    assert(p1_mesh.ndim         == 2);
    assert(p2_mesh.ndim         == 2);
    assert(gp_mean.ndim         == 3);
    assert(gp_std.ndim          == 3);
    assert(param_train.ndim     == 2);
    assert(gp_mean.shape        == gp_std.shape);
    assert(len(param_names)     == 2);

    # First, determine how many coefficients there are.
    n_coef : int = gp_mean.shape[-1];   
    LOGGER.info("Producing GP plots with %d coefficients. The parameters are %s" % (n_coef, str(param_names)));

    # Figure out how many rows/columns of subplots we should make.
    subplot_shape = [n_coef // n_cols, n_cols];
    if (n_coef % n_cols > 0):
        subplot_shape[0] += 1;

    # Set limits for the x/y axes.
    p1_range = [p1_mesh.min()*.99, p1_mesh.max()*1.01];
    p2_range = [p2_mesh.min()*.99, p2_mesh.max()*1.01];

    # Setup the subplots (one for std, another for mean)
    fig_std,    axs_std     = plt.subplots(subplot_shape[0], subplot_shape[1], figsize = figsize);
    fig_mean,   axs_mean    = plt.subplots(subplot_shape[0], subplot_shape[1], figsize = figsize);

    # Cycle through the subplots.
    for i in range(subplot_shape[0]):
        for j in range(subplot_shape[1]):
            # Figure out which combination of parameter values corresponds to the current plot.
            k = j + i * subplot_shape[1];
            LOGGER.debug("Making plot %d" % k);

            # Remove the plot frame.
            axs_std[i, j].set_frame_on(False);
            axs_mean[i, j].set_frame_on(False);


            # -------------------------------------------------------------------------------------
            # There are only n_coef plots. If k > n_coef, then there is nothing to plot but we need 
            # to plot something (to avoid pissing off matplotlib).
            if (k >= n_coef):
                LOGGER.debug("%d > %d (n_coef). Thus, we are making a default plot" % (k, n_coef));
                axs_std[i, j].set_xlim(p1_range);
                axs_std[i, j].set_ylim(p2_range);
                axs_std[i, j].set_frame_on(False);

                axs_mean[i, j].set_xlim(p1_range);
                axs_mean[i, j].set_ylim(p2_range);
                axs_mean[i, j].set_frame_on(False);

                if (j == 0):
                    axs_std[i, j].set_ylabel(param_names[1]);
                    axs_std[i, j].get_yaxis().set_visible(True);
                    axs_mean[i, j].set_ylabel(param_names[1]);
                    axs_mean[i, j].get_yaxis().set_visible(True);
                if (i == subplot_shape[0] - 1):
                    axs_std[i, j].set_xlabel(param_names[0]);
                    axs_std[i, j].get_xaxis().set_visible(True);
                    axs_mean[i, j].set_xlabel(param_names[0]);
                    axs_mean[i, j].get_xaxis().set_visible(True);
                
                continue;


            # -------------------------------------------------------------------------------------
            # Get the coefficient distribution std's for the k'th combination of parameter values.
            std     = gp_std[:, :, k];

            # Plot!!!!
            p       = axs_std[i, j].contourf(p1_mesh, p2_mesh, std, color_levels, cmap = cm);
            fig_std.colorbar(p, ticks = numpy.array([std.min(), std.max()]), format = '%2.2f', ax = axs_std[i, j]);
            axs_std[i, j].scatter(param_train[:, 0], param_train[:, 1], c = 'k', marker = '+');
            axs_std[i, j].set_title(r'$\sqrt{\Sigma^*_{' + str(i + 1) + str(j + 1) + '}}$');
            axs_std[i, j].set_xlim(p1_range);
            axs_std[i, j].set_ylim(p2_range);
            axs_std[i, j].invert_yaxis();
            axs_std[i, j].get_xaxis().set_visible(False);
            axs_std[i, j].get_yaxis().set_visible(False);


            # -------------------------------------------------------------------------------------
            # Get the coefficient distribution mean's for the k'th combination of parameter values.
            mean    = gp_mean[:, :, k];

            # Plot!!!!
            p       = axs_mean[i, j].contourf(p1_mesh, p2_mesh, mean, color_levels, cmap = cm);
            fig_mean.colorbar(p, ticks = numpy.array([mean.min(), mean.max()]), format='%2.2f', ax = axs_mean[i, j]);
            axs_mean[i, j].scatter(param_train[:, 0], param_train[:, 1], c = 'k', marker = '+');
            axs_mean[i, j].set_title(r'$\mu^*_{' + str(i + 1) + str(j + 1) + '}$');
            axs_mean[i, j].set_xlim(p1_range);
            axs_mean[i, j].set_ylim(p2_range);
            axs_mean[i, j].invert_yaxis();
            axs_mean[i, j].get_xaxis().set_visible(False);
            axs_mean[i, j].get_yaxis().set_visible(False);


            # -------------------------------------------------------------------------------------
            # Add plot labels (but only if the current subplot is in the first column or final 
            # row).
            if (j == 0):
                axs_std[i, j].set_ylabel(param_names[1]);
                axs_std[i, j].get_yaxis().set_visible(True);
                axs_mean[i, j].set_ylabel(param_names[1]);
                axs_mean[i, j].get_yaxis().set_visible(True);
            if (i == subplot_shape[0] - 1):
                axs_std[i, j].set_xlabel(param_names[0]);
                axs_std[i, j].get_xaxis().set_visible(True);
                axs_mean[i, j].set_xlabel(param_names[0]);
                axs_mean[i, j].get_xaxis().set_visible(True);

    # Make the plots!
    fig_mean.tight_layout();
    fig_std.tight_layout();
    plt.show();

    # All done!
    return;



def Plot_Heatmap2d( values          : numpy.ndarray, 
                    p1_grid         : numpy.ndarray, 
                    p2_grid         : numpy.ndarray, 
                    param_train     : numpy.ndarray,
                    n_init_train    : int,
                    figsize         : tuple[int]    = (10, 10), 
                    param_names     : list[str]     = ['p1', 'p2'], 
                    title           : str           = ''):
    """
    This plot makes a "heatmap". Specifically, we assume that values represents the samples of 
    a function which depends on two paramaters, p1 and p2. The i,j entry of values represents 
    the value of some function when p1 = p1_grid[i] and p2 = p2_grid[j]. We make an image whose 
    i, j has a color based on values[i, j]. We also add boxes around each pixel that is part of 
    the training set (with special red boxes for elements of the initial training set).

    

    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    values: A 2d numpy ndarray object of shape (n1, n2), where n1 and n2 are the length of p1_grid
    and p2_grid, respectively (the number of p1, p2 values).

    p1_grid: The set of possible values for the p1 parameter. This should be a 1d numpy ndarray 
    whose i'th value holds the i'th value for the p1 parameter.

    p2_grid: The same thing as p1_grid, but for the p2 parameter. 

    param_train: A 2d array of shape (n_train, 2) whose i, j element holds the value of the 
    j'th parameter when we use the i'th combination of testing parameters. We assume the first 
    n_init_train rows in this array hold the combinations that were originally in the training 
    set and the rest were added in successive rounds of training.

    n_init_train: The initial number of combinations of parameters in the training set.

    figsize: A two element tuple specifying the size of the overall figure size. 

    param_names: A two element list housing the names for the two parameters. 

    title: The plot title.
    


    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    Nothing!
    """

    # Checks.
    assert(p1_grid.ndim     == 1);
    assert(p2_grid.ndim     == 1);
    assert(values.ndim      == 2);
    assert(param_train.ndim == 2);
    assert(len(figsize)     == 2);
    assert(len(param_names) == 2);

    n_p1    : int = len(p1_grid);
    n_p2    : int = len(p2_grid);
    assert(values.shape[0] == n_p1);
    assert(values.shape[1] == n_p2);

    # Setup.
    n_train : int   = param_train.shape[0];
    n_test  : int   = len(p1_grid)*len(p2_grid);
    LOGGER.info("Making heatmap. Parameters = %s. There are %d training points (%d initial) and %d testing points." % (str(param_names), n_train, n_init_train, n_test));


    # ---------------------------------------------------------------------------------------------
    # Make the heatmap!

    # Set up the subplots.
    fig, ax = plt.subplots(1, 1, figsize = figsize);
    LOGGER.debug("Making the initial heatmap");

    # Set up the color map.
    from matplotlib.colors import LinearSegmentedColormap;
    cmap = LinearSegmentedColormap.from_list('rg', ['C0', 'w', 'C3'], N = 256);

    # Plot the figure as an image (the i,j pixel is just value[i, j], the value associated with 
    # the i'th value of p1 and j'th value of p2.
    im = ax.imshow(values, cmap = cmap);
    fig.colorbar(im, ax = ax, fraction = 0.04);
    ax.set_xticks(numpy.arange(0, n_p1, 2), labels = numpy.round(p1_grid[::2], 2));
    ax.set_yticks(numpy.arange(0, n_p2, 2), labels = numpy.round(p2_grid[::2], 2));

    # Add the value itself (as text) to the center of each "pixel".
    LOGGER.debug("Adding values to the center of each pixel");
    for i in range(n_p1):
        for j in range(n_p2):
            ax.text(j, i, round(values[i, j], 1), ha = 'center', va = 'center', color = 'k');


    # ---------------------------------------------------------------------------------------------
    # Add boxes around each "pixel" corresponding to a training point. 

    # Stuff to help us plot the boxes.
    grid_square_x   : numpy.ndarray = numpy.arange(-0.5, n_p1, 1);
    grid_square_y   : numpy.ndarray = numpy.arange(-0.5, n_p2, 1);

    # Add boxes around parameter combinations in the training set.
    LOGGER.debug("Adding boxes around parameters in the training set");
    for i in range(n_train):
        p1_index : float = numpy.sum(p1_grid < param_train[i, 0]);
        p2_index : float = numpy.sum(p2_grid < param_train[i, 1]);

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

    # Set plot lables and plot!
    ax.set_xlabel(param_names[0], fontsize = 15);
    ax.set_ylabel(param_names[1], fontsize = 15);
    ax.set_title(title, fontsize = 25);
    plt.show();

    # All done!
    return;