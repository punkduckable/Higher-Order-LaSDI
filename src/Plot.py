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
    n_t             : int   = t_grid.size;
    n_x             : int   = x_grid.size;
    assert(len(figsize)     == 2);

    for d in range(n_IC):
        assert(X_True[d].shape[0]    == n_t);
        assert(X_True[d].shape[1]    == n_x);


    # Set up the matrix of t, x values.
    t_matrix : numpy.ndarray = numpy.empty(shape = (n_t, n_x), dtype = numpy.float32);
    for i in range(n_t):
        t_matrix[i, :] = t_grid[i];

    x_matrix : numpy.ndarray = numpy.empty(shape = (n_t, n_x), dtype = numpy.float32);
    for j in range(n_x):
        x_matrix[:, j] = x_grid[j];
    

    # Reshape each element of X_Pred to have a leading dimension of 1 (the model expects 3d tensors
    # whose leading axis corresponds to the number of parameter values. In our case, this should be
    # one.)
    for d in range(n_IC):
        X_True[d] = X_True[d].reshape((1,) + X_True[d].shape);


    # Compute the predictions. The way this works depends on what class model is.
    if(isinstance(model, Autoencoder)):
            assert(n_IC == 1);

            # Pass the input through the Autoencoder.
            X_Pred  : list[torch.Tensor]    = [model.forward(X_True[0])];

    if(isinstance(model, Autoencoder_Pair)):
            assert(n_IC == 2);

            # Pass the input through the Autoencoder.
            Disp_Pred, Vel_Pred             = model.forward(X_True[0], X_True[1]);
            X_Pred  : list[torch.Tensor]    = [Disp_Pred, Vel_Pred];
    

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
        fig, ax  = plt.subplots(1, 5, width_ratios = [1, 0.05, 1, 1, 0.05], figsize = figsize);
        fig.tight_layout();

        im0 = ax[0].contourf(t_matrix, x_matrix, X_True[d], levels = numpy.linspace(X_min[d], X_max[d], 200));
        ax[0].set_title("True");
        ax[0].set_xlabel("t");
        ax[0].set_ylabel("x");

        fig.colorbar(im0, cax = ax[1], format = "%0.2f", location = "left");

        ax[2].contourf(t_matrix, x_matrix, X_Pred[d], levels = numpy.linspace(X_min[d], X_max[d], 200));
        ax[2].set_title("Prediction");
        ax[2].set_xlabel("t");
        ax[2].set_ylabel("x");


        im3 = ax[3].contourf(t_matrix, x_matrix, Diff_X[d], levels = numpy.linspace(Diff_X_min[d], Diff_X_max[d], 200));
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
                    scale           : int               = 1)            -> None:
    """
    TODO


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

    scale:
    

    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    Nothing!
    """

    # ---------------------------------------------------------------------------------------------
    # Find the predicted solutions
    # ---------------------------------------------------------------------------------------------

    # First generate the latent trajectories. Z is a list of n_IC arrays, each one of which is a 
    # 4d array of shape (n_params, n_samples, n_t, n_z).
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

    # Fetch latent dimension size.
    n_z : int = model.n_z;

    # Only keep the predicted solutions when we use the first parameter value. Note that each
    # element of this updated Z has shape (n_samples, n_t, n_z). 
    for d in range(n_IC):
        Latent_Trajectories[d] = Latent_Trajectories[d][0, :, :, :];

    # Now generate the predictions. The way we do this depend on if we are using an Autoencoder or 
    # an Autoencoder_Pair.
    if(isinstance(model, Autoencoder)):
        assert(n_IC == 1);

        # Pass the predictions through the decoder to get the corresponding fom frames. Note that 
        # X_pred has shape (n_samples, n_t, n_x), where n_x is the number of points along the spatial 
        # axis.
        X_pred        : numpy.ndarray       = model.decoder(torch.Tensor(Latent_Trajectories)).detach().numpy();
        X_pred_mean   : list[numpy.ndarray] = [X_pred.mean(0)];
        X_pred_std    : list[numpy.ndarray] = [X_pred.std(0)];
    
    elif(isinstance(model, Autoencoder_Pair)):
        assert(n_IC == 2);

        # Pass the predictions through the decoder to get the corresponding fom frames. Note that 
        # X_pred has shape (n_samples, n_t, n_x), where n_x is the number of points along the spatial 
        # axis.
        Disp_pred, Vel_Pred = model.decoder(torch.Tensor(Latent_Trajectories[0]), torch.Tensor(Latent_Trajectories[1]));
        Disp_pred           = Disp_pred.detach().numpy();
        Vel_Pred            = Vel_Pred.detach().numpy();

        X_pred_mean   : list[numpy.ndarray] = [Disp_pred.mean(0),   Vel_Pred.mean(0)];
        X_pred_std    : list[numpy.ndarray] = [Disp_pred.std(0),    Vel_Pred.std(0)];
    
    else:
        raise TypeError("model must be Autoencoder or Autoencoder_Pair. Got %s" % str(type(model)));

    # Compute the solution residual (this will tell us how well the predicted solution actually 
    # satisfies the underlying equation).
    r, e = physics.residual(X_pred_mean);

    t_mesh, x_mesh = physics.t_grid, physics.x_grid;
    if (physics.x_grid.ndim > 1):
        raise RuntimeError('plot_prediction supports only 1D physics!');



    # ---------------------------------------------------------------------------------------------
    # Plot!!!!
    # ---------------------------------------------------------------------------------------------

    for d in range(n_IC):
        plt.figure();

        # Plot each component of the d'th derivative of the latent state over time (across the 
        # samples of the latent coefficients)
        plt.subplot(231);
        for s in range(n_samples):
            for i in range(n_z):
                plt.plot(t_mesh, Latent_Trajectories[d][s, :, i], 'C' + str(i), alpha = 0.3);
        plt.title('Latent Space');

        # Plot the mean of the d'th derivative of the fom solution.
        plt.subplot(232);
        plt.contourf(t_mesh, x_mesh, X_pred_mean[d][::scale, ::scale], 100, cmap = plt.cm.jet);
        plt.colorbar();
        plt.title('Decoder Mean Prediction');
        
        # Plot the std of the d'th derivative of the fom solution.
        plt.subplot(233);
        plt.contourf(t_mesh, x_mesh, X_pred_std[d][::scale, ::scale], 100, cmap = plt.cm.jet);
        plt.colorbar();
        plt.title('Decoder Standard Deviation');

        # Plot the d'th derivative of the true fom solution.
        plt.subplot(234);
        plt.contourf(t_mesh, x_mesh, X_True[d][::scale, ::scale], 100, cmap = plt.cm.jet);
        plt.colorbar();
        plt.title('Ground Truth');

        # Plot the error between the mean predicted d'th derivative and the true d'th derivative of
        # the fom solution.
        plt.subplot(235);
        error = numpy.abs(X_True[d] - X_pred_mean[d]);
        plt.contourf(t_mesh, x_mesh, error, 100, cmap = plt.cm.jet);
        plt.colorbar();
        plt.title('Absolute Error');

        # Finally, plot the residual.
        plt.subplot(236);
        plt.contourf(t_mesh[:-1], x_mesh[:-1], r, 100, cmap = plt.cm.jet);
        plt.colorbar();
        plt.title('Residual');

        plt.tight_layout();



def plot_gp2d(  p1_mesh, 
                p2_mesh, 
                gp_mean, 
                gp_std, 
                param_train, 
                param_labels    : list[str]     = ['p1', 'p2'], 
                plot_shape      : list[int]     = [6, 5], 
                figsize         : tuple[int]    = (15, 13), 
                refine                          = 10, 
                cm                              = plt.cm.jet, 
                margin          : float         = 0.05) -> None:
    """
    TODO


    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------




    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    Nothing!
    """
    
    # Checks
    assert(p1_mesh.ndim == 2)
    assert(p2_mesh.ndim == 2)
    assert(gp_mean.ndim == 3)
    assert(gp_std.ndim == 3)
    assert(param_train.ndim == 2)
    assert(gp_mean.shape == gp_std.shape)

    # ???
    plot_shape_ = [gp_mean.shape[-1] // plot_shape[-1], plot_shape[-1]]
    if (gp_mean.shape[-1] % plot_shape[-1] > 0):
        plot_shape_[0] += 1

    # ???
    p1_range = [p1_mesh.min() * (1. - margin), p1_mesh.max() * (1. + margin)]
    p2_range = [p2_mesh.min() * (1. - margin), p2_mesh.max() * (1. + margin)]

    # ???
    fig1, axs1 = plt.subplots(plot_shape_[0], plot_shape_[1], figsize = figsize)
    fig2, axs2 = plt.subplots(plot_shape_[0], plot_shape_[1], figsize = figsize)

    for i in range(plot_shape_[0]):
        for j in range(plot_shape_[1]):
            k = j + i * plot_shape_[1]

            if (k >= gp_mean.shape[-1]):
                axs1[i, j].set_xlim(p1_range)
                axs1[i, j].set_ylim(p2_range)
                axs2[i, j].set_xlim(p1_range)
                axs2[i, j].set_ylim(p2_range)
                if (j == 0):
                    axs1[i, j].set_ylabel(param_labels[1])
                    axs1[i, j].get_yaxis().set_visible(True)
                    axs2[i, j].set_ylabel(param_labels[1])
                    axs2[i, j].get_yaxis().set_visible(True)
                if (i == plot_shape_[0] - 1):
                    axs1[i, j].set_xlabel(param_labels[0])
                    axs1[i, j].get_xaxis().set_visible(True)
                    axs2[i, j].set_xlabel(param_labels[0])
                    axs2[i, j].get_xaxis().set_visible(True)

                continue

            std = gp_std[:, :, k]
            p = axs1[i, j].contourf(p1_mesh, p2_mesh, std, refine, cmap = cm)
            fig1.colorbar(p, ticks = numpy.array([std.min(), std.max()]), format='%2.2f', ax = axs1[i, j])
            axs1[i, j].scatter(param_train[:, 0], param_train[:, 1], c='k', marker='+')
            axs1[i, j].set_title(r'$\sqrt{\Sigma^*_{' + str(i + 1) + str(j + 1) + '}}$')
            axs1[i, j].set_xlim(p1_range)
            axs1[i, j].set_ylim(p2_range)
            axs1[i, j].invert_yaxis()
            axs1[i, j].get_xaxis().set_visible(False)
            axs1[i, j].get_yaxis().set_visible(False)

            mean = gp_mean[:, :, k]
            p = axs2[i, j].contourf(p1_mesh, p2_mesh, mean, refine, cmap = cm)
            fig2.colorbar(p, ticks = numpy.array([mean.min(), mean.max()]), format='%2.2f', ax = axs2[i, j])
            axs2[i, j].scatter(param_train[:, 0], param_train[:, 1], c='k', marker='+')
            axs2[i, j].set_title(r'$\mu^*_{' + str(i + 1) + str(j + 1) + '}$')
            axs2[i, j].set_xlim(p1_range)
            axs2[i, j].set_ylim(p2_range)
            axs2[i, j].invert_yaxis()
            axs2[i, j].get_xaxis().set_visible(False)
            axs2[i, j].get_yaxis().set_visible(False)

            if (j == 0):
                axs1[i, j].set_ylabel(param_labels[1])
                axs1[i, j].get_yaxis().set_visible(True)
                axs2[i, j].set_ylabel(param_labels[1])
                axs2[i, j].get_yaxis().set_visible(True)
            if (i == plot_shape_[0] - 1):
                axs1[i, j].set_xlabel(param_labels[0])
                axs1[i, j].get_xaxis().set_visible(True)
                axs2[i, j].set_xlabel(param_labels[0])
                axs2[i, j].get_xaxis().set_visible(True)

    return



def heatmap2d(values, p1_grid, p2_grid, param_train, n_init, figsize=(10, 10), param_labels=['p1', 'p2'], title=''):
    assert(p1_grid.ndim == 1)
    assert(p2_grid.ndim == 1)
    assert(values.ndim == 2)
    assert(param_train.ndim == 2)

    n_p1 = len(p1_grid)
    n_p2 = len(p2_grid)
    assert(values.shape[0] == n_p1)
    assert(values.shape[1] == n_p2)

    fig, ax = plt.subplots(1, 1, figsize = figsize)

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('rg', ['C0', 'w', 'C3'], N = 256)

    im = ax.imshow(values, cmap = cmap)
    fig.colorbar(im, ax = ax, fraction = 0.04)

    ax.set_xticks(numpy.arange(0, n_p1, 2), labels = numpy.round(p1_grid[::2], 2))
    ax.set_yticks(numpy.arange(0, n_p2, 2), labels = numpy.round(p2_grid[::2], 2))

    for i in range(n_p1):
        for j in range(n_p2):
            ax.text(j, i, round(values[i, j], 1), ha='center', va='center', color='k')

    grid_square_x = numpy.arange(-0.5, n_p1, 1)
    grid_square_y = numpy.arange(-0.5, n_p2, 1)

    n_train = param_train.shape[0]
    for i in range(n_train):
        p1_index = numpy.sum((p1_grid < param_train[i, 0]) * 1)
        p2_index = numpy.sum((p2_grid < param_train[i, 1]) * 1)

        if i < n_init:
            color = 'r'
        else:
            color = 'k'

        ax.plot([grid_square_x[p1_index], grid_square_x[p1_index]], [grid_square_y[p2_index], grid_square_y[p2_index] + 1],
                c=color, linewidth=2)
        ax.plot([grid_square_x[p1_index] + 1, grid_square_x[p1_index] + 1],
                [grid_square_y[p2_index], grid_square_y[p2_index] + 1], c=color, linewidth=2)
        ax.plot([grid_square_x[p1_index], grid_square_x[p1_index] + 1], [grid_square_y[p2_index], grid_square_y[p2_index]],
                c=color, linewidth=2)
        ax.plot([grid_square_x[p1_index], grid_square_x[p1_index] + 1],
                [grid_square_y[p2_index] + 1, grid_square_y[p2_index] + 1], c=color, linewidth=2)

    ax.set_xlabel(param_labels[0], fontsize=15)
    ax.set_ylabel(param_labels[1], fontsize=15)
    ax.set_title(title, fontsize=25)
    return