# -------------------------------------------------------------------------------------------------
# Import and Setup
# -------------------------------------------------------------------------------------------------

import  logging;

import  torch;
import  numpy;
import  matplotlib.pyplot   as      plt;
import  matplotlib          as      mpl;

from    Model               import  Autoencoder, Autoencoder_Pair;


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

def Plot_Frame_2d(  fom_frame   : list[torch.Tensor], 
                    model       : torch.nn.Module, 
                    t_grid      : numpy.ndarray, 
                    x_grid      : numpy.ndarray, 
                    figsize     : tuple[int]        = (15, 4)):
    """
    TODO 

    fom_frame: A list of torch.Tensor objects. The k'th element should be a torch.Tensor object 
    of shape (nt, nx) whose i,j entry holds the value of the k'th time derivative of the fom 
    solution at t_grid[i], x_grid[j].
    """
    
    # Run checks.
    n_derivatives = len(fom_frame);
    assert(len(t_grid.shape)    == 1);
    assert(len(x_grid.shape)    == 1);
    n_t             : int   = t_grid.size;
    n_x             : int   = x_grid.size;

    for i in range(n_derivatives):
        assert(fom_frame[i].shape[0]    == n_t);
        assert(fom_frame[i].shape[1]    == n_x);


    # Set up the matrix of t, x values.
    t_matrix : numpy.ndarray = numpy.empty(shape = (n_t, n_x), dtype = numpy.float32);
    for i in range(n_t):
        t_matrix[i, :] = t_grid[i];

    x_matrix : numpy.ndarray = numpy.empty(shape = (n_t, n_x), dtype = numpy.float32);
    for j in range(n_x):
        x_matrix[:, j] = x_grid[j];


    # Now, make predictions and plot the results.
    if(isinstance(model, Autoencoder)):
        assert(n_derivatives == 1);

        # Pass the input through the Autoencoder.
        X_True  : torch.Tensor  = fom_frame[0];
        X_Pred  : torch.Tensor  = model.forward(X_True);


        # Map everything to numpy arrays.
        X_True  = X_True.numpy();
        X_Pred  = X_Pred.squeeze().detach().numpy();


        # Get bounds.
        epsilon     : float = .0001;
        X_min       : float = min(numpy.min(X_True), numpy.min(X_Pred)) - epsilon;
        X_max       : float = max(numpy.max(X_True), numpy.max(X_Pred)) + epsilon;

        Diff_X_min  : float = numpy.min(Diff_X) - epsilon;
        Diff_X_max  : float = numpy.max(Diff_X) + epsilon;


        # X Plot
        plt.figure(figsize = figsize);

        plt.subplot(1, 3, 1);
        plt.contourf(t_matrix, x_matrix, X_True, levels = numpy.linspace(X_min, X_max, 200));
        plt.title("True");

        plt.subplot(1, 3, 2);
        plt.contourf(t_matrix, x_matrix, X_Pred, levels = numpy.linspace(X_min, X_max, 200));
        plt.title("Prediction");
        plt.colorbar(fraction = 0.1, format = "%0.2f", location = "left");

        plt.subplot(1, 3, 3);
        plt.contourf(t_matrix, x_matrix, Diff_X, levels = numpy.linspace(Diff_X_min, Diff_X_max, 200));
        plt.title("Difference");
        plt.colorbar(fraction = 0.1, format = "%0.2f");


    elif(isinstance(model, Autoencoder_Pair)):
        assert(n_derivatives == 2);

        # Pass the input through the Autoencoder_Pair.
        X_True : torch.Tensor   = fom_frame[0];
        V_True : torch.Tensor   = fom_frame[1];

        X_Pred, V_Pred          = model.forward(Displacement_Frames = X_True.reshape((1,) + X_True.shape), 
                                                Velocity_Frames     = V_True.reshape((1,) + V_True.shape));

        # Map everything to numpy arrays.
        X_True  : numpy.ndarray = X_True.numpy();
        V_True  : numpy.ndarray = V_True.numpy();

        X_Pred  : numpy.ndarray = X_Pred.squeeze().detach().numpy();
        V_Pred  : numpy.ndarray = V_Pred.squeeze().detach().numpy();

        Diff_X  : numpy.ndarray = X_True - X_Pred;
        Diff_V  : numpy.ndarray = V_True - V_Pred;


        # Get bounds.
        epsilon     : float = .0001;
        X_min       : float = min(numpy.min(X_True), numpy.min(X_Pred)) - epsilon;
        X_max       : float = max(numpy.max(X_True), numpy.max(X_Pred)) + epsilon;

        V_min       : float = min(numpy.min(V_True), numpy.min(V_Pred)) - epsilon;
        V_max       : float = max(numpy.max(V_True), numpy.max(V_Pred)) + epsilon;

        Diff_X_min  : float = numpy.min(Diff_X) - epsilon;
        Diff_X_max  : float = numpy.max(Diff_X) + epsilon;

        Diff_V_min  : float = numpy.min(Diff_V) - epsilon;
        Diff_V_max  : float = numpy.max(Diff_V) + epsilon;

        # X Plot
        fig, ax  = plt.subplots(1, 5, width_ratios = [1, 0.05, 1, 1, 0.05], figsize = figsize);
        fig.tight_layout();

        im0 = ax[0].contourf(t_matrix, x_matrix, X_True, levels = numpy.linspace(X_min, X_max, 200));
        ax[0].set_title("True");

        fig.colorbar(im0, cax = ax[1], format = "%0.2f", location = "left");

        ax[2].contourf(t_matrix, x_matrix, X_Pred, levels = numpy.linspace(X_min, X_max, 200));
        ax[2].set_title("Prediction");

        im3 = ax[3].contourf(t_matrix, x_matrix, Diff_X, levels = numpy.linspace(Diff_X_min, Diff_X_max, 200));
        ax[3].set_title("Difference");

        fig.colorbar(im3, cax = ax[4], format = "%0.2f", location = "left");
    
        
        # V Plot
        fig, ax  = plt.subplots(1, 5, width_ratios = [1, 0.05, 1, 1, 0.05], figsize = figsize);
        fig.tight_layout();

        im0 = ax[0].contourf(t_matrix, x_matrix, V_True, levels = numpy.linspace(V_min, V_max, 200));
        ax[0].set_title("True");

        fig.colorbar(im0, cax = ax[1], format = "%0.2f", location = "left");

        ax[2].contourf(t_matrix, x_matrix, V_Pred, levels = numpy.linspace(V_min, V_max, 200));
        ax[2].set_title("Prediction");

        im3 = ax[3].contourf(t_matrix, x_matrix, Diff_V, levels = numpy.linspace(Diff_V_min, Diff_V_max, 200));
        ax[3].set_title("Difference");
        
        fig.colorbar(im3, cax = ax[4], format = "%0.2f", location = "left");

    # All done!
    plt.show();
