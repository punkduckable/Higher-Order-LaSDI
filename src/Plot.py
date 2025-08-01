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
from    SolveROMs                   import  sample_roms;
from    ParameterSpace              import  ParameterSpace;


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

def Plot_Latent_Trajectories(physics         : Physics,
                             model           : torch.nn.Module,
                             latent_dynamics : LatentDynamics,
                             gp_list         : list[GaussianProcessRegressor],
                             param_grid      : numpy.ndarray,
                             n_samples       : int,
                             U_True          : list[list[torch.Tensor]],
                             t_Grid          : list[torch.Tensor],
                             file_prefix     : str,
                             figsize         : tuple[int]    = (15, 13)) -> None:
    """
    This function plots the latent trajectories of the latent dynamics model for a combination of 
    parameter values. Specifically, we fetch the FOM IC for the given parameter values, encode then, 
    and then sample the GP posterior distribution to get samples of the latent dynamics, solve and 
    plot each resulting dynamical solution, and then plot the encodings of 
    the FOM trajectory. 


    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    physics : Physics
        A Physics object which acts as a wrapper for the FOM. We use this to get the FOM IC.

    model : torch.nn.Module
        The model we use to encode the FOM IC and the FOM trajectories.

    latent_dynamics : LatentDynamics
        The LatentDynamics model we use to simulate the latent dynamics forward in time.

    gp_list : list[GaussianProcessRegressor], len = n_coef
        A list of GaussianProcessRegressor objects which hold the GP posterior distributions for 
        each latent dynamics coefficient. We use these to sample from the GP posterior distribution
        to get samples of the latent dynamics.

    param_grid : numpy.ndarray, shape = (n_param, n_p)
        A numpy array whose rows holds the parameter values whose latent dynamics we want to plot.
        We assume that the i'th row hodls the i'th combination of parameter values.

    n_samples : int
        The number of samples we want to draw from the GP posterior distribution for each 
        combination of parameter values.

    U_True : list[list[torch.Tensor]], len = n_param
        The i'th element is an n_IC element list whose j'th element is a torch.Tensor of shape 
        (n_t_i,) + physics.Frame_Shape whose k'th row holds the j'th time derivative of the FOM 
        solution for the i'th combination of prameter values at t_Grid[i][k].

    t_Grid : list[torch.Tensor], len = n_param
        The i'th element is a 1D torch.Tensor object which holds the time grid for the i'th 
        combination of parameter values. We assume that this tensor has shape (n_t_i,).

    file_prefix : str
        The prefix of the file name we use to save the plots. Usually the name of the FOM model.

    figsize : tuple[int], len = 2
        A two element tuple specifying the size of the overall figure size. 


    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    Nothing!
    """ 

    # Checks
    assert(isinstance(physics, Physics));
    assert(isinstance(model, torch.nn.Module));
    assert(isinstance(latent_dynamics, LatentDynamics));
    assert(isinstance(gp_list, list));
    assert(len(gp_list)     == latent_dynamics.n_coefs);
    for i in range(latent_dynamics.n_coefs):
        assert(isinstance(gp_list[i], GaussianProcessRegressor));

    assert(isinstance(param_grid, numpy.ndarray));
    assert(param_grid.ndim     == 2);
    assert(param_grid.shape[1] == physics.n_p);
    n_param : int = param_grid.shape[0];

    assert(isinstance(n_samples, int));
    assert(isinstance(U_True, list));
    assert(isinstance(t_Grid, list));
    assert(len(t_Grid)      == n_param);
    assert(len(U_True)      == n_param);
    for i in range(n_param):
        assert(isinstance(U_True[i], list));
        assert(len(U_True[i])  == physics.n_IC);
        assert(isinstance(t_Grid[i], torch.Tensor));
        assert(t_Grid[i].ndim     == 1);
        n_t_i : int = t_Grid[i].shape[0];   # number of time steps for the i'th combination of parameter values.
        for j in range(physics.n_IC):
            assert(isinstance(U_True[i][j], torch.Tensor));
            assert(U_True[i][j].ndim >= 2);
            assert(U_True[i][j].shape[0]    == n_t_i);

    assert(isinstance(figsize, tuple));
    assert(len(figsize)     == 2);


    # ---------------------------------------------------------------------------------------------
    # Generate the Latent Trajectories.

    # First generate the latent trajectories. This is a an n_param element list whose i'th element
    # is an n_IC element list whose j'th element is a 3d array of shape (n_t(i), n_samples, n_z). 
    # Here, n_param is the number of combinations of parameter values.
    LOGGER.info("Solving the latent dynamics using %d samples of the posterior distributions for %d combinations of parameter values" % (n_samples, n_param));
    Predicted_Latent_Trajectories : list[list[numpy.ndarray]] = sample_roms( 
                                                                    model           = model, 
                                                                    physics         = physics, 
                                                                    latent_dynamics = latent_dynamics, 
                                                                    gp_list         = gp_list, 
                                                                    param_grid      = param_grid,
                                                                    t_Grid          = t_Grid,
                                                                    n_samples       = n_samples);
    
    # Now encode the FOM trajectories. Store these in an n_param element list whose i'th element
    # is an n_IC element list whose j'th element is a numpy array of shape (n_t(i), n_z) holding
    # the encoding of the j'th FOM trajectory for the i'th combination of parameter values.
    True_Latent_Trajectories : list[list[numpy.ndarray]] = [];          # len = n_param
    for i in range(n_param):
        ith_True_Latent_Trajectories : list[numpy.ndarray] = [];
        ith_Encoding : torch.Tensor | tuple[torch.Tensor] = model.Encode(*U_True[i]);
        if(isinstance(ith_Encoding, tuple)):
            # If the encoding is a tuple, then we need to convert it to a list.
            for j in range(len(ith_Encoding)):
                ith_True_Latent_Trajectories.append(ith_Encoding[j].detach().numpy());
        elif(isinstance(ith_Encoding, torch.Tensor)):
            ith_True_Latent_Trajectories.append(ith_Encoding.detach().numpy());
        else:
            raise ValueError("ith_Encoding is not a tuple or a torch.Tensor");
        
        True_Latent_Trajectories.append(ith_True_Latent_Trajectories);
        

    # ---------------------------------------------------------------------------------------------
    # Make the plots!

    # Set up the subplots.
    LOGGER.info("Making latent trajectory plots for %d combinations of parameter values" % n_param);
    for i in range(n_param):
        for j in range(physics.n_IC):
            # Set up the plot for this combination of parameter values.
            plt.figure(figsize = figsize);

            # Plot the predicted latent trajectories
            for s in range(n_samples):
                for k in range(latent_dynamics.n_z):
                    plt.plot(Predicted_Latent_Trajectories[i][j][:, s, k], 'C' + str(k), linewidth = 1, alpha = 0.3);

            # Plot each component of the latent trajectories
            for k in range(latent_dynamics.n_z):
                plt.plot(True_Latent_Trajectories[i][j][:, k], 'C' + str(k), linewidth = 3, alpha = 0.75);
            
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

            # Save the figure.
            save_file_path : str = os.path.join(os.path.join(os.path.pardir, "Figures"), save_file_name);
            plt.savefig(save_file_path);

            # Show the plot for this IC and combination of parameter values.
            plt.show();

    # All done!
    return;
    


def Plot_Heatmap2d( values          : numpy.ndarray, 
                    param_space     : ParameterSpace,
                    figsize         : tuple[int]    = (10, 10), 
                    title           : str           = '',
                    save_file_name  : str           = "Heatmap") -> None:
    """
    This plot makes a "heatmap". Specifically, we assume that values represents the samples of 
    a function which depends on two paramaters, p1 and p2 (the two variables in the 
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
        The name of the file in which we want to save the figure in the Figures directiory.
    


    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    Nothing!
    """

    # Checks
    assert(isinstance(values, numpy.ndarray));
    assert(isinstance(param_space, ParameterSpace));
    assert(param_space.n_p  == 2);
    assert(values.ndim      == 2);

    p1_grid : numpy.ndarray     = param_space.test_meshgrid[0][:, 0];
    p2_grid : numpy.ndarray     = param_space.test_meshgrid[1][0, :];
    n1      : int               = p1_grid.shape[0];
    n2      : int               = p2_grid.shape[0];
    assert(values.shape[0]  == n1);
    assert(values.shape[1]  == n2);

    assert(isinstance(figsize, tuple));
    assert(len(figsize)     == 2);
    
    # Setup.
    n_train         : int           = param_space.n_train();
    n_test          : int           = param_space.n_test();
    param_names     : list[str]     = param_space.param_names;
    n_init_train    : int           = param_space.n_init_train;
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
    # the i'th value of p1 and j'th value of p2
    im = ax.imshow(values.T, cmap = cmap);
    fig.colorbar(im, ax = ax, fraction = 0.04);
    ax.set_xticks(numpy.arange(0, n1, 2), labels = numpy.round(p1_grid[::2], 2));
    ax.set_yticks(numpy.arange(0, n2, 2), labels = numpy.round(p2_grid[::2], 2));

    # Add the value itself (as text) to the center of each "pixel".
    LOGGER.debug("Adding values to the center of each pixel");
    for i in range(n1):
        for j in range(n2):
            ax.text(i, j, round(values[i, j], 2), ha = 'center', va = 'center', color = 'k');


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

    # Set plot lables and plot!
    ax.set_xlabel(param_names[0], fontsize = 15);
    ax.set_ylabel(param_names[1], fontsize = 15, rotation = 0);
    ax.set_title(title, fontsize = 25);

    # Save the figure.
    save_file_path : str = os.path.join(os.path.join(os.path.pardir, "Figures"), save_file_name);
    fig.savefig(save_file_path);
    
    # Show the plot and then return!
    plt.show();
    return;