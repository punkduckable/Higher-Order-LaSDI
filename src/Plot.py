# -------------------------------------------------------------------------------------------------
# Import and Setup
# -------------------------------------------------------------------------------------------------

import os;
import sys;
from pathlib import Path;

# Resolve paths relative to the project root (Higher-Order-LaSDI/), independent of CWD.
Figures_Path    : str   = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "Figures");
Physics_Path    : str   = os.path.abspath(os.path.join(os.path.dirname(__file__), "Physics"));
LD_Path         : str   = os.path.abspath(os.path.join(os.path.dirname(__file__), "LatentDynamics"));
Model_Path      : str   = os.path.abspath(os.path.join(os.path.dirname(__file__), "Models"));
sys.path.append(Physics_Path);
sys.path.append(LD_Path);
sys.path.append(Model_Path);

import  logging;

import  torch;
import  numpy;
import  matplotlib.pyplot           as      plt;
import  matplotlib                  as      mpl;
from    sklearn.gaussian_process    import  GaussianProcessRegressor;

from    Physics                     import  Physics;
from    LatentDynamics              import  LatentDynamics;
from    Autoencoder                 import  Autoencoder;
from    Autoencoder_Pair            import  Autoencoder_Pair;
from    SolveROMs                   import  sample_roms;
from    ParameterSpace              import  ParameterSpace;
from    Trainer                     import  Trainer;


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
                             trainer         = None,
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
    assert isinstance(physics, Physics),                "type(physics) = %s" % type(physics);
    assert isinstance(model, torch.nn.Module),          "type(model) = %s" % type(model);
    assert isinstance(latent_dynamics, LatentDynamics), "type(latent_dynamics) = %s" % type(latent_dynamics);
    assert isinstance(gp_list, list),                   "type(gp_list) = %s" % type(gp_list);
    assert len(gp_list)     == latent_dynamics.n_coefs, "len(gp_list) = %d != latent_dynamics.n_coefs = %d" % (len(gp_list), latent_dynamics.n_coefs);
    for i in range(latent_dynamics.n_coefs):
        assert isinstance(gp_list[i], GaussianProcessRegressor), "type(gp_list[%d]) = %s" % (i, type(gp_list[i]));

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
    Predicted_Latent_Trajectories : list[list[numpy.ndarray]] = sample_roms( 
                                                                    model           = model, 
                                                                    physics         = physics, 
                                                                    latent_dynamics = latent_dynamics, 
                                                                    gp_list         = gp_list, 
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
    for i in range(n1):
        for j in range(n2):
            ax.text(i, j, round(values[i, j], 2), fontsize = 10, ha = 'center', va = 'center', color = 'k');


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
    ax.set_xlabel(param_names[0], fontsize = 15, loc='right');
    
    # Position y-axis label at the top-left (horizontal), avoiding overlap with tick labels
    ax.set_ylabel(param_names[1], fontsize = 15, rotation = 0, loc='top', labelpad=10);
    
    ax.set_title(title, fontsize = 25);

    # Save the figure under Higher-Order-LaSDI/Figures (independent of CWD).
    figures_dir: Path = Path(Figures_Path);
    figures_dir.mkdir(parents=True, exist_ok=True);
    save_file_path: str = str(figures_dir / save_file_name);
    fig.savefig(save_file_path);
    
    # Show the plot and then return!
    plt.show();
    return;



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