# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

import  sys;
import  os;
from    pathlib                     import  Path;
import  logging;
import  argparse;

import  numpy;
import  torch;
import  matplotlib.pyplot           as      plt;

# Expose `src/` as the import root for the repository sub-libraries.
PROJECT_DIR         : Path  = Path(__file__).resolve().parent.parent;
SRC_Path            : str   = str(PROJECT_DIR / "src");
sys.path.append(SRC_Path);

from    Plotting.Metrics            import  Generate_Heatmap_Data;
from    Plotting.Plot               import  Plot_Heatmap, Plot_Latent_Trajectories;
from    Plotting.Plot               import  Plot_Meltpool_Dimensions, trainSpace_RelativeErrors_Heatmap;
from    Plotting.Animate            import  make_solution_movies;
from    Interpolate                 import  Interpolate;
from    Interpolate.Rollouts        import  Mean_Rollout; 
from    Utilities.Logging           import  Initialize_Logger;
from    Initialize                  import  Initialize_Trainer;

# Set up the command line arguments
parser = argparse.ArgumentParser(description        = "",
                                 formatter_class    = argparse.RawTextHelpFormatter);
parser.add_argument('--artifact', 
                    default     = None,
                    required    = True,
                    type        = str,
                    help        = 'the saved model/config/data from a training run.\n');
parser.add_argument('--relative-error-plot',
                    action      = "store_true",
                    help        = 'If true, we generate a heatmap of the relative error between FOM solutions in the training set.')


# Set up the logger.
Initialize_Logger(level = logging.INFO);
LOGGER : logging.Logger = logging.getLogger(__name__);


# -------------------------------------------------------------------------------------------------
# Plotting script
# -------------------------------------------------------------------------------------------------


def analyze_experiment(artifact_path : str, make_train_rel_error_heatmap: bool = False) -> None:
    """
    Loads a saved experiment artifact and runs the post-processing/plotting portion of the LaSDI
    workflow.

    The artifact should be the ``.npy`` file written by ``scripts/run_experiment.py`` after
    training. It must contain the serialized config, parameter space, physics object,
    encoder_decoder, latent dynamics model, and trainer data needed to reproduce the plotting
    state. This function intentionally disables the restart checkpoint side effect when loading the
    artifact because analysis should not overwrite the active training checkpoint.


    
    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    artifact_path : str
        Path to the saved experiment artifact. If this is a relative path that does not exist from
        the current working directory, we also look for it under ``<project>/results``.
    make_train_rel_error_heatmap:
        If true, we generate a heatmap of the relative error between FOM solutions in the training 
        set.

    
    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    Nothing!
    """ 

    artifact_file : Path = Path(artifact_path).expanduser();
    if((artifact_file.is_absolute() == False) and (artifact_file.is_file() == False)):
        artifact_file = PROJECT_DIR / "results" / artifact_file;
    artifact_file = artifact_file.resolve();
    if(artifact_file.is_file() == False):
        raise FileNotFoundError("Artifact file does not exist: %s" % artifact_file);

    LOGGER.info("Loading artifact from: %s" % artifact_file);

    # Load the saved artifact and extract the config.
    restart_dict : dict = numpy.load(str(artifact_file), allow_pickle = True).item();
    if("config" in restart_dict):
        config : dict = restart_dict["config"];
    elif(("trainer" in restart_dict) and ("config" in restart_dict["trainer"])):
        LOGGER.warning("Artifact has no top-level config; falling back to trainer config.");
        config : dict = restart_dict["trainer"]["config"];
    else:
        raise KeyError("Artifact must contain either restart_dict['config'] or restart_dict['trainer']['config']");

    # Load the trainer, sampler, parameter space, physics, encoder_decoder, and latent dynamics.
    # This mirrors restart loading in run_experiment.py, except it does not write a checkpoint.
    LOGGER.info("Setting up trainer/sampler/physics/data...");
    trainer, sampler, param_space, physics, encoder_decoder, latent_dynamics = Initialize_Trainer(
        config,
        restart_dict,
        make_restart_checkpoint = False,
    );
    LOGGER.info("Done loading!");


    # ---------------------------------------------------------------------------------------------
    # Plot Setup
    # ---------------------------------------------------------------------------------------------

    # Set up coefficient interpolator. 
    encoder_decoder.cpu();
    trainer._check_train_coefficients();
    interpolator : Interpolate = Interpolate(latent_dynamics.train_coefs);

    # Number of coefficient/ROM samples used for plotting + uncertainty metrics.
    # Most samplers expose this as an attribute; fall back to 20 for custom samplers.
    n_samples_plot  : int = int(getattr(sampler, "n_samples", 20));
    
    # Compute the relative error between the FOM solution and its prediction when we rollout the 
    # IC using the encoder_decoder.
    Max_Rollout_Rel_Error, Max_STD, Rollout_Rel_Error, STD, coef_means, coef_stds  = Generate_Heatmap_Data(
                                                                                        encoder_decoder = encoder_decoder, 
                                                                                        physics         = physics,
                                                                                        param_space     = param_space,
                                                                                        latent_dynamics = latent_dynamics,
                                                                                        interpolator    = interpolator,
                                                                                        t_Test          = trainer.t_Test,
                                                                                        U_Test          = trainer.U_Test,
                                                                                        n_samples       = n_samples_plot,
                                                                                        trainer         = trainer);

    # Find the index of the parameter combination that has the largest relative error; we unravel the 
    # index to get the row, column number of the maximum entry of Max_Rollout_Rel_Error, then keep
    # the row number.
    i_worst        : int   = int(numpy.unravel_index(numpy.argmax(Max_Rollout_Rel_Error), Max_Rollout_Rel_Error.shape)[0]);

    # Plot the latent trajectories for the i_worst'th element of the test set.
    Plot_Latent_Trajectories(  physics         = physics,
                               encoder_decoder = encoder_decoder,
                               latent_dynamics = latent_dynamics,
                               interpolator    = interpolator,
                               param_grid      = param_space.test_space[i_worst, :].reshape(1, -1),
                               n_samples       = n_samples_plot,
                               U_True          = [trainer.U_Test[i_worst]],
                               t_Grid          = [trainer.t_Test[i_worst]],
                               file_prefix     = config["physics"]["type"],
                               trainer         = trainer,
                               figsize         = (15, 13));


    # Plot the relative error between the trajectories for the final training set.
    if(make_train_rel_error_heatmap == True):
        trainSpace_RelativeErrors_Heatmap(  trainer     = trainer, 
                                            param_space = param_space, 
                                            file_prefix = config["physics"]["type"]);



    # ---------------------------------------------------------------------------------------------
    # Plot relative error trajectories
    # ---------------------------------------------------------------------------------------------

    # Setup
    Recon_Rel_Error         : list[list[numpy.ndarray]] = [];
    Max_Recon_Rel_Error     : numpy.ndarray             = numpy.zeros((param_space.n_test(), physics.n_IC));

    # Cycle through the combinations of parameter values.
    for i in range(param_space.n_test()):
        # Reconstruct the FOM solution, store it in a list.
        LOGGER.debug("Reconstructing the FOM solution for parameter combination %d (%s)" % (i, str(param_space.test_space[i])));
        ith_Reconstruction : torch.Tensor | tuple[torch.Tensor, torch.Tensor] = encoder_decoder(*trainer.U_Test[i]);
        if(isinstance(ith_Reconstruction, tuple)):
            ith_Reconstruction = list(ith_Reconstruction);
        elif(isinstance(ith_Reconstruction, torch.Tensor)):
            ith_Reconstruction = [ith_Reconstruction];
        else:
            raise ValueError("ith_Encoding is not a tuple or a torch.Tensor");
    
        # Setup for the i'th combination of parameter values.
        n_IC                    : int                   = physics.n_IC;
        ith_Recon_Rel_Error     : list[numpy.ndarray]   = [];
        n_t_i                   : int                   = trainer.t_Test[i].shape[0];

        # Cycle through the ICs.
        for j in range(n_IC):
            # Setup a tensor to hold the relative error for the j'th IC and the i'th combination of 
            # parameter values.
            ij_Recon_Rel_Error      : numpy.ndarray = numpy.zeros(n_t_i);

            # Fetch the reconstruction and true solution.
            if hasattr(trainer, "has_normalization") and trainer.has_normalization():
                ij_Reconstruction = trainer.denormalize_tensor(ith_Reconstruction[j], j).detach().numpy();   # physical units
                ij_True           = trainer.denormalize_tensor(trainer.U_Test[i][j], j).detach().numpy();    # physical units
            else:
                ij_Reconstruction   : numpy.ndarray = ith_Reconstruction[j].detach().numpy();   # shape = (n_t_i, physics.Frame_Shape)
                ij_True             : numpy.ndarray = trainer.U_Test[i][j].detach().numpy();    # shape = (n_t_i, physics.Frame_Shape)

            # Compute the std of each component of the true solution.
            ij_True_std         : float          = numpy.std(ij_True);

            # For each frame, compute the relative error between the true and predicted FOM solutions.
            # We normalize the error by the std of the true solution.
            for k in range(n_t_i):
                ij_Recon_Rel_Error[k] = numpy.mean(numpy.abs(ij_Reconstruction[k, ...] - ij_True[k, ...]))/ij_True_std;

            # Append the relative error for the j'th IC.
            ith_Recon_Rel_Error.append(ij_Recon_Rel_Error);

            # Compute the maximum relative error for the j'th time derivative of the solution for 
            # the i'th combination of parameter values.
            Max_Recon_Rel_Error[i, j] = numpy.max(ij_Recon_Rel_Error);
        
        # Append the relative error for the i'th combination of parameter values.
        Recon_Rel_Error.append(ith_Recon_Rel_Error);

    

    # First, plot the rollout relative error.
    for i in range(physics.n_IC):
        plt.figure();
        plt.plot(trainer.t_Test[i_worst], Rollout_Rel_Error[i_worst][i]);
        plt.xlabel("time (s)");
        plt.ylabel("Relative Error");
        plt.grid(True, which = "both", alpha = 0.25);

        if(i == 0):     
            title_str       : str = "Relative Error of the rollout of U for %s"           % str(param_space.test_space[i_worst]);
            save_file_name  : str = config["physics"]["type"] + "_U_Rollout_Rel_Error_%s.png"                   % str(param_space.test_space[i_worst]);   
        elif(i == 1):   
            title_str       : str = "Relative Error of the rollout of D_t U for %s"       % str(param_space.test_space[i_worst]);
            save_file_name  : str = config["physics"]["type"] + "_Dt_U_Rollout_Rel_Error_%s.png"                % str(param_space.test_space[i_worst]);
        else:           
            title_str       : str = "Relative Error of the rollout of D_t^%d U for %s"    % (i, str(param_space.test_space[i_worst]));
            save_file_name  : str = config["physics"]["type"] + "_Dt^%d_U_Rollout_Rel_Error_%s.png"             % (i, str(param_space.test_space[i_worst]));

        # Plot the figure.
        plt.title(title_str);
    
        # Now save the figure.
        figures_dir: Path = Path(__file__).resolve().parent.parent / "Figures";
        figures_dir.mkdir(parents=True, exist_ok=True);
        plt.savefig(str(figures_dir / save_file_name));


    # Next, plot the reconstruction relative error.
    for i in range(physics.n_IC):
        plt.figure();
        plt.plot(trainer.t_Test[i_worst], Recon_Rel_Error[i_worst][i]);
        plt.xlabel("time (s)");
        plt.ylabel("Relative Error");
        
        if(i == 0):     
            title_str       : str = "Relative Error of the reconstruction of U for %s"        % str(param_space.test_space[i_worst]);
            save_file_name  : str = config["physics"]["type"] + "_U_Recon_Rel_Error_%s.png"                         % str(param_space.test_space[i_worst]);   
        elif(i == 1):   
            title_str       : str = "Relative Error of the reconstruction of D_t U for %s"    % str(param_space.test_space[i_worst]);
            save_file_name  : str = config["physics"]["type"] + "_Dt_U_Recon_Rel_Error_%s.png"                      % str(param_space.test_space[i_worst]);
        else:           
            title_str       : str = "Relative Error of the reconstruction of D_t^%d U for %s" % (i, str(param_space.test_space[i_worst]));
            save_file_name  : str = config["physics"]["type"] + "_Dt^%d_U_Recon_Rel_Error_%s.png"                   % (i, str(param_space.test_space[i_worst]));

        # Plot the figure.
        plt.title(title_str);
    
        # Now save the figure.
        figures_dir: Path = Path(__file__).resolve().parent.parent / "Figures";
        figures_dir.mkdir(parents=True, exist_ok=True);
        plt.savefig(str(figures_dir / save_file_name));
    
    plt.show();



    # ---------------------------------------------------------------------------------------------
    # Make animations of the solution, its reconstruction, and the error between the two.
    # Also, for thermal simulations, plot the heatpool dimensions.
    # ---------------------------------------------------------------------------------------------

    # Make movies for the mean predicted solution, true solution, and error for the i_worst'th 
    # combination of parameters.

    # If X_Positions has the form (2, N_Positions) or (3, N_Positions), then the solution must 
    # either be a scalar field on a 2d or 3d domain, or a 2d/3d vector field in a 2d/3d domain. 
    # In these cases, we can make an animation of the solution.... let's do that!
    if((len(physics.X_Positions.shape) == 2) and (physics.X_Positions.shape[0] in (2, 3))):
        
        # First, generate latent trajectories for the i_worst'th element of the test set.
        LOGGER.debug("Generating trajectory plot for testing combination %d: %s" % (i_worst, param_space.test_space[i_worst]));

        # Generate the solution trajectory using the mean for the posterior distribution.
        param_worst    : numpy.ndarray         = param_space.test_space[i_worst, :].reshape(1, -1);
        t_worst        : torch.Tensor          = trainer.t_Test[i_worst];                          # shape = (n_t)
        U_True_worst   : list[torch.Tensor]    = trainer.U_Test[i_worst];                          # length = n_IC        
        Zi_mean_np     : list[numpy.ndarray]   = Mean_Rollout(  encoder_decoder = encoder_decoder, # n_IC element list whose j'th element has shape (n_t(i), n_z)
                                                                physics         = physics, 
                                                                latent_dynamics = latent_dynamics, 
                                                                interpolator    = interpolator, 
                                                                param_grid      = param_worst, 
                                                                t_Grid          = [t_worst],
                                                                trainer         = trainer)[0];

        # Map Zi_mean_np to a tensor and then decode.
        Zi_mean     : list[torch.Tensor]    = [];
        for i in range(len(Zi_mean_np)):
            Zi_mean.append(torch.Tensor(Zi_mean_np[i]));
        U_Pred_worst : list[torch.Tensor]          = list(encoder_decoder.Decode(*Zi_mean));             # length = n_IC

        # Make a movie for each derivative of the solution.
        n_IC        : int                   = physics.n_IC;
        for i in range(n_IC):
            if(i == 0):
                prefix : str = "%s_U_%s"        % (config["physics"]["type"], str(param_space.test_space[i_worst]));
            elif(i == 1):
                prefix : str = "%s_Dt_U_%s"     % (config["physics"]["type"], str(param_space.test_space[i_worst]));
            else:
                prefix : str = "%s_Dt^%d_U_%s"  % (config["physics"]["type"], i, str(param_space.test_space[i_worst]));

            # Make the movie.
            # Check normalization status and apply denormalization appropriately.
            has_norm = hasattr(trainer, "has_normalization") and trainer.has_normalization();
            LOGGER.info(f"Animation for IC {i}: has_normalization = {has_norm}");
            
            if has_norm:
                LOGGER.info(f"  U_True_worst[{i}] range before denorm: [{U_True_worst[i].min().item():.3e}, {U_True_worst[i].max().item():.3e}]");
                LOGGER.info(f"  U_Pred_worst[{i}] range before denorm: [{U_Pred_worst[i].min().item():.3e}, {U_Pred_worst[i].max().item():.3e}]");
                
                # Both U_True_worst and U_Pred_worst should be in normalized units
                U_i_true_np = trainer.denormalize_tensor(U_True_worst[i], i).detach().numpy();
                U_i_pred_np = trainer.denormalize_tensor(U_Pred_worst[i], i).detach().numpy();
                
                LOGGER.info(f"  U_true_np range after denorm: [{U_i_true_np.min():.3e}, {U_i_true_np.max():.3e}]");
                LOGGER.info(f"  U_pred_np range after denorm: [{U_i_pred_np.min():.3e}, {U_i_pred_np.max():.3e}]");
            else:
                # WARNING: If normalization is disabled but data was normalized, this will show normalized values
                LOGGER.warning(f"Normalization is disabled or not configured properly!");
                LOGGER.warning(f"  If training data was normalized, animations will show NORMALIZED (not physical) units.");
                LOGGER.warning(f"  U_True_worst[{i}] range: [{U_True_worst[i].min().item():.3e}, {U_True_worst[i].max().item():.3e}]");
                LOGGER.warning(f"  U_Pred_worst[{i}] range: [{U_Pred_worst[i].min().item():.3e}, {U_Pred_worst[i].max().item():.3e}]");
                
                U_i_true_np = U_True_worst[i].detach().numpy();
                U_i_pred_np = U_Pred_worst[i].detach().numpy();

            # Flatten predictions so that they have shape (N_t, C, n_nodes) for make_solution_movies.
            n_nodes : int   = int(physics.X_Positions.shape[1]);
            n_t     : int   = int(t_worst.shape[0]);

            def _flatten_for_movie(U: numpy.ndarray) -> numpy.ndarray:
                assert U.shape[0] == n_t, \
                    "U.shape = %s, U.shape[0] must be %d (number of time steps)" % (str(U.shape), n_t);

                # Already flattened scalar field.
                if U.ndim == 2:
                    assert U.shape[1] == n_nodes, \
                        "U.shape = %s, expected second dim to be n_nodes=%d" % (str(U.shape), n_nodes);
                    return U[:, None, :];  # (n_t, 1, n_nodes)

                # Already in (n_t, C, n_nodes) form.
                if U.ndim == 3:
                    assert U.shape[2] == n_nodes, \
                        "U.shape = %s, expected last dim to be n_nodes=%d" % (str(U.shape), n_nodes);
                    return U

                # CNN / gridded case: (n_t, C, ...spatial...)
                assert U.ndim >= 4, "U.shape = %s, expected at least 4D tensor for gridded data" % str(U.shape);
                C = int(U.shape[1]);
                spatial_prod = int(numpy.prod(U.shape[2:]));
                assert spatial_prod == n_nodes, \
                    "U.shape = %s; prod(U.shape[2:]) = %d, but n_nodes = %d" % (str(U.shape), spatial_prod, n_nodes);
                return U.reshape(n_t, C, n_nodes);

            U_i_true_np = _flatten_for_movie(U_i_true_np);
            U_i_pred_np = _flatten_for_movie(U_i_pred_np);

            # For Thermal, compute and plot melt pool length/width/depth for the same "worst"
            # parameter combination. Only the state U (not time derivatives) has a melt pool
            # interpretation. Arrays are already denormalized above, so the threshold is in
            # physical temperature units.
            if (i == 0) and (config["physics"]["type"] == "Thermal"):
                assert "Thermal" in config["physics"], "Thermal physics config missing `Thermal` section";
                assert "threshold" in config["physics"]["Thermal"], "Thermal physics config missing `threshold`";
                Plot_Meltpool_Dimensions(t_Grid      = t_worst,
                                         U_True      = U_i_true_np,
                                         U_Pred      = U_i_pred_np,
                                         node_coords = physics.X_Positions,
                                         threshold   = float(config["physics"]["Thermal"]["threshold"]),
                                         param       = param_space.test_space[i_worst],
                                         file_prefix = config["physics"]["type"],
                                         n_for_avg   = 3,
                                         show_plot   = False);

            if U_i_true_np.shape[1] == 1:
                data    = U_i_true_np;
            else:
                data    = numpy.linalg.norm(U_i_true_np, axis = 1);
            vmin    = data.min();
            vmax    = data.max();
            if(hasattr(physics, "threshold")):
                threshold = physics.threshold;
            else:
                threshold = None;
            make_solution_movies(U_True         = U_i_true_np, 
                                 U_Pred         = U_i_pred_np, 
                                 X              = physics.X_Positions, 
                                 T              = t_worst.detach().numpy(),
                                 vmin           = vmin,
                                 vmax           = vmax,
                                 fname_prefix   = prefix, 
                                 threshold      = threshold);
    


    # ---------------------------------------------------------------------------------------------
    # Plot the heatmaps
    # ---------------------------------------------------------------------------------------------

    if(param_space.n_p in (2, 3)):
        n_IC : int = latent_dynamics.n_IC;
        
        # Plot maximum (across the frames) relative error between a frame and its reconstruction 
        # under the autoencoder. Do this for each combination of parameter values and derivative 
        # of the FOM solution.
        for d in range(n_IC):
            if(d == 0):
                # NOTE: The implementation normalizes by a single global std for this parameter
                # combination and derivative (computed over all time steps + spatial nodes), not a
                # per-time/per-node std.
                title           : str   = r'$\text{max}_{k} \frac{\text{mean}_{j} \left| u_{\text{Pred}}(t_k, x_j) - u_{\text{True}}(t_k, x_j) \right|} {\sigma \left( u_{\text{True}} \right) }$';
                save_file_name  : str   = config["physics"]["type"] + "_U_Reconstruction_Relative_Error_Heatmap.png";
            elif(d == 1):
                title           : str   = r'$\text{max}_{k} \frac{\text{mean}_{j} \left| \frac{d}{dt}u_{\text{Pred}}(t_k, x_j) - \frac{d}{dt}u_{\text{True}}(t_k, x_j) \right|} {\sigma \left( \frac{d}{dt}u_{\text{True}} \right) }$';
                save_file_name  : str   = config["physics"]["type"] + "_Dt_U_Reconstruction_Relative_Error_Heatmap.png";
            else:
                title           : str   = r'$\text{max}_{k} \frac{\text{mean}_{j} \left| \frac{d^{%d}}{dt^{%d}}u_{\text{Pred}}(t_k, x_j) - \frac{d^{%d}}{dt^{%d}}u_{\text{True}}(t_k, x_j) \right|} {\sigma \left( \frac{d^{%d}}{dt^{%d}}u_{\text{True}} \right) }$' % (d, d, d, d, d, d);
                save_file_name  : str   = config["physics"]["type"] + "_Dt^%d_U_Reconstruction_Relative_Error_Heatmap.png" % d;

            Plot_Heatmap(   values          = Max_Recon_Rel_Error[:, d].reshape(param_space.test_grid_sizes) * 100, 
                            param_space     = param_space,
                            title           = title, 
                            save_file_name  = save_file_name);
        

        # Plot maximum (across the frames) relative error between a frame and the frame that the 
        # encoder_decoder predicts when we rollout the IC for the corresponding combination of 
        # parameter values. Do this for each combination of parameter values and derivative of 
        # the FOM solution.
        for d in range(n_IC):
            if(d == 0):
                # NOTE: The implementation normalizes by a single global std for this parameter
                # combination and derivative (computed over all time steps + spatial nodes), not a
                # per-time/per-node std.
                title           : str   = r'$\text{max}_{k} \frac{\text{mean}_{j} \left| u_{\text{Rollout}}(t_k, x_j) - u_{\text{True}}(t_k, x_j) \right|} {\sigma \left( u_{\text{True}} \right) }$';
                save_file_name  : str   = config["physics"]["type"] + "_U_Rollout_Rel_Error_Heatmap.png";
            elif(d == 1):
                title           : str   = r'$\text{max}_{k} \frac{\text{mean}_{j} \left| \frac{d}{dt}u_{\text{Rollout}}(t_k, x_j) - \frac{d}{dt}u_{\text{True}}(t_k, x_j) \right|} {\sigma \left( \frac{d}{dt}u_{\text{True}} \right) }$';
                save_file_name  : str   = config["physics"]["type"] + "_Dt_U_Rollout_Rel_Error_Heatmap.png";
            else:
                title           : str   = r'$\text{max}_{k} \frac{\text{mean}_{j} \left| \frac{d^{%d}}{dt^{%d}}u_{\text{Rollout}}(t_k, x_j) - \frac{d^{%d}}{dt^{%d}}u_{\text{True}}(t_k, x_j) \right|} {\sigma \left( \frac{d^{%d}}{dt^{%d}}u_{\text{True}} \right) }$' % (d, d, d, d, d, d);
                save_file_name  : str   = config["physics"]["type"] + "_Dt^%d_U_Rollout_Rel_Error_Heatmap.png" % d;

            Plot_Heatmap(   values          = Max_Rollout_Rel_Error[:, d].reshape(param_space.test_grid_sizes) * 100, 
                            param_space     = param_space,
                            title           = title, 
                            save_file_name  = save_file_name);

        # Plot the std of the component of the frame with the largest std (across the samples) in 
        # the reconstruction of that component of that frame. Do this for each combination of 
        # parameter values and derivative of the FOM solution.
        for d in range(n_IC):
            if(d == 0):
                title           : str   = r'$\text{max}_{i, j} \sigma_{k \in \{1, \ldots, %d\}} \left[ u_{\text{Rollout}}(k)(t_i, x_j) \right]$' % n_samples_plot;
                save_file_name  : str   = config["physics"]["type"] + "_U_STD_Heatmap.png";
            elif(d == 1):
                title           : str   = r'$\text{max}_{i, j} \sigma_{k \in \{ 1, \ldots, %d\}} \left[\frac{d}{dt}u_{\text{Rollout}}(k)(t_i, x_j) \right]$' % (n_samples_plot);
                save_file_name  : str   = config["physics"]["type"] + "_Dt_U_STD_Heatmap.png";      
            else:
                title           : str   = r'$\text{max}_{i, j} \sigma_{k \in \{ 1, \ldots, %d\}} \left[\frac{d^{%d}}{dt^{%d}}u_{\text{Rollout}}(k)(t_i, x_j) \right]$' % (n_samples_plot, d, d);
                save_file_name  : str   = config["physics"]["type"] + "_Dt^%d_U_STD_Heatmap.png" % d;


            Plot_Heatmap(   values          = Max_STD[:, d].reshape(param_space.test_grid_sizes) * 100,
                            param_space     = param_space, 
                            title           = title,
                            save_file_name  = save_file_name);


        # Plot the mean and std of each coefficient at each testing parameter.
        for d in range(latent_dynamics.n_coefs):
            title           : str   = "Coefficient %d mean" % d;
            save_file_name  : str   = config["physics"]["type"] + "Coefficient_%d_mean.png" % d;

            Plot_Heatmap(   values          = coef_means[:, d].reshape(param_space.test_grid_sizes),
                            param_space     = param_space, 
                            title           = title,
                            save_file_name  = save_file_name,
                            show_plot       = False,
                            annotate_cells  = False);

            title           : str   = "Coefficient %d std" % d;
            save_file_name  : str   = config["physics"]["type"] + "Coefficient_%d_std.png" % d;

            Plot_Heatmap(   values          = coef_stds[:, d].reshape(param_space.test_grid_sizes),
                            param_space     = param_space, 
                            title           = title,
                            save_file_name  = save_file_name,
                            show_plot       = False,
                            annotate_cells  = False);
    else:
        LOGGER.warning("Skipping parameter-space heatmaps because param_space.n_p = %d; Plot_Heatmap supports only 2D or 3D parameter spaces." % param_space.n_p);

    # All done!
    LOGGER.info("All done!");
    return;




def main():
    args : argparse.Namespace = parser.parse_args(sys.argv[1:]);
    analyze_experiment(artifact_path = args.artifact, make_train_rel_error_heatmap = args.relative_error_plot);
    return;


if __name__ == "__main__":
    main();
