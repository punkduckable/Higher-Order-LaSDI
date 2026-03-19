# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

# Add LatentDynamics, Physics directories to the search path.
import  sys;
import  os;
LD_Path         : str   = os.path.abspath(os.path.join(os.path.dirname(__file__), "LatentDynamics"));
Physics_Path    : str   = os.path.abspath(os.path.join(os.path.dirname(__file__), "Physics"));
Utils_Path      : str   = os.path.abspath(os.path.join(os.path.dirname(__file__), "Utilities"));
Sample_Path     : str   = os.path.abspath(os.path.join(os.path.dirname(__file__), "Sample"));
sys.path.append(LD_Path); 
sys.path.append(Physics_Path); 
sys.path.append(Utils_Path); 
sys.path.append(Sample_Path);

import  yaml;
import  argparse;
import  logging;
import  time;

import  numpy;
import  torch;
import  matplotlib.pyplot           as      plt;
from    sklearn.gaussian_process    import  GaussianProcessRegressor;
from    pathlib                     import  Path;

import  SolveROMs;
from    EncoderDecoder              import  EncoderDecoder;
from    ParameterSpace              import  ParameterSpace;
from    Physics                     import  Physics;
from    Enums                       import  NextStep;
from    LatentDynamics              import  LatentDynamics;
from    Trainer                     import  Trainer;
from    GaussianProcess             import  fit_gps;
from    Initialize                  import  Initialize_Trainer;
from    Sampler                     import  Sampler;
from    Logging                     import  Initialize_Logger, Log_Dictionary;
from    Plot                        import  Plot_Heatmap2d, Plot_Latent_Trajectories, trainSpace_RelativeErrors_Heatmap;
from    Animate                     import  make_solution_movies;
from    SolveROMs                   import  average_rom;


# Set up the logger.
Initialize_Logger(level = logging.INFO);
LOGGER : logging.Logger = logging.getLogger(__name__);

# Set up the command line arguments
parser = argparse.ArgumentParser(description        = "",
                                 formatter_class    = argparse.RawTextHelpFormatter);
parser.add_argument('--config', 
                    default     = None,
                    required    = True,
                    type        = str,
                    help        = 'config file to run LasDI workflow.\n');



# -------------------------------------------------------------------------------------------------
# Main function
# -------------------------------------------------------------------------------------------------

def main():
    # ---------------------------------------------------------------------------------------------
    # Setup
    # ---------------------------------------------------------------------------------------------

    LOGGER.info("Setting up...");
    timer     : float = time.perf_counter();

    # Load in the argument
    args : argparse.Namespace = parser.parse_args(sys.argv[1:]);
    LOGGER.debug("config file: %s" % args.config);

    # Load the configuration file. 
    with open(args.config, 'r') as f:
        config      = yaml.safe_load(f);
    
    # Report the configuration settings.
    Log_Dictionary(LOGGER = LOGGER, D = config, level = logging.DEBUG);

    # Check if we are loading from a restart or not. If so, load it.
    use_restart         : bool  = config['workflow']['use_restart'];
    restart_filename    : str   = "";
    if (use_restart == True):
        restart_filename : str = config['workflow']['restart_file'];
        LOGGER.info("Loading from restart (%s)" % restart_filename);

        # Set up the restart path under Higher-Order-LaSDI/results (independent of CWD).
        _SRC_DIR: Path = Path(__file__).resolve().parent;      # Higher-Order-LaSDI/src
        _PROJECT_DIR: Path = _SRC_DIR.parent;                 # Higher-Order-LaSDI
        results_dir: Path = _PROJECT_DIR / "results";
        restart_path: str = str(results_dir / restart_filename);
    
    LOGGER.info("Done! Took %fs" % (time.perf_counter() - timer));



    # ---------------------------------------------------------------------------------------------
    # Train!
    # ---------------------------------------------------------------------------------------------

    # Determine what the next step is. If we are loading from a restart, then the restart should
    # have logged then next step. Otherwise, we set the next step to "PickSample", which will 
    # prompt the code to set up the training set of parameters.
    if (use_restart == True):
        if(os.path.isfile(restart_path) == False):
            LOGGER.error("Restart file (%s) does not exist. Stopping the workflow." % restart_path);
            exit();
        
        # TODO(kevin): in long term, we should switch to hdf5 format.
        restart_dict    = numpy.load(restart_path, allow_pickle = True).item();
        next_step       = restart_dict['next_step'];
    else:
        restart_dict    = {};
        next_step       = NextStep.RunSample;
    
    # Initialize the trainer.
    trainer, sampler, param_space, physics, encoder_decoder, latent_dynamics = Initialize_Trainer(config, restart_dict);

    # Calculate and print the number of parameters
    count_parameters(encoder_decoder, latent_dynamics, trainer);

    # Start running steps.
    next_step = step(trainer, sampler, next_step, config);

    # Report the result of training.
    LOGGER.info("Steps completed. Completed %d/%d training steps. The next step step succeeded. Preparing for the next step." % (trainer.restart_iter, trainer.max_iter));



    # ---------------------------------------------------------------------------------------------
    # Save!
    # ---------------------------------------------------------------------------------------------

    # Save!
    LOGGER.info("Saving results to %s" % restart_filename);
    Save(   param_space         = param_space,
            config              = config,
            physics             = physics,
            encoder_decoder     = encoder_decoder, 
            latent_dynamics     = latent_dynamics,
            trainer             = trainer,
            next_step           = next_step,
            restart_filename    = restart_filename);



    # ---------------------------------------------------------------------------------------------
    # Plot Setup
    # ---------------------------------------------------------------------------------------------

    # Set up gaussian processes. 
    encoder_decoder.cpu();

    # Get a GP for each coefficient in the latent dynamics.
    gp_list         : list[GaussianProcessRegressor]    = fit_gps(param_space.train_space, trainer.best_train_coefs);

    # Number of coefficient/ROM samples used for plotting + uncertainty metrics.
    # Most samplers expose this as an attribute; fall back to 20 for custom samplers.
    n_samples_plot  : int = int(getattr(sampler, "n_samples", 20));
    
    # Compute the relative error between the FOM solution and its prediction when we rollout the 
    # IC using the encoder_decoder.
    Max_Rollout_Rel_Error, Max_STD, Rollout_Rel_Error, STD  = SolveROMs.Rollout_Error_and_STD(
                                                                encoder_decoder = encoder_decoder, 
                                                                physics         = physics,
                                                                param_space     = param_space,
                                                                latent_dynamics = latent_dynamics,
                                                                gp_list         = gp_list,
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
                               gp_list         = gp_list,
                               param_grid      = param_space.test_space[i_worst, :].reshape(1, -1),
                               n_samples       = n_samples_plot,
                               U_True          = [trainer.U_Test[i_worst]],
                               t_Grid          = [trainer.t_Test[i_worst]],
                               file_prefix     = config["physics"]["type"],
                               trainer         = trainer,
                               figsize         = (15, 13));


    # Plot the relative error between the trajectories for the final training set.
    if(config['workflow']['plot_train_rel_errors'] == True):
        trainSpace_RelativeErrors_Heatmap(  trainer     = trainer, 
                                            param_space = trainer.param_space, 
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
        Zi_mean_np     : list[numpy.ndarray]   = average_rom(   encoder_decoder = encoder_decoder, # n_IC element list whose j'th element has shape (n_t(i), n_z)
                                                                physics         = physics, 
                                                                latent_dynamics = latent_dynamics, 
                                                                gp_list         = gp_list, 
                                                                param_grid      = param_worst, 
                                                                t_Grid          = [t_worst],
                                                                trainer         = trainer)[0];   # shape = (n_t, n_IC, n_z)

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

    if(param_space.n_p == 2):
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

            Plot_Heatmap2d(     values          = Max_Recon_Rel_Error[:, d].reshape(param_space.test_grid_sizes) * 100, 
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

            Plot_Heatmap2d(     values          = Max_Rollout_Rel_Error[:, d].reshape(param_space.test_grid_sizes) * 100, 
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


            Plot_Heatmap2d( values          = Max_STD[:, d].reshape(param_space.test_grid_sizes) * 100,
                            param_space     = param_space, 
                            title           = title,
                            save_file_name  = save_file_name);


    # All done!
    LOGGER.info("All done!");
    return;





# -------------------------------------------------------------------------------------------------
# Step
# -------------------------------------------------------------------------------------------------

def step(trainer        : Trainer,
         sampler        : Sampler,  
         next_step      : NextStep, 
         config         : dict) -> NextStep:
    """
    Runs the next step of the training procedure and recursively continues until the workflow is
    complete or encounters a failure. The full cycle is:

        RunSample → Train → PickSample → RunSample → Train → PickSample → ... → Complete

    When loading from a restart, pass in the `next_step` saved in the restart file; this function
    will pick up exactly where the previous run left off and run to completion.

    
    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------
    
    trainer : Trainer
        A Trainer class object that we use when training the encoder_decoder for a particular 
        instance of the settings.
    
    sampler : Sampler
        The sampler object used to select the "worst" testing parameter combination during greedy 
        sampling.

    next_step : NextStep
        The step to execute first. When restarting, this should be loaded from the restart file.

    config : dict
        This should be a dictionary that we loaded from a .yml file. It should house all the 
        settings we expect to use to generate the data and train the encoder_decoder.


    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    next_step : NextStep
        The step that would come next (informational; the workflow has already stopped). 
    """


    # ---------------------------------------------------------------------------------------------
    # Run the next step 
    # ---------------------------------------------------------------------------------------------

    LOGGER.info("Running %s" % next_step);
    if (next_step is NextStep.Train):
        # If our next step is to train, then let's train! This will set trainer.restart_iter to 
        # the iteration number of the last iterating training.
        trainer.train();


        # Next, check if the restart_iter falls below the "max_greedy_iter". The later is the last
        # iteration at which we want to run greedy sampling. If the restart_iter is below the 
        # max_greedy_iter, then we should pick a new sample (perform greedy sampling). Otherwise, 
        # we should continue training.
        if (trainer.restart_iter <= trainer.max_greedy_iter):
            next_step = NextStep.PickSample;
        else:
            next_step = NextStep.Train;


    elif (next_step is NextStep.PickSample):
        # Use greedy sampling to pick that sample. Note that if the training set is empty, this 
        # function does nothing.
        next_step = sampler.Sample(trainer);


    elif (next_step is NextStep.RunSample):
        # Generate the trajectories for all new testing and training parameters. Append these new
        # trajectories to trainer's U_Train and U_Test attributes.
        next_step = sampler.Generate_Training_Data(trainer);
        
        if(config["workflow"]["plot_train_rel_errors"] == True):
            trainSpace_RelativeErrors_Heatmap(  trainer     = trainer, 
                                                param_space = trainer.param_space, 
                                                file_prefix = "initial_" + config["physics"]["type"]);


    else:
        raise RuntimeError("Unknown next step!");
    


    # ---------------------------------------------------------------------------------------------
    # Wrap up
    # ---------------------------------------------------------------------------------------------

    # Check if training has finished. Recall that a trainer object's restart_iter member holds the 
    # iteration number of the last iteration in the last round of training. Likewise, its 
    # "max_iter" member specifies the total number of iterations we want to train for. Thus, if 
    # restart_iter goes above max_iter, then it is time to stop running steps. 
    if(trainer.restart_iter >= trainer.max_iter):
        return next_step;
        
    # Otherwise, continue the workflow.
    LOGGER.info("Next step is: %s" % next_step);
    next_step = step(trainer, sampler, next_step, config);

    # All done!
    return next_step;





# -------------------------------------------------------------------------------------------------
# Save
# -------------------------------------------------------------------------------------------------

def Save(   param_space         : ParameterSpace, 
            config              : dict,
            physics             : Physics, 
            encoder_decoder     : EncoderDecoder, 
            latent_dynamics     : LatentDynamics,
            trainer             : Trainer, 
            next_step           : NextStep, 
            restart_filename    : str               = "") -> None:
    """
    This function saves a trained encoder_decoder, trainer, latent dynamics, etc. You should call 
    this function after running the LASDI algorithm.


    
    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    param_space : ParameterSpace 
        holds the training and testing parameter combinations.
    
    config : dict
        This should be a dictionary that we loaded from a .yml file. It should house all the 
        settings we expect to use to generate the data and train the encoder_decoder.

    physics : Physics
        defines the FOM model. We can use it to fetch the initial conditions and FOM solution for
        a particular combination of parameter values. physics, latent_dynamics, and encoder_decoder 
        should have the same number of initial conditions.

    encoder_decoder : EncoderDecoder
        maps between the FOM and ROM spaces. physics, latent_dynamics, and encoder_decoder should 
        have the same number of initial conditions.

    latent_dynamics : LatentDynamics 
        defines the dynamics in encoder_decoder's latent space. physics, latent_dynamics, and 
        encoder_decoder should have the same number of initial conditions.

    trainer : Trainer
        trains encoder_decoder using physics to define the FOM, latent_dynamics to define the ROM, 
        and encoder_decoder to connect them.

    next_step : NextStep
        An enumeration indicating the next step (should we continue training). This should 
        have been returned by the final call to the step function.

    restart_filename : str
        If we loaded from a restart, then this is the name of the restart we loaded.
        Otherwise, if we did not load from a restart, this should be an empty string.


    
    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    Nothing!
    """

    # Checks.
    n_IC    : int   = latent_dynamics.n_IC;
    assert encoder_decoder.n_IC     == n_IC, "encoder_decoder.n_IC = %d != n_IC = %d" % (encoder_decoder.n_IC, n_IC);
    assert(physics.n_IC             == n_IC);


    # Save restart (or final) file.
    date        = time.localtime();
    date_str    = "{month:02d}_{day:02d}_{year:04d}_{hour:02d}_{minute:02d}";
    date_str    = date_str.format(month     = date.tm_mon, 
                                  day       = date.tm_mday, 
                                  year      = date.tm_year, 
                                  hour      = date.tm_hour, 
                                  minute    = date.tm_min);
    
    # Set up the restart filename.
    if(len(restart_filename) > 0):
        # Extract the non-extension portion of the restart filename.
        restart_filename_no_ext : str = restart_filename.split('.')[0];

        # now append the new date to the restart filename.
        restart_filename = restart_filename_no_ext + '__' + date_str + '.npy';
    else:
        restart_filename : str = config["physics"]["type"] + '_' + date_str + '.npy';
    
    # Set up the restart path.
    # Use an absolute results directory under the project root (Higher-Order-LaSDI/results),
    # independent of the current working directory.
    from pathlib import Path;
    if hasattr(trainer, "path_results"):
        results_dir = Path(trainer.path_results);
    else:
        src_dir = Path(__file__).resolve().parent;
        project_dir = src_dir.parent;
        results_dir = project_dir / "results";
    results_dir.mkdir(parents=True, exist_ok=True);
    restart_path = str(results_dir / restart_filename);

    # Build the restart save dictionary and then save it.
    restart_dict = {'parameter_space'   : param_space.export(),
                    'physics'           : physics.export(),
                    'encoder_decoder'   : encoder_decoder.export(),
                    'latent_dynamics'   : latent_dynamics.export(),
                    'trainer'           : trainer.export(),
                    'timestamp'         : date_str,
                    'next_step'         : next_step};
    numpy.save(restart_path, restart_dict);

    # All done!
    return;





# -------------------------------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------------------------------

def count_parameters(   encoder_decoder : EncoderDecoder, 
                        latent_dynamics : LatentDynamics,
                        trainer         : Trainer) -> None:
    """
    Calculate and print the number of parameters in the encoder_decoder, latent dynamics, and 
    trainer.
    
    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------
    
    encoder_decoder : EncoderDocoder
        The neural network encoder_decoder.
        
    latent_dynamics : LatentDynamics
        The latent dynamics encoder_decoder.
        
    trainer : Trainer
        The trainer object which may contain learnable coefficients.
    """
    
    # Count encoder_decoder parameters
    total_params        = 0;
    trainable_params    = 0;
    
    for param in encoder_decoder.parameters():
        total_params += param.numel();
        if param.requires_grad:
            trainable_params += param.numel();
    

    # Count learnable coefficients from trainer (only applies if we are learning the latent 
    #dynamics coefficients)
    coef_params = 0;
    if hasattr(trainer, 'test_coefs') and trainer.test_coefs is not None:
        coef_params = trainer.test_coefs.numel();
    
    # Print summary
    LOGGER.info("=" * 80);
    LOGGER.info("EncoderDecoder Parameter Summary");
    LOGGER.info("=" * 80);
    LOGGER.info("EncoderDecoder:");
    LOGGER.info("  Total parameters:      {:,}".format(total_params));
    LOGGER.info("  Trainable parameters:  {:,}".format(trainable_params));
    LOGGER.info("  Non-trainable:         {:,}".format(total_params - trainable_params));
    
    if coef_params > 0:
        LOGGER.info("Learnable Coefficients:");
        LOGGER.info("  Total parameters:      {:,}".format(coef_params));
    
    grand_total = total_params + coef_params;
    grand_trainable = trainable_params + coef_params;
    
    LOGGER.info("=" * 80);
    LOGGER.info("Grand Total:");
    LOGGER.info("  Total parameters:      {:,}".format(grand_total));
    LOGGER.info("  Trainable parameters:  {:,}".format(grand_trainable));
    LOGGER.info("=" * 80);
    
    return;

if __name__ == "__main__":
    main();
