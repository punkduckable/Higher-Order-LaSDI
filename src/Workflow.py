# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

# Add LatentDynamics, Physics directories to the search path.
import  sys;
import  os;
LD_Path         : str   = os.path.abspath(os.path.join(os.path.dirname(__file__), "LatentDynamics"));
Physics_Path    : str   = os.path.abspath(os.path.join(os.path.dirname(__file__), "Physics"));
Utils_Path      : str   = os.path.abspath(os.path.join(os.path.dirname(__file__), "Utilities"));
sys.path.append(LD_Path); 
sys.path.append(Physics_Path); 
sys.path.append(Utils_Path); 

import  yaml;
import  argparse;
import  logging;
import  time;
import  random;

import  numpy;
import  torch;
import  matplotlib.pyplot           as      plt;
from    sklearn.gaussian_process    import  GaussianProcessRegressor;

import  SolveROMs;
from    Enums                       import  NextStep, Result;
from    ParameterSpace              import  ParameterSpace;
from    Physics                     import  Physics;
from    LatentDynamics              import  LatentDynamics;
from    GPLaSDI                     import  BayesianGLaSDI;
from    GaussianProcess             import  fit_gps, eval_gp;
from    Initialize                  import  Initialize_Trainer;
from    Sample                      import  Run_Samples, Update_Train_Space;
from    Logging                     import  Initialize_Logger, Log_Dictionary;
from    Plot                        import  Plot_Heatmap2d, Plot_Latent_Trajectories;
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

        # Set up the restart path.
        restart_path : str = os.path.join(os.path.join(os.path.pardir, "results"), restart_filename);
    
    LOGGER.info("Done! Took %fs" % (time.perf_counter() - timer));



    # ---------------------------------------------------------------------------------------------
    # Run the next step
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
        result          = restart_dict['result'];
    else:
        restart_dict    = {};
        next_step       = NextStep.RunSample;
        result          = Result.Unexecuted;
    
    # Initialize the trainer.
    trainer, param_space, physics, model, latent_dynamics = Initialize_Trainer(config, restart_dict);

    # Start running steps.
    result, next_step = step(trainer, next_step, config, use_restart);

    # Report the result
    if (  result is Result.Fail):
        raise RuntimeError('Previous step has failed. Stopping the workflow.');
    elif (result is Result.Success):
        LOGGER.info("Previous step succeeded. Preparing for the next step.");
        result = Result.Unexecuted;
    elif (result is Result.Complete):
        LOGGER.info("Workflow is finished.");


    # ---------------------------------------------------------------------------------------------
    # Plot!
    # ---------------------------------------------------------------------------------------------

    # ---------------------------------------------------------------------------------------------
    # Setup 

    # Set up gaussian processes. 
    model.cpu();

    # Get a GP for each coefficient in the latent dynamics.
    gp_list         : list[GaussianProcessRegressor]    = fit_gps(param_space.train_space, trainer.best_coefs);

    # Figure out which elements of the test set are in the training set.
    in_train_set : torch.Tensor = torch.zeros(param_space.n_test(), dtype = torch.bool);
    for i in range(param_space.n_train()):
        for j in range(param_space.n_test()):
            if(numpy.all(param_space.train_space[i, :] == param_space.test_space[j, :])):
                in_train_set[j] = True;
                break;

    # Now, randomly sample an element of the test set that isn't in the training set.
    i_random    : int   = random.randrange(0, param_space.n_test());
    while(in_train_set[i_random] == True):
        i_random    : int   = random.randrange(0, param_space.n_test());
    
    # Plot the latent trajectories for the i_random'th element of the test set.
    Plot_Latent_Trajectories(  physics         = physics,
                               model           = model,
                               latent_dynamics = latent_dynamics,
                               gp_list         = gp_list,
                               param_grid      = param_space.test_space[i_random, :].reshape(1, -1),
                               n_samples       = trainer.n_samples,
                               U_True          = [trainer.U_Test[i_random]],
                               t_Grid          = [trainer.t_Test[i_random]],
                               file_prefix     = config["physics"]["type"],
                               figsize         = (15, 13));

    # Compute the relative error between the FOM solution and its reconstruction under the model.
    skip_proportion : float = .05;
    Max_Rel_Error, Max_STD, Rel_Error, STD  = SolveROMs.Compute_Error_and_STD(
                                                model           = model, 
                                                physics         = physics,
                                                param_space     = param_space,
                                                latent_dynamics = latent_dynamics,
                                                gp_list         = gp_list,
                                                t_Test          = trainer.t_Test,
                                                U_Test          = trainer.U_Test,
                                                n_samples       = trainer.n_samples,
                                                skip_proportion = skip_proportion);



    # ---------------------------------------------------------------------------------------------
    # For each combination of parameter values, compute the relative error between the FOM 
    # solution and its reconstruction under the model.

    # Setup
    Rel_Error_Reconstruction        : list[list[numpy.ndarray]] = [];
    Max_Rel_Error_Reconstruction    : numpy.ndarray             = numpy.zeros((param_space.n_test(), physics.n_IC));

    # Cycle through the combinations of parameter values.
    for i in range(param_space.n_test()):
        # Reconstruct the FOM solution, store it in a list.
        LOGGER.debug("Reconstructing the FOM solution for parameter combination %d (%s)" % (i, str(param_space.test_space[i])));
        ith_Reconstruction : torch.Tensor | tuple[torch.Tensor, torch.Tensor] = model(*trainer.U_Test[i]);
        if(isinstance(ith_Reconstruction, tuple)):
            ith_Reconstruction = list(ith_Reconstruction);
        elif(isinstance(ith_Reconstruction, torch.Tensor)):
            ith_Reconstruction = [ith_Reconstruction];
        else:
            raise ValueError("ith_Encoding is not a tuple or a torch.Tensor");
    
        # Setup for the i'th combination of parameter values.
        n_IC                                : int                   = physics.n_IC;
        ith_Rel_Error_Reconstruction        : list[numpy.ndarray]   = [];
        n_t_i                               : int                   = trainer.t_Test[i].shape[0];

        # Cycle through the ICs.
        for j in range(n_IC):
            # Setup a tensor to hold the relative error for the j'th IC and the i'th combination of 
            # parameter values.
            ij_Rel_Error_Reconstruction : numpy.ndarray = numpy.zeros(n_t_i);

            # Fetch the reconstruction and true solution.
            ij_Reconstruction   : numpy.ndarray = ith_Reconstruction[j].detach().numpy();
            ij_True             : numpy.ndarray = trainer.U_Test[i][j].detach().numpy();

            # Compute the Absolute Error and norm of each frame.
            ij_Abs_Error        : numpy.ndarray = numpy.linalg.norm((ij_Reconstruction - ij_True).reshape(n_t_i, -1), axis = 1);    # (n_t_i)
            ij_Norms            : numpy.ndarray = numpy.linalg.norm(ij_True.reshape(n_t_i, -1), axis = 1);                          # (n_t_i)

            # Cycle through the frames.
            for k in range(n_t_i):
                # If the time step is before the skip proportion, set the relative error to 0. 
                # Otherwise, compute the relative error.
                if(trainer.t_Test[i][k] < skip_proportion*trainer.t_Test[i][-1]):
                    ij_Rel_Error_Reconstruction[k] = 0;
                else:
                    # Compute the relative error for the k'th frame.
                    ij_Rel_Error_Reconstruction[k] = ij_Abs_Error[k]/ij_Norms[k];
            
            # Append the relative error for the j'th IC.
            ith_Rel_Error_Reconstruction.append(ij_Rel_Error_Reconstruction);

            # Compute the maximum relative error for the j'th time derivative of the solution for 
            # the i'th combination of parameter values.
            Max_Rel_Error_Reconstruction[i, j] = numpy.max(ij_Rel_Error_Reconstruction);
        
        # Append the relative error for the i'th combination of parameter values.
        Rel_Error_Reconstruction.append(ith_Rel_Error_Reconstruction);
    


    # ---------------------------------------------------------------------------------------------
    # Plot Rel_Error for the i_random'th combination of parameters.

    for i in range(physics.n_IC):
        plt.figure();
        plt.plot(trainer.t_Test[i_random], Rel_Error[i_random][i]);
        plt.xlabel("time (s)");
        plt.ylabel("Relative Error");

        if(i == 0):     
            title_str       : str = "Relative Error of the reconstruction of U for parameter combination %s"        % str(param_space.test_space[i_random]);
            save_file_name  : str = config["physics"]["type"] + "_U_Relative_Error_%s"                              % str(param_space.test_space[i_random]);   
        elif(i == 1):   
            title_str       : str = "Relative Error of the reconstruction of D_t U for parameter combination %s"    % str(param_space.test_space[i_random]);
            save_file_name  : str = config["physics"]["type"] + "_Dt_U_Relative_Error_%s"                           % str(param_space.test_space[i_random]);
        else:           
            title_str       : str = "Relative Error of the reconstruction of D_t^%d U for parameter combination %s" % (i, str(param_space.test_space[i_random]));
            save_file_name  : str = config["physics"]["type"] + "_Dt^%d_U_Relative_Error_%s"                        % (i, str(param_space.test_space[i_random]));

        # Plot the figure.
        plt.title(title_str);
    
        # Now save the figure.
        plt.savefig(os.path.join(os.path.join(os.path.pardir, "Figures"), save_file_name));
    plt.show();



    # ---------------------------------------------------------------------------------------------
    # Make movies for the mean predicted solution, true solution, and error for the i_random'th 
    # combination of parameters.

    # If X_Positions has the form (2, N_Positions), then the solution must either be a 
    # scalar field or a 2d vector field. Let's plot the solution.
    if(len(physics.X_Positions.shape) == 2 and  physics.X_Positions.shape[0] == 2):
        
        # First, generate latent trajectories for the i_random'th element of the test set.
        LOGGER.debug("Generating trajectory plot for testing combination %d: %s" % (i_random, param_space.test_space[i_random]));

        # Generate the solution trajectory using the mean for the posterior distribution.
        param_random    : numpy.ndarray         = param_space.test_space[i_random, :].reshape(1, -1);
        t_random        : numpy.ndarray         = trainer.t_Test[i_random];
        U_True          : list[torch.Tensor]    = trainer.U_Test[i_random];
        Zi_mean_np      : list[numpy.ndarray]   = average_rom(  model           = model,            # n_IC element list whose j'th element has shape (n_t(i), n_z)
                                                                physics         = physics, 
                                                                latent_dynamics = latent_dynamics, 
                                                                gp_list         = gp_list, 
                                                                param_grid      = param_random, 
                                                                t_Grid          = [t_random])[0];  

        # Map Zi_mean_np to a tensor and then decode.
        Zi_mean     : list[torch.Tensor]    = [];
        for i in range(len(Zi_mean_np)):
            Zi_mean.append(torch.Tensor(Zi_mean_np[i]));
        U_Pred      : tuple[torch.Tensor] | torch.Tensor    = model.Decode(*Zi_mean);

        # Convert U_Pred to a list
        if(isinstance(U_Pred, tuple)):
            U_Pred = list(U_Pred);
        elif(isinstance(U_Pred, torch.Tensor)):
            U_Pred = [U_Pred];
        else:
            raise ValueError("U_Pred is not a tuple or a torch.Tensor");

        # Fetch the positions.
        X           : numpy.ndarray         = physics.X_Positions;

        # Make a movie for each derivative of the solution.
        n_IC        : int                   = physics.n_IC;
        for i in range(n_IC):
            if(i == 0):
                prefix : str = "%s_U_%s" % (config["physics"]["type"], str(param_space.test_space[i_random]));
            else:
                prefix : str = "%s_(Dt^%d)U_%s" % (config["physics"]["type"], i, str(param_space.test_space[i_random]));
            make_solution_movies(U_True         = U_True[i].detach().numpy(), 
                                 U_Pred         = U_Pred[i].detach().numpy(), 
                                 X              = X, 
                                 T              = t_random,
                                 fname_prefix   = prefix);
    

    # ---------------------------------------------------------------------------------------------
    # Plot the heatmaps

    if(param_space.n_p == 2):
        n_IC : int = latent_dynamics.n_IC;
        
        # Plot the maximum (across the frames) relative reconstruction error between each frame of each 
        # derivative of the FOM solution for each combination of parameter values and their 
        # corresponding reconstructions.
        for d in range(n_IC):
            if(d == 0):
                title           : str   = r'$\text{max}_{t, i} \frac{\left| u_{\bar{\xi}}(t, x_i) - u_{\text{True}}(t, x_i) \right|} {\text{max}_{j} \left| u_{\text{True}}(t, x_j) \right|}$';
                save_file_name  : str   = config["physics"]["type"] + "_U_Relative_Error_Reconstruction_Heatmap";
            elif(d == 1):
                title           : str   = r'$\text{max}_{t, i} \frac{\left| \frac{d}{dt}u_{\bar{\xi}}(t, x_i) - \frac{d}{dt}u_{\text{True}}(t, x_i) \right|}{\text{max}_{j} \left| \frac{d}{dt}u_{\text{True}}(t, x_j) \right|}$';
                save_file_name  : str   = config["physics"]["type"] + "_Dt_U_Relative_Error_Reconstruction_Heatmap";
            else:
                title           : str   = r'$\text{max}_{t, i} \frac{\left| \frac{d^{%d}}{dt^{%d}}u_{\bar{\xi}}(t, x_i) - \frac{d^{%d}}{dt^{%d}}u_{\text{True}}(t, x_i) \right|}{\text{max}_{j} \left| \frac{d^{%d}}{dt^{%d}}u_{\text{True}}(t, x_j) \right|}$' % (d, d, d, d, d, d);
                save_file_name  : str   = config["physics"]["type"] + "_Dt^%d_U_Relative_Error_Reconstruction_Heatmap" % d;

            Plot_Heatmap2d(     values          = Max_Rel_Error_Reconstruction[:, d].reshape(param_space.test_grid_sizes) * 100, 
                                param_space     = param_space,
                                title           = title, 
                                save_file_name  = save_file_name);
        

        # Plot maximum (across the frames) relative reconstruction error between each frame of each 
        # derivative of the FOM solution for each combination of parameter values and their 
        # corresponding reconstructions.
        for d in range(n_IC):
            if(d == 0):
                title           : str   = r'$\text{max}_{t, i} \frac{\left| u_{\bar{\xi}}(t, x_i) - u_{\text{True}}(t, x_i) \right|} {\text{max}_{j} \left| u_{\text{True}}(t, x_j) \right|}$';
                save_file_name  : str   = config["physics"]["type"] + "_U_Relative_Error_Heatmap";
            elif(d == 1):
                title           : str   = r'$\text{max}_{t, i} \frac{\left| \frac{d}{dt}u_{\bar{\xi}}(t, x_i) - \frac{d}{dt}u_{\text{True}}(t, x_i) \right|}{\text{max}_{j} \left| \frac{d}{dt}u_{\text{True}}(t, x_j) \right|}$';
                save_file_name  : str   = config["physics"]["type"] + "_Dt_U_Relative_Error_Heatmap";
            else:
                title           : str   = r'$\text{max}_{t, i} \frac{\left| \frac{d^{%d}}{dt^{%d}}u_{\bar{\xi}}(t, x_i) - \frac{d^{%d}}{dt^{%d}}u_{\text{True}}(t, x_i) \right|}{\text{max}_{j} \left| \frac{d^{%d}}{dt^{%d}}u_{\text{True}}(t, x_j) \right|}$' % (d, d, d, d, d, d);
                save_file_name  : str   = config["physics"]["type"] + "_Dt^%d_U_Relative_Error_Heatmap" % d;

            Plot_Heatmap2d(     values          = Max_Rel_Error[:, d].reshape(param_space.test_grid_sizes) * 100, 
                                param_space     = param_space,
                                title           = title, 
                                save_file_name  = save_file_name);

        # Plot the std of the component of the frame with the largest std (across the samples) in 
        # the reconstruction of that component of that frame. Do this for each combination of 
        # parameter values and derivative of the FOM solution.
        for d in range(n_IC):
            if(d == 0):
                title           : str   = r'$\text{max}_{(t, i)} \sigma_{j \in \{1, \ldots, %d\}} \left[ u_{\xi(j)}(t, x_i) \right]$' % trainer.n_samples;
                save_file_name  : str   = config["physics"]["type"] + "_U_STD_Heatmap";
            elif(d == 1):
                title           : str   = r'$\text{max}_{(t, i)} \sigma_{j \in \{ 1, \ldots, %d\}} \left[\frac{d}{dt}u_{\xi(j)}(t, x_i) \right]$' % (trainer.n_samples);
                save_file_name  : str   = config["physics"]["type"] + "_Dt_U_STD_Heatmap";      
            else:
                title           : str   = r'$\text{max}_{(t, i)} \sigma_{j \in \{ 1, \ldots, %d\}} \left[\frac{d^{%d}}{dt^{%d}}u_{\xi(j)}(t, x_i) \right]$' % (trainer.n_samples, d, d);
                save_file_name  : str   = config["physics"]["type"] + "_Dt^%d_U_STD_Heatmap" % d;


            Plot_Heatmap2d( values          = Max_STD[:, d].reshape(param_space.test_grid_sizes) * 100,
                            param_space     = param_space, 
                            title           = title,
                            save_file_name  = save_file_name);


    # ---------------------------------------------------------------------------------------------
    # Save!
    # ---------------------------------------------------------------------------------------------

    # Save!
    Save(   param_space         = param_space,
            config              = config,
            physics             = physics,
            model               = model, 
            latent_dynamics     = latent_dynamics,
            trainer             = trainer,
            next_step           = next_step,
            result              = result,
            restart_filename    = restart_filename);

    # All done!
    return;




# -------------------------------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------------------------------

def step(trainer        : BayesianGLaSDI, 
         next_step      : NextStep, 
         config         : dict, 
         use_restart    : bool = False) -> tuple[Result, NextStep]:
    """
    This function runs the next step of the training procedure. Depending on what we have done, 
    that next step could be training, picking new samples, generating FOM solutions, or 
    collecting samples. 

    
    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------
    
    trainer : BayesianGLaSDI
        A Trainer class object that we use when training the model for a particular instance of 
        the settings.

    next_step : NextStep
        specifies what the next step is. 

    config : dict
        This should be a dictionary that we loaded from a .yml file. It should house all the 
        settings we expect to use to generate the data and train the models.

    use_restart : bool
         if True, we return after a single step.

    

    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    result, next_step
    
    result : Result
        indicates what happened during the current step

    next_step : NextStep
        Indicates what we should do next. 
    """


    # ---------------------------------------------------------------------------------------------
    # Run the next step 
    # ---------------------------------------------------------------------------------------------

    LOGGER.info("Running %s" % next_step);
    if (next_step is NextStep.Train):
        # If our next step is to train, then let's train! This will set trainer.restart_iter to 
        # the iteration number of the last iterating training.
        trainer.train();

        # Check if we should stop running steps.
        # Recall that a trainer object's restart_iter member holds the iteration number of the last
        # iteration in the last round of training. Likewise, its "max_iter" member specifies the 
        # total number of iterations we want to train for. Thus, if restart_iter goes above 
        # max_iter, then it is time to stop running steps. Otherwise, we can mark the current step
        # as a success and move onto the next one.
        if (trainer.restart_iter >= trainer.max_iter):
            result  = Result.Complete;
        else:
            result  = Result.Success;

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
        result, next_step = Update_Train_Space(trainer, config);


    elif (next_step is NextStep.RunSample):
        # Generate the trajectories for all new testing and training parameters. Append these new
        # trajectories to trainer's U_Train and U_Test attributes.
        result, next_step = Run_Samples(trainer, config);


    else:
        raise RuntimeError("Unknown next step!");
    


    # ---------------------------------------------------------------------------------------------
    # Wrap up
    # ---------------------------------------------------------------------------------------------

    # If fail or complete, break the loop regardless of use_restart.
    if ((result is Result.Fail) or (result is Result.Complete)):
        return result, next_step;
        
    # Continue the workflow.
    LOGGER.info("Next step is: %s" % next_step);
    result, next_step = step(trainer, next_step, config);

    # All done!
    return result, next_step;



def Save(   param_space         : ParameterSpace, 
            config              : dict,
            physics             : Physics, 
            model               : torch.nn.Module, 
            latent_dynamics     : LatentDynamics,
            trainer             : BayesianGLaSDI, 
            next_step           : NextStep, 
            result              : Result,
            restart_filename    : str               = "") -> None:
    """
    This function saves a trained model, trainer, latent dynamics, etc. You should call this 
    function after running the LASDI algorithm.


    
    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    param_space : ParameterSpace 
        holds the training and testing parameter combinations.
    
    config : dict
        This should be a dictionary that we loaded from a .yml file. It should house all the 
        settings we expect to use to generate the data and train the models.

    physics : Physics
        defines the FOM model. We can use it to fetch the initial conditions and FOM solution for
        a particular combination of parameter values. physics, latent_dynamics, and model should 
        have the same number of initial conditions.

    model : torch.nn.Module
        maps between the FOM and ROM spaces. physics, latent_dynamics, and model should have the 
        same number of initial conditions.

    latent_dynamics : LatentDynamics 
        defines the dynamics in model's latent space. physics, latent_dynamics, and model should 
        have the same number of initial conditions.

    trainer : BayesianGLaSDI
        trains model using physics to define the FOM, latent_dynamics to define the ROM, and 
        model to connect them.

    next_step : NextStep
        An enumeration indicating the next step (should we continue training). This should 
        have been returned by the final call to the step function.

    result : Result
        An enumeration indicating the result of the last step of training. This should have
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
    assert(model.n_IC       == n_IC);
    assert(physics.n_IC     == n_IC);


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
        # Append the new date to the restart filename.
        restart_filename = restart_filename + '.' + date_str;
    else:
        restart_filename : str = config["physics"]["type"] + '_' + date_str + '.npy';
    
    # Set up the restart path.
    restart_path        = os.path.join(os.path.join(os.path.pardir, "results"), restart_filename);

    # Build the restart save dictionary and then save it.
    restart_dict = {'parameter_space'   : param_space.export(),
                    'physics'           : physics.export(),
                    'model'             : model.export(),
                    'latent_dynamics'   : latent_dynamics.export(),
                    'trainer'           : trainer.export(),
                    'timestamp'         : date_str,
                    'next_step'         : next_step,
                    'result'            : result};
    numpy.save(restart_path, restart_dict);

    # All done!
    return;


if __name__ == "__main__":
    main();