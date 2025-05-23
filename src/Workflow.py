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
from    Plot                        import  Plot_Heatmap2d, Plot_GP2d;

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
    use_restart : bool = config['workflow']['use_restart'];
    restart_filename : str = None;
    if (use_restart == True):
        restart_filename : str = config['workflow']['restart_file'];
        LOGGER.debug("Loading from restart (%s)" % restart_filename);

        from pathlib import Path
        Path(os.path.dirname(restart_filename)).mkdir(parents = True, exist_ok = True);
    
    LOGGER.info("Done! Took %fs" % (time.perf_counter() - timer));



    # ---------------------------------------------------------------------------------------------
    # Run the next step
    # ---------------------------------------------------------------------------------------------

    # Determine what the next step is. If we are loading from a restart, then the restart should
    # have logged then next step. Otherwise, we set the next step to "PickSample", which will 
    # prompt the code to set up the training set of parameters.
    if ((use_restart == True) and (os.path.isfile(restart_filename))):
        # TODO(kevin): in long term, we should switch to hdf5 format.
        restart_dict    = numpy.load(restart_filename, allow_pickle = True).item();
        next_step       = restart_dict['next_step'];
        result          = restart_dict['result'];
    else:
        restart_dict    = None;
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

    # Set up gaussian processes. 
    model.cpu();

    # Get a GP for each coefficient in the latent dynamics.
    gp_list         : list[GaussianProcessRegressor]    = fit_gps(param_space.train_space, trainer.best_coefs);

    gp_pred_mean, gp_pred_std = eval_gp(gp_list, param_space.test_space);
    gp_pred_mean    = gp_pred_mean.reshape(param_space.test_grid_sizes + [-1]);
    gp_pred_std     = gp_pred_std.reshape(param_space.test_grid_sizes + [-1]);

    Max_Rel_Error, Max_STD  = SolveROMs.Compute_Error_and_STD(
                                                model           = model, 
                                                physics         = physics,
                                                param_space     = param_space,
                                                latent_dynamics = latent_dynamics,
                                                gp_list         = gp_list,
                                                t_Test          = trainer.t_Test,
                                                X_Test          = trainer.X_Test,
                                                n_samples       = trainer.n_samples);

    if(param_space.n_p == 2):
        n_IC : int = latent_dynamics.n_IC;
        
        # Plot the mean and STD of the posterior distribution for each coefficient evaluated at
        # each combination of parameter values.
        Plot_GP2d(  p1_mesh     = param_space.test_meshgrid[0], 
                    p2_mesh     = param_space.test_meshgrid[1], 
                    gp_mean     = gp_pred_mean, 
                    gp_std      = gp_pred_std, 
                    param_train = param_space.train_space, 
                    param_names = param_space.param_names, 
                    n_cols      = 5);
        
        # Plot maximum (across the frames) relative reconstruction error between each frame of each 
        # derivative of the FOM solution for each combination of parameter values and their 
        # corresponding reconstructions.
        for d in range(n_IC):
            if(d == 0):
                title   : str   = r'$max_{t} \frac{\left\| u_\bar{\xi}(t, x) - u(t, x) \right\|_x} {\| u(t, x)\|_x}$';
            elif(d == 1):
                title   : str   = r'$max_{t} \frac{\left\| \frac{d}{dt}u_{\bar{\xi}}(t, x) - \frac{d}{dt}u(t, x) \right\|_x}{\left\| \frac{d}{dt}u(t, x) \right\|_x}$';
            else:
                title   : str   = r'$max_{t} \frac{\left\| \frac{d^{%d}}{dt^{%d}}u_{\bar{\xi}}(t, x) - \frac{d^{%d}}{dt^{%d}}u(t, x) \right\|_x}{\left\| \frac{d^{%d}}{dt^{%d}}u(t, x) \right\|_x}$' % (d, d, d, d, d, d);

            Plot_Heatmap2d(     values          = Max_Rel_Error[d] * 100, 
                                param_space     = param_space,
                                title           = title);

        # Plot the std of the component of the frame with the largest std (across the samples) in 
        # the reconstruction of that component of that frame. Do this for each combination of 
        # parameter values and derivative of the FOM solution.
        for d in range(n_IC):
            if(d == 0):
                title   : str   = r'$max_{(t, x)} \sigma_{i \in \{1, \ldots, %d\}} \left[ u_{\xi^i} \right]$' % trainer.n_samples;
            elif(d == 1):
                title   : str   = r'$max_{(t, x)} \sigma_{i \in \{ 1, \ldots, %d\}} \left[\frac{d}{dt}u_{\xi^i} \right]$' % (trainer.n_samples);
            else:
                title   : str   = r'$max_{(t, x)} \sigma_{i \in \{ 1, \ldots, %d\}} \left[\frac{d^{%d}}{dt^{%d}}u_{\xi^i} \right]$' % (trainer.n_samples, d, d);

            Plot_Heatmap2d( values          = Max_STD[d] * 100,
                            param_space     = param_space, 
                            title           = title);



    # ---------------------------------------------------------------------------------------------
    # Save!
    # ---------------------------------------------------------------------------------------------

    # Save!
    Save(   param_space         = param_space,
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
        # trajectories to trainer's X_Train and X_Test attributes.
        result, next_step = Run_Samples(trainer, config);


    else:
        raise RuntimeError("Unknown next step!");
    


    # ---------------------------------------------------------------------------------------------
    # Wrap up
    # ---------------------------------------------------------------------------------------------

    # If fail or complete, break the loop regardless of use_restart.
    if ((result is Result.Fail) or (result is Result.Complete)):
        return result, next_step;
        
    # Continue the workflow if not using restart.
    LOGGER.info("Next step is: %s" % next_step);
    if (use_restart == False):
        result, next_step = step(trainer, next_step, config);

    # All done!
    return result, next_step;



def Save(   param_space         : ParameterSpace, 
            physics             : Physics, 
            model               : torch.nn.Module, 
            latent_dynamics     : LatentDynamics,
            trainer             : BayesianGLaSDI, 
            next_step           : NextStep, 
            result              : Result,
            restart_filename    : str               = None) -> None:
    """
    This function saves a trained model, trainer, latent dynamics, etc. You should call this 
    function after running the LASDI algorithm.


    
    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    param_space : ParameterSpace 
        holds the training and testing parameter combinations.
    
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
        Otherwise, if we did not load from a restart, this should be None.


    
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
                                  hour      = date.tm_hour + 3, 
                                  minute    = date.tm_min);
    
    if(restart_filename != None):
        # rename old restart file if exists.
        if (os.path.isfile(restart_filename) == True):
            old_timestamp = restart_dict['timestamp'];
            os.rename(restart_filename, restart_filename + '.' + old_timestamp);
        restart_path : str = restart_filename;
    else:
        restart_path : str = 'lasdi_' + date_str + '.npy';
    
    # Make sure we place this in the results dictionary.
    restart_path        = os.path.join(os.path.join(os.path.pardir, "results"), restart_path);

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