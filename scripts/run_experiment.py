# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

import  sys;
import  os;
from    pathlib                     import  Path;
import  shutil;

# Expose `src/` as the import root for the HLaSDI package.
PROJECT_DIR         : Path  = Path(__file__).resolve().parent.parent;
SRC_Path            : str   = str(PROJECT_DIR / "src");

if(SRC_Path not in sys.path):
    sys.path.insert(0, SRC_Path);

import  yaml;
import  argparse;
import  logging;
import  time;

import  numpy;

from    HLaSDI.EncoderDecoder              import  EncoderDecoder;
from    HLaSDI.ParameterSpace       import  ParameterSpace;
from    HLaSDI.Physics                     import  Physics;
from    HLaSDI.Enums                import  NextStep;
from    HLaSDI.LatentDynamics              import  LatentDynamics;
from    HLaSDI.Trainer                     import  Trainer;
from    HLaSDI.Initialize           import  Initialize_Trainer;
from    HLaSDI.Sample                      import  Sampler;
from    HLaSDI.Schemas              import  validate_experiment_config;
from    HLaSDI.Utilities.Logging    import  Initialize_Logger, Log_Dictionary;


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

    # Load in the argument
    args : argparse.Namespace = parser.parse_args(sys.argv[1:]);
    LOGGER.debug("config file: %s" % args.config);

    LOGGER.info("Setting up...");
    timer     : float = time.perf_counter();

    # Load the configuration file. 
    with open(args.config, 'r') as f:
        raw_config  = yaml.safe_load(f);
    config = validate_experiment_config(raw_config);
    
    # Report the validated configuration settings.
    Log_Dictionary(LOGGER = LOGGER, D = config.to_runtime_dict(), level = logging.INFO);

    # Check if we are loading from a restart or not. If so, load it.
    use_restart         : bool  = config.workflow.use_restart;
    if (use_restart == True):
        restart_path    : str   = str(PROJECT_DIR / config.workflow.restart_file);
        LOGGER.info("Loading from restart (%s)" % restart_path);
    
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
        
        restart_dict    = numpy.load(restart_path, allow_pickle = True).item();
        next_step       = restart_dict['next_step'];
    else:
        restart_dict    = {};
        next_step       = NextStep.RunSample;
    
    # Initialize the trainer.
    trainer, sampler, param_space, physics, encoder_decoder, latent_dynamics = Initialize_Trainer(config, restart_dict);

    # Back up the exact config file used to start this trainer immediately after the run-specific
    # results directory exists. This preserves the launch-time settings even if the source YAML is
    # edited before training/analysis completes.
    config_backup_path : Path = Path(trainer.results_dir) / Path(args.config).name;
    shutil.copy2(args.config, config_backup_path);
    LOGGER.info("Copied run config to %s" % config_backup_path);

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
    Save(   param_space         = param_space,
            config              = config,
            physics             = physics,
            encoder_decoder     = encoder_decoder, 
            latent_dynamics     = latent_dynamics,
            trainer             = trainer,
            next_step           = next_step);



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

    # Check if training has finished. Recall that a trainer object's restart_iter member holds the 
    # iteration number of the last iteration in the last round of training. Likewise, its 
    # "max_iter" member specifies the total number of iterations we want to train for. Thus, if 
    # restart_iter goes above max_iter, then it is time to stop running steps. 
    if(trainer.restart_iter >= trainer.max_iter):
        return next_step;


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
        # if training has finished, then 
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
        
    else:
        raise RuntimeError("Unknown next step!");
    


    # ---------------------------------------------------------------------------------------------
    # Move onto the next step!
    # ---------------------------------------------------------------------------------------------
        
    # Continue the workflow!
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
            next_step           : NextStep) -> None:
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


    
    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    Nothing!
    """

    # Checks.
    n_IC    : int   = latent_dynamics.n_IC;
    assert encoder_decoder.n_IC     == n_IC, "encoder_decoder.n_IC = %d != n_IC = %d" % (encoder_decoder.n_IC, n_IC);
    assert(physics.n_IC             == n_IC);

    # Set up the restart path.
    # Use an absolute results directory under the project root (Higher-Order-LaSDI/results),
    # independent of the current working directory.
    results_dir = Path(trainer.results_dir);
    results_dir.mkdir(parents=True, exist_ok=True);
    restart_file = str(results_dir / (config.physics.type + '.npy'));
    LOGGER.info("Saving results to %s" % restart_file);

    # Build the restart save dictionary and then save it.
    config_dict = config.to_runtime_dict() if hasattr(config, "to_runtime_dict") else config;
    restart_dict = {'parameter_space'   : param_space.export(),
                    'config'            : config_dict,
                    'physics'           : physics.export(),
                    'encoder_decoder'   : encoder_decoder.export(),
                    'latent_dynamics'   : latent_dynamics.export(),
                    'trainer'           : trainer.export(),
                    'next_step'         : next_step};
    numpy.save(restart_file, restart_dict);

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
    # dynamics coefficients)
    coef_params = sum(t.numel() for t in latent_dynamics.parameters());
    
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
