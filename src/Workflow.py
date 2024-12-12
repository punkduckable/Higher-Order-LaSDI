# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

# Add LatentDynamics, Physics directories to the search path.
import  sys;
import  os;
LD_Path         : str = os.path.abspath(os.path.join(os.path.dirname(__file__), "LatentDynamics"));
Physics_Path    : str = os.path.abspath(os.path.join(os.path.dirname(__file__), "Physics"));
sys.path.append(LD_Path); 
sys.path.append(Physics_Path); 

import  yaml;
import  argparse;


import  numpy as np;

from    Enums               import  NextStep, Result;
from    GPLaSDI             import  BayesianGLaSDI;
from    InputParser         import  InputParser;
from    Initialize          import  Initialize_Trainer;
from    Sample              import  Run_Samples, Pick_Samples, Collect_Samples;

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
    print("config file: %s" % args.config);

    # Load the configuration file. 
    with open(args.config, 'r') as f:
        config      = yaml.safe_load(f)
        cfg_parser  = InputParser(config, name = 'main')

    # Check if we are loading from a restart or not. If so, load it.
    use_restart : bool = cfg_parser.getInput(['workflow', 'use_restart'], fallback = False)
    if (use_restart == True):
        restart_filename : str = cfg_parser.getInput(['workflow', 'restart_file'], datatype = str)
        from pathlib import Path
        Path(os.path.dirname(restart_filename)).mkdir(parents = True, exist_ok = True)
    


    # ---------------------------------------------------------------------------------------------
    # Run the next step
    # ---------------------------------------------------------------------------------------------

    # Determine what the next step is. If we are loading from a restart, then the restart should
    # have logged then next step. Otherwise, we set the next step step to "PickSample", which will 
    # prompt the code to set up the training set of parameters.
    if ((use_restart == True) and (os.path.isfile(restart_filename))):
        # TODO(kevin): in long term, we should switch to hdf5 format.
        restart_dict    = np.load(restart_filename, allow_pickle = True).item()
        next_step       = restart_dict['next_step']
        result          = restart_dict['result']
    else:
        restart_dict    = None
        next_step       = NextStep.PickSample
        result          = Result.Unexecuted
    
    # Initialize the trainer.
    trainer, param_space, physics, model, latent_dynamics = Initialize_Trainer(config, restart_dict)

    # Run the next step.
    result, next_step = step(trainer, next_step, config, use_restart)

    # Report the result
    if (  result is Result.Fail):
        raise RuntimeError('Previous step has failed. Stopping the workflow.')
    elif (result is Result.Success):
        print("Previous step succeeded. Preparing for the next step.")
        result = Result.Unexecuted
    elif (result is Result.Complete):
        print("Workflow is finished.")



    # ---------------------------------------------------------------------------------------------
    # Save, shutdown
    # ---------------------------------------------------------------------------------------------

    # Save restart (or final) file.
    import time
    date        = time.localtime()
    date_str    = "{month:02d}_{day:02d}_{year:04d}_{hour:02d}_{minute:02d}"
    date_str    = date_str.format(month     = date.tm_mon, 
                                  day       = date.tm_mday, 
                                  year      = date.tm_year, 
                                  hour      = date.tm_hour + 3, 
                                  minute    = date.tm_min)
    
    if (use_restart == True):
        # rename old restart file if exists.
        if (os.path.isfile(restart_filename) == True):
            old_timestamp = restart_dict['timestamp']
            os.rename(restart_filename, restart_filename + '.' + old_timestamp)
        restart_path : str = restart_filename
    else:
        restart_path : str = 'lasdi_' + date_str + '.npy'
    
    # Build the restart save dictionary and then save it.
    restart_dict = {'parameter_space'   : param_space.export(),
                    'physics'           : physics.export(),
                    'model'             : model.export(),
                    'latent_dynamics'   : latent_dynamics.export(),
                    'trainer'           : trainer.export(),
                    'timestamp'         : date_str,
                    'next_step'         : next_step,
                    'result'            : result};
    np.save(restart_path, restart_dict)

    # All done!
    return




# -------------------------------------------------------------------------------------------------
# Step
# -------------------------------------------------------------------------------------------------

def step(trainer        : BayesianGLaSDI, 
         next_step      : NextStep, 
         config         : dict, 
         use_restart    : bool = False):
    """
    This function runs the next step of the training procedure. Depending on what we have done, 
    that next step could be training, picking new samples, generating fom solutions, or 
    collecting samples. 

    
    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------
    
    trainer: A Trainer class object that we use when training the model for a particular instance
    of the settings.

    next_step: a NextStep object (a kind of enumeration) specifying what the next step is. 

    config: This should be a dictionary that we loaded from a .yml file. It should house all the 
    settings we expect to use to generate the data and train the models.

    use_restart: a boolean which, if True, will prompt us to return after a single step.

    

    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    A Returns object (a kind of enumeration) that indicates what happened during the current step.
    """


    # ---------------------------------------------------------------------------------------------
    # Run the next step 
    # ---------------------------------------------------------------------------------------------

    print("Running %s" % next_step);
    if (next_step is NextStep.Train):
        # If our next step is to train, then let's train! This will set trainer.restart_iter to 
        # the iteration number of the last iterating training.
        trainer.train()

        # Check if we should stop running steps.
        # Recall that a trainer object's restart_iter member holds the iteration number of the last
        # iteration in the last round of training. Likewise, its "max_iter" member specifies the 
        # total number of iterations we want to train for. Thus, if restart_iter goes above 
        # max_iter, then it is time to stop running steps. Otherwise, we can mark the current step
        # as a success and move onto the next one.
        if (trainer.restart_iter >= trainer.max_iter):
            result  = Result.Complete
        else:
            result  = Result.Success

        # Next, check if the restart_iter falls below the "max_greedy_iter". The later is the last
        # iteration at which we want to run greedy sampling. If the restart_iter is below the 
        # max_greedy_iter, then we should pick a new sample (perform greedy sampling). Otherwise, 
        # we should continue training.
        if (trainer.restart_iter <= trainer.max_greedy_iter):
            next_step = NextStep.PickSample
        else:
            next_step = NextStep.Train



    elif (next_step is NextStep.PickSample):
        result, next_step = Pick_Samples(trainer, config)



    elif (next_step is NextStep.RunSample):
        result, next_step = Run_Samples(trainer, config)



    elif (next_step is NextStep.CollectSample):
        # Note: We should only reach here if we are using the offline stuff... since I disabled 
        # that, we should never reach this step.
        print("NextStep = Collect_Samples. Has something gone wrong? We should only be here if running offline (which should be disabled).");
        result, next_step = Collect_Samples(trainer, config)



    else:
        raise RuntimeError("Unknown next step!")
    


    # ---------------------------------------------------------------------------------------------
    # Wrap up
    # ---------------------------------------------------------------------------------------------

    # If fail or complete, break the loop regardless of use_restart.
    if ((result is Result.Fail) or (result is Result.Complete)):
        return result, next_step
        
    # Continue the workflow if not using restart.
    print("Next step is: %s" % next_step);
    if (use_restart == False):
        result, next_step = step(trainer, next_step, config)

    # All done!
    return result, next_step




if __name__ == "__main__":
    main()