# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

# Add LatentDynamics, Physics directories to the search path.
import  sys;
import  os;
LD_Path         : str = os.path.abspath(os.path.join(os.curdir, "LatentDynamics"));
Physics_Path    : str = os.path.abspath(os.path.join(os.curdir, "Physics"));
sys.path.append(LD_Path); 
sys.path.append(Physics_Path); 

import  yaml;
import  argparse;
import  h5py;

import  numpy as np;
import  torch;

from    Enums               import  NextStep, Result;
from    GPLaSDI             import  BayesianGLaSDI;
from    Model               import  Autoencoder;
from    SINDy               import  SINDy;
from    burgers1d           import  Burgers1D;
from    Physics             import  Physics
from    ParameterSpace      import  ParameterSpace;
from    InputParser         import  InputParser;



# -------------------------------------------------------------------------------------------------
# Function + Class dictionaries, Command line arguments setup
# -------------------------------------------------------------------------------------------------

# Set up the dictionaries; we use this to allow the code to call different classes, functions 
# depending on the settings.
trainer_dict    = {'gplasdi'    : BayesianGLaSDI};
latent_dict     = {'ae'         : Autoencoder};
ld_dict         = {'sindy'      : SINDy};
physics_dict    = {'burgers1d'  : Burgers1D};

# Set up the command line arguments
parser = argparse.ArgumentParser(description        = "",
                                 formatter_class    = argparse.RawTextHelpFormatter);
parser.add_argument('config_file', 
                    metavar     = 'string', 
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
    print("config file: %s" % args.config_file);

    # Load the configuration file. 
    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)
        cfg_parser = InputParser(config, name='main')

    # Check if we are loading from a restart or not. If so, load it.
    use_restart : bool = cfg_parser.getInput(['workflow', 'use_restart'], fallback = False)
    if (use_restart):
        restart_filename : str = cfg_parser.getInput(['workflow', 'restart_file'], datatype = str)
        from pathlib import Path
        Path(os.path.dirname(restart_filename)).mkdir(parents = True, exist_ok = True)
    


    # ---------------------------------------------------------------------------------------------
    # Run the next step
    # ---------------------------------------------------------------------------------------------

    # Determine what the next step is. If we are loading from a restart, then the restart should
    # have logged then next step. Otherwise, we set the next step step to "PickSample", which will 
    # prompt the code to set up the training set of parameters.
    if (use_restart and (os.path.isfile(restart_filename))):
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

    if ((not use_restart) and physics.offline):
        raise RuntimeError("Offline physics solver needs to use restart files!")

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
    
    if (use_restart):
        # rename old restart file if exists.
        if (os.path.isfile(restart_filename)):
            old_timestamp = restart_dict['timestamp']
            os.rename(restart_filename, restart_filename + '.' + old_timestamp)
        restart_path : str = restart_filename
    else:
        restart_path : str = 'lasdi_' + date_str + '.npy'
    
    # Build the restart save dictionary and then save it.
    restart_dict = {
                'parameter_space'   : param_space.export(),
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

    use_restart: a boolean which, if true, will prompt us to return after a single step.

    

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
        result, next_step = pick_samples(trainer, config)



    elif (next_step is NextStep.RunSample):
        result, next_step = run_samples(trainer, config)



    elif (next_step is NextStep.CollectSample):
        result, next_step = collect_samples(trainer, config)



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
    if (not use_restart):
        result, next_step = step(trainer, next_step, config)

    # All done!
    return result, next_step



# -------------------------------------------------------------------------------------------------
# Initialization functions
# -------------------------------------------------------------------------------------------------

def Initialize_Trainer(config, restart_dict : dict = None):
    """
    Initialize a LaSDI object with a latent space model and physics object according to config 
    file. Currently only 'gplasdi' is available.

    

    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    config: This should be a dictionary that we loaded from a .yml file. It should house all the 
    settings we expect to use to generate the data and train the models. We expect this dictionary 
    to contain the following keys (if a key is within a dictionary that is specified by another 
    key, then we tab the sub-key relative to the dictionary key): 
        - physics           (used by "initialize_physics")
            - type
        - latent_dynamics   (how we parameterize the latent dynamics; e.g. SINDy)
            - type
        - lasdi
            - type

    restart_dict: The dictionary returned by numpy.load when we load from a restart. This should
    contain the following keys:
        - parameter_space
        - model
        - latent_dynamics
        - trainer
            
    
    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    A "BayesianGLaSDI" object that has been initialized using the settings in config/is ready to 
    begin training.
    """

    '''
    Initialize a LaSDI class with a latent space model according to config file.
    Currently only 'gplasdi' is available.
    '''

    # Set up a ParameterSpace object. This will keep track of all parameter combinations we want
    # to try during testing and training. We load the set of possible parameters and their possible
    # values using the configuration file. If we are using a restart file, then load it's 
    # ParameterSpace object.
    param_space = ParameterSpace(config)
    if (restart_dict is not None):
        param_space.load(restart_dict['parameter_space'])
    
    # Get the "physics" object we use to generate the fom dataset.
    physics         = Initialize_Physics(config, param_space.param_name_list)

    # Get the Model (autoencoder). We try to learn dynamics that describe how the latent space of
    # this model evolve over time. If we are using a restart file, then load the saved model 
    # parameters from file.
    Model           = Initialize_Model(physics, config)
    if (restart_dict is not None):
        Model.load(restart_dict['model'])

    # Initialize the latent dynamics model. If we are using a restart file, then load the saved
    # latent dynamics from this file. 
    ld_type = config['latent_dynamics']['type']
    assert(ld_type in config['latent_dynamics'])
    assert(ld_type in ld_dict)
    latent_dynamics = ld_dict[ld_type](Model.n_z, physics.nt, config['latent_dynamics'])
    if (restart_dict is not None):
        latent_dynamics.load(restart_dict['latent_dynamics'])

    # Fetch the trainer type. Note that only "gplasdi" is allowed.
    trainer_type = config['lasdi']['type']
    assert(trainer_type in config['lasdi'])
    assert(trainer_type in trainer_dict)

    # Initialize the trainer object. If we are using a restart file, then load the 
    # trainer from that file.
    trainer = trainer_dict[trainer_type](physics, Model, latent_dynamics, param_space, config['lasdi'][trainer_type])
    if (restart_dict is not None):
        trainer.load(restart_dict['trainer'])

    # All done!
    return trainer, param_space, physics, Model, latent_dynamics



def Initialize_Model(physics : Physics, config : dict) -> torch.nn.Module:
    '''
    Initialize a Model (autoencoder) according to config file. 
    

    
    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    physics: A "Physics" object that allows us to generate the fom dataset. Each Physics object has 
    a corresponding PDE with parameters, and a way to generate a solution to that equation given
    a particular set of parameter values (and an IC, BCs).

    config: This should be a dictionary that we loaded from a .yml file. It should house all the 
    settings we expect to use to generate the data and train the models. We expect this dictionary 
    to contain the following keys (if a key is within a dictionary that is specified by another 
    key, then we tab the sub-key relative to the dictionary key): 
        - latent_space
            - type
    
            
    
    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    A torch.nn.Module object that acts as the trainable model in the gplasdi framework. This model 
    should have a latent space of some form. We learn a set of dynamics to describe how this latent
    space evolves over time. 
    '''

    # First, determine what model we are using in the latent dynamics. Make sure the user 
    # included all the information that is necessary to initialize the corresponding dynamics.
    latent_type : str = config['latent_space']['type']
    assert(latent_type in config['latent_space'])
    assert(latent_type in latent_dict)
    
    # Next, initialize the latent space.
    latent_cfg      : dict              = config['latent_space'][latent_type]
    latent_space    : torch.nn.Module   = latent_dict[latent_type](physics, latent_cfg)

    # All done!
    return latent_space



def Initialize_Physics(config: dict, param_name_list : list[str]) -> Physics:
    '''
    Initialize a physics FOM model according to config file.
    Currently only 'burgers1d' is available.

    

    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    config: This should be a dictionary that we loaded from a .yml file. It should house all the 
    settings we expect to use to generate the data and train the models. We expect this dictionary 
    to contain the following keys (if a key is within a dictionary that is specified by another 
    key, then we tab the sub-key relative to the dictionary key): 
        - physics 
            - type

    param_name_list: A list housing the names of the parameters in the physics model. There should
    be an entry in the configuration file for each named parameter. 
            
    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    A "Physics" object initialized using the parameters in the config['physics'] dictionary. 
    '''

    # First, determine what kind of "physics" object we want to load.
    physics_cfg     : dict      = config['physics']
    physics_type    : str       = physics_cfg['type']

    # Next, initialize the "physics" object we are using to build the simulations.
    physics         : Physics   = physics_dict[physics_type](physics_cfg, param_name_list)

    # All done!
    return physics





# -------------------------------------------------------------------------------------------------
# Sampling functions
# -------------------------------------------------------------------------------------------------

def pick_samples(trainer, config):
    """
    This function uses greedy sampling to pick a new parameter point.

    If physics is offline solver, then we save parameter points to a hdf file that the fom solver 
    can read.
    """

    # for initial step, get initial parameter points from parameter space.
    if (trainer.X_train.size(0) == 0):
        new_sample = trainer.param_space.train_space
    else:
        # for greedy sampling, get a new parameter and append training space.
        new_sample = trainer.get_new_sample_point()
        trainer.param_space.appendTrainSpace(new_sample)

    # for initial step, get initial parameter points from parameter space.
    new_tests = 0
    if (trainer.X_test.size(0) == 0):
        new_test_params = trainer.param_space.test_space
        new_tests = new_test_params.shape[0]
    # TODO(kevin): greedy sampling for a new test parameter?

    # For online physics solver, we go directly obtain new solutions.
    if not trainer.physics.offline:
        next_step, result = NextStep.RunSample, Result.Success
        return result, next_step

    # Save parameter points in hdf5 format, for offline fom solver to read and run simulations.
    from os.path import dirname, exists
    from os import remove
    from pathlib import Path
    cfg_parser = InputParser(config)

    train_param_file = cfg_parser.getInput(['workflow', 'offline_greedy_sampling', 'train_param_file'], fallback="new_train.h5")
    Path(dirname(train_param_file)).mkdir(parents=True, exist_ok=True)

    with h5py.File(train_param_file, 'w') as f:
        f.create_dataset("train_params", new_sample.shape, data = new_sample)
        f.create_dataset("parameters", (len(trainer.param_space.param_name_list),), data = trainer.param_space.param_name_list)
        f.attrs["n_params"] = trainer.param_space.n_param
        f.attrs["new_points"] = new_sample.shape[0]

    # clean up the previous test parameter point file.
    test_param_file = cfg_parser.getInput(['workflow', 'offline_greedy_sampling', 'test_param_file'], fallback = "new_test.h5")
    Path(dirname(test_param_file)).mkdir(parents=True, exist_ok=True)
    if exists(test_param_file):
        remove(test_param_file)

    if (new_tests > 0):
        with h5py.File(test_param_file, 'w') as f:
            f.create_dataset("test_params", new_test_params.shape, data = new_test_params)
            f.create_dataset("parameters", (len(trainer.param_space.param_name_list),), data = trainer.param_name_list)
            f.attrs["n_params"] = trainer.param_space.n_param
            f.attrs["new_points"] = new_test_params.shape[0]

    # Next step is to collect sample from the offline FOM simulation.
    next_step, result = NextStep.CollectSample, Result.Success
    return result, next_step



def run_samples(trainer, config):
    """
    update trainer.X_train and trainer.X_test based on param_space.train_space and param_space.test_space.
    """
    
    if trainer.physics.offline:
        raise RuntimeError("Current physics solver is offline. RunSamples stage cannot be run online!")

    cfg_parser = InputParser(config)

    new_trains = trainer.param_space.n_train() - trainer.X_train.size(0)
    assert(new_trains > 0)
    new_train_params = trainer.param_space.train_space[-new_trains:, :]

    new_tests = trainer.param_space.n_test() - trainer.X_test.size(0)
    if (new_tests > 0):
        new_test_params = trainer.param_space.test_space[-new_tests:, :]

    # We run FOM simulation directly here.

    new_X = trainer.physics.generate_solutions(new_train_params)
    trainer.X_train = torch.cat([trainer.X_train, new_X], dim = 0)
    assert(trainer.X_train.size(0) == trainer.param_space.n_train())

    if (new_tests > 0):
        new_X = trainer.physics.generate_solutions(new_test_params)
        trainer.X_test = torch.cat([trainer.X_test, new_X], dim = 0)
        assert(trainer.X_test.size(0) == trainer.param_space.n_test())

    # Since FOM simulations are already collected, we go to training phase directly.
    next_step, result = NextStep.Train, Result.Success
    return result, next_step



def collect_samples(trainer, config):
    cfg_parser = InputParser(config)
    assert(trainer.physics.offline)

    train_param_file = cfg_parser.getInput(['workflow', 'offline_greedy_sampling', 'train_param_file'], fallback="new_train.h5")
    train_sol_file = cfg_parser.getInput(['workflow', 'offline_greedy_sampling', 'train_sol_file'], fallback="new_Xtrain.h5")

    with h5py.File(train_param_file, 'r') as f:
        new_trains = f.attrs["new_points"]

    with h5py.File(train_sol_file, 'r') as f:
        new_X = torch.Tensor(f['train_sol'][...])
        assert(new_X.shape[0] == new_trains)
        assert(new_X.shape[1] == trainer.physics.nt)
        assert(list(new_X.shape[2:]) == trainer.physics.qgrid_size)
        trainer.X_train = torch.cat([trainer.X_train, new_X], dim = 0)

    assert(trainer.X_train.size(0) == trainer.param_space.n_train())

    test_param_file = cfg_parser.getInput(['workflow', 'offline_greedy_sampling', 'test_param_file'], fallback="new_test.h5")
    test_sol_file = cfg_parser.getInput(['workflow', 'offline_greedy_sampling', 'test_sol_file'], fallback="new_Xtest.h5")
    import os.path
    if (os.path.isfile(test_param_file)):
        with h5py.File(test_param_file, 'r') as f:
            new_tests = f.attrs["new_points"]

        with h5py.File(test_sol_file, 'r') as f:
            new_X = torch.Tensor(f['test_sol'][...])
            assert(new_X.shape[0] == new_tests)
            assert(new_X.shape[1] == trainer.physics.nt)
            assert(list(new_X.shape[2:]) == trainer.physics.qgrid_size)
            trainer.X_test = torch.cat([trainer.X_test, new_X], dim = 0)
        
        assert(trainer.X_test.size(0) == trainer.param_space.n_test())

    next_step, result = NextStep.Train, Result.Success
    return result, next_step



if __name__ == "__main__":
    main()