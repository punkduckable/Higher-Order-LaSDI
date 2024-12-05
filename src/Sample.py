# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

import  h5py;
import  torch;
import  numpy               as      np; 

from    InputParser         import  InputParser;
from    Enums               import  NextStep, Result;  
from    GPLaSDI             import  BayesianGLaSDI;



# -------------------------------------------------------------------------------------------------
# Sampling functions
# -------------------------------------------------------------------------------------------------

def Pick_Samples(trainer : BayesianGLaSDI, config : dict) -> tuple[NextStep, Result]:
    """
    This function uses greedy sampling to pick a new parameter point.

    

    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    trainer: A BayesianGLaSDI object that we use for training. We sample a new training point 
    from this trainer.

    config: This should be a dictionary that we loaded from a .yml file. It should house all the 
    settings we expect to use to generate the data and train the models.
    

    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    A tuple: (NextStep.RunSample, Result.Success). The first returned value, NextStep.RunSample, 
    indicates that we have a new sample and need to generate the fom solution using the 
    corresponding parameter values for the IC. The second returned value, Result.Success, indicates 
    that we were able to pick a new sample without running into any problems. 
    """

    # First, figure out which samples we need to run simulations for. 
    if (trainer.X_train.size(0) == 0):
        # If this is the initial step then trainer.X_Train will be empty, meaning that we need to 
        # run a simulation for every combination of parameters in the train_space. 
        new_sample  : np.ndarray    = trainer.param_space.train_space
    else:
        # If this is not the initial step, then we need to use greedy sampling to pick the new 
        # combination of parameter values.
        new_sample  : np.ndarray    = trainer.get_new_sample_point()
        trainer.param_space.appendTrainSpace(new_sample)

    # Now that we know the new points we need to generate simulations for, we need to get ready to
    # actually run those simulations.
    next_step, result = NextStep.RunSample, Result.Success
    return result, next_step

    """
    This is code that uses the offline stuff (which I disabled). I kept it here for reference.
    
    # First, figure out which samples we need to run simulations for. 
    if (trainer.X_train.size(0) == 0):
        # If this is the initial step then trainer.X_Train will be empty, meaning that we need to 
        # run a simulation for every combination of parameters in the train_space. 
        new_sample  : np.ndarray    = trainer.param_space.train_space
    else:
        # If this is not the initial step, then we need to use greedy sampling to pick the new 
        # combination of parameter values.
        new_sample  : np.ndarray    = trainer.get_new_sample_point()
        trainer.param_space.appendTrainSpace(new_sample)

    # If this is the initial step, we also need to fetch the number of testing points.
    new_tests : int = 0
    if (trainer.X_test.size(0) == 0):
        new_test_params     = trainer.param_space.test_space
        new_tests           = new_test_params.shape[0]
    # TODO(kevin): greedy sampling for a new test parameter?

    # Now that we know the new points we need to generate simulations for, we need to get ready to
    # actually run those simulations.
    if not trainer.physics.offline:
        next_step, result = NextStep.RunSample, Result.Success
        return result, next_step

    # Now that we know the new points we need to generate simulations for, we need to get ready to
    # actually run those simulations.
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
    """



def Run_Samples(trainer : BayesianGLaSDI, config : dict) -> tuple[NextStep, Result]:
    """
    This function updates trainer.X_train and trainer.X_test based on param_space.train_space and 
    param_space.test_space.


    
    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    trainer: A BayesianGLaSDI object that we use for training. 

    config: This should be a dictionary that we loaded from a .yml file. It should house all the 
    settings we expect to use to generate the data and train the models.

    

    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------


    A tuple: (NextStep.Train, Result.Success). The first returned value, NextStep.Train, 
    indicates that we have generated the fom solution for the new training point and need to 
    resume training. The second return value, Result.Success, indicates that we were able to 
    generate the fom solution without running into any problems. 
    """
    
    cfg_parser          : InputParser   = InputParser(config)

    # Figure out how many new training examples there are. Note: we require there is at least one.
    new_trains          : int           = trainer.param_space.n_train() - trainer.X_train.size(0)
    assert(new_trains > 0)

    # Fetch the parameters. The i'th row of this matrix gives the i'th combination of parameter
    # values for which we have not generated a fom solution.
    new_train_params    : np.ndarray    = trainer.param_space.train_space[-new_trains:, :]

    # Figure out how many new testing parameter combinations there are. If there are any, fetch 
    # them from the param space.
    new_tests           : int           = trainer.param_space.n_test() - trainer.X_test.size(0)
    if (new_tests > 0):
        new_test_params     : np.ndarray    = trainer.param_space.test_space[-new_tests:, :]

    # Generate the fom solutions for the new training points. After we have generated them, we
    # append them to trainer's X_Train variable.
    new_X           : torch.Tensor      = trainer.physics.generate_solutions(new_train_params)
    trainer.X_train                     = torch.cat([trainer.X_train, new_X], dim = 0)
    assert(trainer.X_train.size(0) == trainer.param_space.n_train())
    
    # Do the same thins for the testing points.
    if (new_tests > 0):
        new_X = trainer.physics.generate_solutions(new_test_params)
        trainer.X_test = torch.cat([trainer.X_test, new_X], dim = 0)
        assert(trainer.X_test.size(0) == trainer.param_space.n_test())

    # We are now done. Since we now have the new fom solutions, the next step is training.
    next_step, result = NextStep.Train, Result.Success
    return result, next_step



# Note: This code is for offline stuff... I did not document it.
def Collect_Samples(trainer : BayesianGLaSDI, config : dict):
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
