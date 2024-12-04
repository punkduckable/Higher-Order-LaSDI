# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

import  h5py;
import  torch;

from    InputParser         import  InputParser;
from    Enums               import  NextStep, Result;  
from    GPLaSDI             import  BayesianGLaSDI;



# -------------------------------------------------------------------------------------------------
# Sampling functions
# -------------------------------------------------------------------------------------------------

def Pick_Samples(trainer : BayesianGLaSDI, config : dict):
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



def Run_Samples(trainer : BayesianGLaSDI, config : dict):
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
