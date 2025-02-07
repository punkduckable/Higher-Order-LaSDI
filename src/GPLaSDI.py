# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

import  sys;
import  os;
Physics_Path    : str   = os.path.abspath(os.path.join(os.path.dirname(__file__), "Physics"));
LD_Path         : str   = os.path.abspath(os.path.join(os.path.dirname(__file__), "LatentDynamics"));
Utils_Path      : str   = os.path.abspath(os.path.join(os.path.dirname(__file__), "Utilities"));
sys.path.append(Physics_Path);
sys.path.append(LD_Path);
sys.path.append(Utils_Path);

import  logging;

import  torch;
import  numpy;
from    torch.optim                 import  Optimizer;
from    sklearn.gaussian_process    import  GaussianProcessRegressor;

from    GaussianProcess             import  sample_coefs, fit_gps;
from    Model                       import  Autoencoder, Autoencoder_Pair;
from    Timing                      import  Timer;
from    ParameterSpace              import  ParameterSpace;
from    Physics                     import  Physics;
from    LatentDynamics              import  LatentDynamics;
from    Simulate                    import  get_FOM_max_std;
from    FiniteDifference            import  Derivative1_Order4;


# Setup Logger
LOGGER : logging.Logger = logging.getLogger(__name__);



# -------------------------------------------------------------------------------------------------
# BayesianGLaSDI class
# -------------------------------------------------------------------------------------------------

# move optimizer parameters to device
def optimizer_to(optim : Optimizer, device : str) -> None:
    """
    This function moves an optimizer object to a specific device. 


    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    optim: The optimizer whose device we want to change.

    device: The device we want to move optim onto. 


    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    Nothing.
    """

    # Cycle through the optimizer's parameters.
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device);
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device);
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device);
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device);



class BayesianGLaSDI:
    X_Train : list[torch.Tensor]    = [];   # An n_IC element list of ndarrays of shape (n_param, n_t, ...), each holding sequences of some derivative of FOM states
    X_Test  : list[torch.Tensor]    = [];   # Same as X_Test, but used for the test set (X_train holds sequences for the training set).

    def __init__(self, 
                 physics            : Physics, 
                 model              : torch.nn.Module, 
                 latent_dynamics    : LatentDynamics, 
                 param_space        : ParameterSpace, 
                 config             : dict):
        """
        This class runs a full GPLaSDI training. As input, it takes the model defined as a 
        torch.nn.Module object, a Physics object to recover FOM ICs + information on the time 
        discretization, a latent dynamics object, and a parameter space object (which holds the 
        testing and training sets of parameters).

        The "train" method runs the active learning training loop, computes the reconstruction and 
        SINDy loss, trains the GPs, and samples a new FOM data point.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        physics: A "Physics" object that we use to fetch the FOM initial conditions (which we 
        encode into latent ICs). Each Physics object has a corresponding PDE with parameters, and a 
        way to generate a solution to that equation given a particular set of parameter values (and 
        an IC, BCs). We use this object to generate FOM solutions which we then use to train the
        model/latent dynamics.
         
        model: An model object that we use to compress the FOM state to a reduced, latent state.

        latent_dynamics: A LatentDynamics object which describes how we specify the dynamics in the
        model's latent space.

        param_space: A Parameter space object which holds the set of testing and training 
        parameters. 

        config: A dictionary housing the LaSDI settings. This should be the 'lasdi' sub-dictionary 
        of the config file.

        
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Nothing!
        """

        LOGGER.info("Initializing a GPLaSDI object"); 

        self.physics                        = physics;
        self.model                          = model;
        self.latent_dynamics                = latent_dynamics;
        self.param_space                    = param_space;
        
        # Set placeholder tensors to hold the testing and training data. We expect to set up 
        # X_Train to be an n_IC element list of tensors of shape (n_param, n_t, n_x[0], ... , 
        # n_x[Nd - 1]), where n_param is the number of parameter combinations in the training set, 
        # n_t is the number of time steps per FOM solution, and n_x[0], ... , n_x[Nd - 1] represent 
        # the number of steps along the qgrid (spatial axes + vector dimension). X_Test has an 
        # analogous shape, but it's leading dimension has a size matching the number of 
        # combinations of parameters in the testing set.
        # 
        # the latent_dynamics object specifies n_IC while the physics object specifies n_t and the 
        # shape of each fom frame. Using this, we can initialize X_Train and X_Test to hold 
        # tensors whose leading dimension is 0 (indicating that we currently have no testing/
        # training data).
        for i in range(self.latent_dynamics.n_IC):
            FOM_sequence_shape : tuple[int] = (0, self.physics.n_t) + tuple(self.physics.spatial_qgrid_shape);
            self.X_Train.append(torch.empty(FOM_sequence_shape, dtype = torch.float32));
            self.X_Test.append( torch.empty(FOM_sequence_shape, dtype = torch.float32));

        # Initialize a timer object. We will use this while training.
        self.timer                          = Timer();

        # Extract training/loss hyperparameters from the configuration file. 
        self.n_samples          : int       = config['n_samples'];      # Number of samples to draw per coefficient per combination of parameters
        self.lr                 : float     = config['lr'];             # Learning rate for the optimizer.
        self.n_iter             : int       = config['n_iter'];         # Number of iterations for one train and greedy sampling
        self.max_iter           : int       = config['max_iter'];       # We stop training if restart_iter goes above this number. 
        self.max_greedy_iter    : int       = config['max_greedy_iter'];# We stop performing greedy sampling if restart_iter goes above this number.
        self.n_rollout          : int       = config['n_rollout'];      # The number of epochs for simulate forward when computing the rollout loss.
        self.loss_weights       : dict      = config['loss_weights'];   # A dictionary housing the weights of the various parts of the loss function.

        LOGGER.debug("  - n_samples = %d, lr = %f, n_iter = %d, ld_weight = %f, coef_weight = %f" \
                     % (self.n_samples, self.lr, self.n_iter, self.loss_weights['ld'], self.loss_weights['coef']));

        # Set up the optimizer and loss function.
        self.optimizer          : Optimizer = torch.optim.Adam(model.parameters(), lr = self.lr);
        self.MSE                            = torch.nn.MSELoss();

        # Set paths for checkpointing. 
        self.path_checkpoint    : str       = os.path.join(os.path.pardir, "checkpoint");
        self.path_results       : str       = os.path.join(os.path.pardir, "results");

        # Make sure the checkpoints and results directories exist.
        from pathlib import Path;
        Path(os.path.dirname(self.path_checkpoint)).mkdir(  parents = True, exist_ok = True);
        Path(os.path.dirname(self.path_results)).mkdir(     parents = True, exist_ok = True);

        # Set the device to train on. We default to cpu.
        device = config['device'] if 'device' in config else 'cpu';
        if (device == 'cuda'):
            assert(torch.cuda.is_available());
            self.device = device;
        elif (device == 'mps'):
            assert(torch.backends.mps.is_available());
            self.device = device;
        else:
            self.device = 'cpu';

        # Set up variables to aide checkpointing.
        self.best_coefs     : numpy.ndarray = None;             # The best coefficients from the iteration with lowest testing loss
        self.restart_iter   : int           = 0;                # Iteration number at the end of the last training period
        
        # All done!
        return;



    def train(self) -> None:
        """
        Runs a round of training on the model.

        
        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        None!


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Nothing!
        """

        # Make sure we have at least one training data point (the 0 axis of X_Train[0] corresponds 
        # to which combination of training parameters we use).
        assert(self.X_Train[0].shape[0] > 0);
        assert(self.X_Train[0].shape[0] == self.param_space.n_train());

        # Map everything to self's device.
        device              : str                   = self.device;
        model_device        : torch.nn.Module       = self.model.to(device);
        X_Train_device      : list[torch.Tensor]    = [];
        for i in range(len(self.X_Train)):
            X_Train_device.append(self.X_Train[i].to(device));

        # Make sure the checkpoints and results directories exist.
        from pathlib import Path
        Path(self.path_checkpoint).mkdir(   parents = True, exist_ok = True);
        Path(self.path_results).mkdir(      parents = True, exist_ok = True);
        
        # Final setup.
        n_train             : int               = self.param_space.n_train();
        n_IC                : int               = self.latent_dynamics.n_IC;
        n_rollout           : int               = self.n_rollout;
        ld                  : LatentDynamics    = self.latent_dynamics;
        best_loss           : float             = numpy.Inf;                    # Stores the lowest loss we get in this round of training.

        # Determine number of iterations we should run in this round of training.
        next_iter   : int = min(self.restart_iter + self.n_iter, self.max_iter);
        
        # Run the iterations!
        LOGGER.info("Training for %d epochs (starting at %d, going to %d) with %d parameters" % (next_iter - self.restart_iter, self.restart_iter, next_iter, n_train));
        for iter in range(self.restart_iter, next_iter):
            # Begin timing the current training step.            
            self.timer.start("train_step");

            # Zero out the gradients. 
            self.optimizer.zero_grad();


            # -------------------------------------------------------------------------------------
            # Forward pass

            # Run the forward pass. This results in a list of tensors, each of which has shape 
            # (n_param, n_t, n_z), where n_param is the number of parameter combinations in the 
            # training set, n_t is the number of time steps in each time series, and n_z is the 
            # latent space dimension. We decode these tensors to get X_Pred, a list of tensors, 
            # each one of which should have shape (n_parma, n_t, n_x[0], ... , n_x[n_space - 1], 
            # where n_space is the number of spatial dimensions and n_x[k] represents the shape of 
            # the k'th spatial axis.
            Z       = model_device.Encode(*X_Train_device);
            if(isinstance(Z, torch.Tensor)):
                Z       : list[torch.Tensor]    = [Z];
            else:
                Z       : list[torch.Tensor]    = list(Z);
            
            X_Pred = model_device.Decode(*Z);
            if(isinstance(X_Pred, torch.Tensor)):
                X_Pred  : list[torch.Tensor]    = [X_Pred];
            else:
                X_Pred  : list[torch.Tensor]    = list(X_Pred);
            

            # -------------------------------------------------------------------------------------
            # Compute losses

            # Different kinds of models have different losses.
            if(isinstance(model_device, Autoencoder)):
                # Compute the reconstruction loss. 
                loss_recon      : torch.Tensor          = self.MSE(X_Train_device[0], X_Pred[0]);

                # Compute the latent dynamics and coefficient losses. Also fetch the current latent
                # dynamics coefficients for each training point. The latter is stored in a 3d array 
                # called "coefs" of shape (n_train, N_z, N_l), where N_{\mu} = n_train = number of 
                # training parameter combinations, N_z = latent space dimension, and N_l = number of 
                # terms in the SINDy library.
                coefs, loss_ld, loss_coef       = ld.calibrate(Latent_States = Z, dt = self.physics.dt);

                # Compute the final loss.
                loss = (self.loss_weights['recon']  * loss_recon + 
                        self.loss_weights['ld']     * loss_ld / n_train + 
                        self.loss_weights['coef']   * loss_coef / n_train);


            elif(isinstance(model_device, Autoencoder_Pair)):
                # Setup. 
                Z_D     : torch.Tensor  = Z[0];
                Z_V     : torch.Tensor  = Z[1];
                D_Pred  : torch.Tensor  = X_Pred[0];
                V_Pred  : torch.Tensor  = X_Pred[1];


                # --------------------------------------------------------------------------------
                # Consistency losses

                # Make sure Z_V actually looks like the time derivative of Z_X. 
                loss_consistency_Z    : torch.Tensor    = torch.zeros(1, dtype = torch.float32);
                for i in range(n_train):
                    dZ_Di_dt              : torch.Tensor    = Derivative1_Order4(X = Z_D[i, :, :], h = self.physics.dt);
                    loss_consistency_Z  : torch.Tensor      = loss_consistency_Z + self.MSE(dZ_Di_dt, Z_V[i, :, :]);

                # Next, make sure that V_Pred actually looks like the derivative of D_Pred. 
                loss_consistency_X    : torch.Tensor    = torch.zeros(1, dtype = torch.float32);
                for i in range(n_train):
                    dD_Pred_i_dt        : torch.Tensor      = Derivative1_Order4(X = D_Pred[i, ...], h = self.physics.dt);
                    loss_consistency_X  : torch.Tensor      = loss_consistency_X + self.MSE(dD_Pred_i_dt, V_Pred[i, ...]);

                # Compute the consistency loss
                loss_consistency    : torch.Tensor      = loss_consistency_Z + loss_consistency_X;


                # --------------------------------------------------------------------------------
                # Reconstruction loss

                # Compute the reconstruction loss. 
                loss_recon_D    : torch.Tensor      = self.MSE(X_Train_device[0], D_Pred);
                loss_recon_V    : torch.Tensor      = self.MSE(X_Train_device[1], V_Pred);
                loss_recon      : torch.Tensor      = loss_recon_D + loss_recon_V;
            

                # --------------------------------------------------------------------------------
                # Latent Dynamics, Coefficient losses

                # Build the Latent States for calibration.
                Latent_States   : list[torch.Tensor]    = [Z_D, Z_V];

                # Compute the latent dynamics and coefficient losses. Also fetch the current latent
                # dynamics coefficients for each training point. The latter is stored in a 2d array 
                # called "coefs" of shape (n_train, n_coefs), where n_train = number of training 
                # parameter parameters and n_coefs denotes the number of coefficients in the latent
                # dynamics model. 
                coefs, loss_ld, loss_coef       = ld.calibrate(Latent_States = Latent_States, dt = self.physics.dt);


                # --------------------------------------------------------------------------------
                # Rollout losses

                # First, select the latent states we want to simulate forward in time; we need 
                # to have corresponding targets. Each element of Z_Rollout_IC has shape 
                # (n_param, n_t - n_rollout, n_z).
                Z_Rollout_IC    : list[torch.Tensor]    = [Z_D[:, :(-n_rollout), :], Z_V[:, :(-n_rollout), :]];


                # Now set up a fake set of "times". We just need n_rollout times such that the
                # (i + 1)'th time is self.physics.dt greater than the i'th time.
                rollout_times   : numpy.ndarray = numpy.empty((n_rollout), dtype = numpy.float32);
                rollout_times[0] = 0.0;
                for i in range(1, n_rollout):
                    rollout_times[i] = rollout_times[i - 1] + self.physics.dt

                # Simulate the frames forward in time. Each element of Z_rollout should have shape
                # (n_param, n_rollout, n_t - n_rollout, n_z)
                Z_Rollout   : list[torch.Tensor]    = self.latent_dynamics.simulate(coefs = coefs, IC = Z_Rollout_IC, times = rollout_times)

                # Only keep the final simulated frame from each latent trajectory.
                for d in range(self.latent_dynamics.n_IC):
                    Z_Rollout[d]    = Z_Rollout[d][:, -1, :, :];
                
                # Decode the predictions
                D_Rollout_Pred, V_Rollout_Pred  = model_device.Decode(*Z_Rollout);
                
                # Get the corresponding targets (the i'th prediction should match the 
                # (i + n_rollout)'th fom frame. Same for the latent states. Note that the elements 
                # of X_Train should have shape (n_param, n_t, ...).
                D_Rollout_Target    : torch.Tensor  = X_Train_device[0][:, n_rollout:, ...];
                V_Rollout_Target    : torch.Tensor  = X_Train_device[1][:, n_rollout:, ...];

                ZD_Rollout_Target   : torch.Tensor  = Z_D[:, n_rollout:, :];
                ZV_Rollout_Target   : torch.Tensor  = Z_V[:, n_rollout:, :];

                # Compute the rollout loss!
                loss_rollout_ZD     : torch.Tensor  = self.MSE(ZD_Rollout_Target, Z_Rollout[0]);
                loss_rollout_ZV     : torch.Tensor  = self.MSE(ZV_Rollout_Target, Z_Rollout[1]);
                loss_rollout_D      : torch.Tensor  = self.MSE(D_Rollout_Target, D_Rollout_Pred);
                loss_rollout_V      : torch.Tensor  = self.MSE(V_Rollout_Target, V_Rollout_Pred);

                loss_rollout        : torch.Tensor  = loss_rollout_ZD + loss_rollout_ZV + loss_rollout_D + loss_rollout_V;


                # --------------------------------------------------------------------------------
                # Total loss

                # Compute the final loss.
                loss = (self.loss_weights['recon']          * loss_recon + 
                        self.loss_weights['consistency']    * loss_consistency +
                        self.loss_weights['rollout']        * loss_rollout +
                        self.loss_weights['ld']             * loss_ld / n_train + 
                        self.loss_weights['coef']           * loss_coef / n_train);


            # Convert coefs to numpy and find the maximum element.
            coefs           : numpy.ndarray = coefs.numpy();
            max_coef        : numpy.float32 = numpy.abs(coefs).max();


            # -------------------------------------------------------------------------------------
            # Backward Pass

            #  Run back propagation and update the model parameters. 
            loss.backward();
            self.optimizer.step();

            # Check if we hit a new minimum loss. If so, make a checkpoint, record the loss and 
            # the iteration number. 
            if loss.item() < best_loss:
                LOGGER.debug("Got a new lowest loss (%f) on epoch %d" % (loss.item(), iter));
                torch.save(model_device.cpu().state_dict(), self.path_checkpoint + '/' + 'checkpoint.pt');
                
                # Update the best set of parameters. 
                self.best_coefs : numpy.ndarray = coefs;
                best_loss       : float         = loss.item();


            # -------------------------------------------------------------------------------------
            # Report Results from this iteration 

            # Report the current iteration number and losses
            if(isinstance(model_device, Autoencoder)):
                LOGGER.info("Iter: %05d/%d, Total: %3.10f, Recon: %3.10f, LD: %3.10f, Coef: %3.10f, max|c|: %04.1f, "
                            % (iter + 1, self.max_iter, loss.item(), loss_recon.item(), loss_ld.item(), loss_coef.item(), max_coef));
            elif(isinstance(model_device, Autoencoder_Pair)):
                LOGGER.info("Iter: %05d/%d, Total: %3.6f, Recon D: %3.6f, Recon V: %3.6f,  Cons Z: %3.6f, Cons X: %3.6f, Roll D: %3.6f, Roll V: %3.6f, Roll ZD: %3.6f, Roll ZV: %3.6f, LD: %3.6f, Coef: %3.6f, max|c|: %04.1f, "
                            % (iter + 1, self.max_iter, loss.item(), loss_recon_D.item(), loss_recon_V.item(), loss_consistency_Z.item(), loss_consistency_X.item(), loss_rollout_D.item(), loss_rollout_V.item(), loss_rollout_ZD.item(), loss_rollout_ZV.item(), loss_ld.item(), loss_coef.item(), max_coef)); 

            # If there are fewer than 6 training examples, report the set of parameter combinations.
            if n_train < 6:
                param_string : str  = 'Param: ' + str(numpy.round(self.param_space.train_space[0, :], 4));
                for i in range(1, n_train - 1):
                    param_string    = param_string + ', ' + str(numpy.round(self.param_space.train_space[i, :], 4));
                param_string        = param_string + ', ' + str(numpy.round(self.param_space.train_space[-1, :], 4));

                LOGGER.debug(param_string);

            # Otherwise, report the final 6 parameter combinations.
            else:
                param_string : str  = 'Param: ...';
                for i in range(5):
                    param_string    = param_string + ', ' + str(numpy.round(self.param_space.train_space[-6 + i, :], 4));
                param_string        = param_string + ', ' + str(numpy.round(self.param_space.train_space[-1, :], 4));
            
                LOGGER.debug(param_string);

            # We have finished a training step, stop the timer.
            self.timer.end("train_step");
        
        # We are ready to wrap up the training procedure.
        self.timer.start("finalize");

        # Now that we have completed another round of training, update the restart iteration.
        self.restart_iter += self.n_iter;

        # Recover the model + coefficients which attained the lowest loss. If we recorded 
        # our best loss in this round of training, then we replace the model's parameters 
        # with those from the iteration that got the best loss. Otherwise, we use the current 
        # set of coefficients and serialize the current model.
        if ((self.best_coefs is not None) and (self.best_coefs.shape[0] == n_train)):
            state_dict  = torch.load(self.path_checkpoint + '/' + 'checkpoint.pt');
            self.model.load_state_dict(state_dict);
        else:
            self.best_coefs : numpy.ndarray = coefs;
            torch.save(model_device.cpu().state_dict(), self.path_checkpoint + '/' + 'checkpoint.pt');

        # Report timing information.
        self.timer.end("finalize");
        self.timer.print();

        # All done!
        return;



    def get_new_sample_point(self) -> numpy.ndarray:
        """
        This function finds the element of the testing set whose corresponding latent dynamics 
        gives the highest variance FOM time series. 

        How does this work? The latent space coefficients change with parameter values. For each 
        coefficient, we fit a gaussian process whose input is the parameter values. Thus, for each 
        potential parameter value and coefficient, we can find a distribution for that coefficient 
        when we use that parameter value.

        With this in mind, for each combination of parameters in self.param_space's test space, 
        we draw a set of samples of the coefficients at that combination of parameter values. For
        each combination, we solve the latent dynamics forward in time (using the sampled set of
        coefficient values to define the latent dynamics). This gives us a time series of latent 
        states. We do this for each sample, for each testing parameter. 

        For each time step and parameter combination, we get a set of latent frames. We map that 
        set to a set of FOM frames and then find the STD of each component of those FOM frames 
        across the samples. This give us a number. We find the corresponding number for each time 
        step and combination of parameter values and then return the parameter combination that 
        gives the biggest number (for some time step). This becomes the new sample point.

        Thus, the sample point is ALWAYS an element of the testing set. 



        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        None!

        

        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        a 2d numpy ndarray object of shape (1, n_p) whose (0, j) element holds the value of 
        the j'th parameter in the new sample. Here, n_p is the number of parameters.
        """

        self.timer.start("new_sample");
        assert(self.X_Test[0].size(0)       >  0);
        assert(self.X_Test[0].size(0)       == self.param_space.n_test());
        assert(self.best_coefs.shape[0]     == self.param_space.n_train());

        coefs : numpy.ndarray = self.best_coefs;
        LOGGER.info('\n~~~~~~~ Finding New Point ~~~~~~~');

        # Move the model to the cpu (this is where all the GP stuff happens) and load the model 
        # from the last checkpoint. This should be the one that obtained the best loss so far. 
        # Remember that coefs should specify the coefficients from that iteration. 
        model       : torch.nn.Module   = self.model.cpu();
        n_test      : int               = self.param_space.n_test();
        n_IC        : int               = self.latent_dynamics.n_IC;
        model.load_state_dict(torch.load(self.path_checkpoint + '/' + 'checkpoint.pt'));

        # Map the initial conditions for the FOM to initial conditions in the latent space.
        Z0 : list[list[numpy.ndarray]]  = model.latent_initial_conditions(self.param_space.test_space, self.physics);

        # Train the GPs on the training data, get one GP per latent space coefficient.
        gp_list : list[GaussianProcessRegressor] = fit_gps(self.param_space.train_space, coefs);

        # For each combination of parameter values in the testing set, for each coefficient, 
        # draw a set of samples from the posterior distribution for that coefficient evaluated at
        # the testing parameters. We store the samples for a particular combination of parameter 
        # values in a 2d numpy.ndarray of shape (n_sample, n_coef), whose i, j element holds the 
        # i'th sample of the j'th coefficient. We store the arrays for different parameter values 
        # in a list of length (number of combinations of parameters in the testing set). 
        coef_samples : list[numpy.ndarray] = [sample_coefs(gp_list, self.param_space.test_space[i], self.n_samples) for i in range(n_test)];

        # Now, solve the latent dynamics forward in time for each set of coefficients in 
        # coef_samples. There are n_test combinations of parameter values, and we have n_samples 
        # sets of coefficients for each combination of parameter values. For each of those, we want 
        # to solve the corresponding latent dynamics for n_t time steps. Each one of the frames 
        # in that solution live in \mathbb{R}^{n_z}. Thus, we need to store the results in a 4d 
        # array of shape (n_test, n_samples, n_t, n_z) whose i, j, k, l element holds the l'th 
        # component of the k'th frame of the solution to the latent dynamics when we use the 
        # j'th sample of the coefficients for the i'th testing parameter value and when the latent
        # dynamics uses the encoding of the i'th FOM IC as its IC. 
        # 
        # In general, we may need to store n_IC derivatives of the latent solution to get the 
        # full latent state, so we store the results in a list of length n_IC, where n_IC - 1 
        # is the number of derivatives of the latent state we need to fully define the latent 
        # state's initial condition.
        LatentStates    : list[numpy.ndarray]   = [];
        for i in range(n_IC):
            LatentStates.append(numpy.ndarray([n_test, self.n_samples, self.physics.n_t, model.n_z]));
        
        n_t : int = self.physics.n_t;
        n_z : int = self.latent_dynamics.dim;
        for i in range(n_test):
            # Reshape each element of the IC to have shape (1, n_z), which is what simulate expects
            Z0_i     = Z0[i];
            for d in range(n_IC):
                Z0_i[d] = Z0_i[d].reshape(1, -1);

             # Cycle through the samples.            
            for j in range(self.n_samples):
                LatentState_ij : list[numpy.ndarray] = self.latent_dynamics.simulate(coef_samples[i][j, :], Z0_i, self.physics.t_grid);
                for k in range(n_IC):
                    LatentStates[k][i, j, :, :] = LatentState_ij[k].reshape(n_t, n_z);

        # Find the index of the parameter with the largest std.
        m_index : int = get_FOM_max_std(model, LatentStates);

        # We have found the testing parameter we want to add to the training set. Fetch it, then
        # stop the timer and return the parameter. 
        new_sample : numpy.ndarray = self.param_space.test_space[m_index, :].reshape(1, -1);
        LOGGER.info('New param: ' + str(numpy.round(new_sample, 4)) + '\n');
        self.timer.end("new_sample");

        # All done!
        return new_sample;



    def export(self) -> dict:
        """
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        A dictionary housing most of the internal variables in self. You can pass this dictionary 
        to self (after initializing it using ParameterSpace, model, and LatentDynamics 
        objects) to make a GLaSDI object whose internal state matches that of self.
        """

        dict_ = {'X_Train'          : self.X_Train, 
                 'X_Test'           : self.X_Test, 
                 'lr'               : self.lr, 
                 'n_iter'           : self.n_iter,
                 'n_samples'        : self.n_samples, 
                 'best_coefs'       : self.best_coefs, 
                 'max_iter'         : self.max_iter,
                 'max_iter'         : self.max_iter, 
                 'weights'          : self.loss_weights, 
                 'restart_iter'     : self.restart_iter, 
                 'timer'            : self.timer.export(), 
                 'optimizer'        : self.optimizer.state_dict()};
        return dict_;



    def load(self, dict_ : dict) -> None:
        """
        Modifies self's internal state to match the one whose export method generated the dict_ 
        dictionary.


        -------------------------------------------------------------------------------------------
        Arguments 
        -------------------------------------------------------------------------------------------

        dict_: This should be a dictionary returned by calling the export method on another 
        GLaSDI object. We use this to make self hav the same internal state as the object that 
        generated dict_. 
        

        -------------------------------------------------------------------------------------------
        Returns  
        -------------------------------------------------------------------------------------------
        
        Nothing!
        """

        # Extract instance variables from dict_.
        self.X_Train        : list[torch.Tensor]    = dict_['X_Train'];
        self.X_Test         : list[torch.Tensor]    = dict_['X_Test'];
        self.best_coefs     : numpy.ndarray         = dict_['best_coefs'];
        self.restart_iter   : int                   = dict_['restart_iter'];

        # Load the timer / optimizer. 
        self.timer.load(dict_['timer']);
        self.optimizer.load_state_dict(dict_['optimizer']);
        if (self.device != 'cpu'):
            optimizer_to(self.optimizer, self.device);

        # All done!
        return;
    