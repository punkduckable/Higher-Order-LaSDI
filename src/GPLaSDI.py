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
from    FiniteDifference            import  Derivative1_Order4, Derivative2_Order2_NonUniform;


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
    # An n_Train element list. The i'th element is is an n_IC element list whose j'th element is a
    # numpy ndarray of shape (nt(u), Frame_Shape) holding a sequence of samples of the j'th 
    # derivative of the FOM solution when we use the i'th combination of parameter values. 
    X_Train : list[list[torch.Tensor]]  = [];  
    t_Train : list[torch.Tensor]        = []; 

    # Same as X_Test, but used for the test set.
    X_Test  : list[list[torch.Tensor]]  = [];  
    t_Test  : list[torch.Tensor]        = [];

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
        
        # Initialize a timer object. We will use this while training.
        self.timer                          = Timer();

        # Extract training/loss hyperparameters from the configuration file. 
        self.lr                     : float     = config['lr'];                     # Learning rate for the optimizer.
        self.n_samples              : int       = config['n_samples'];              # Number of samples to draw per coefficient per combination of parameters
        self.n_rollout_init         : int       = config['n_rollout_init'];         # The initial number of epochs for simulate forward when computing the rollout loss.
        self.iter_rollout_increase  : int       = config['iter_rollout_increase'];  # We increase n_rollout after this many iterations.
        self.rollout_increase_amt   : int       = config['rollout_increase_amt'];   # We increase n_rollout by this much each time we increase it.
        self.n_iter                 : int       = config['n_iter'];                 # Number of iterations for one train and greedy sampling
        self.max_iter               : int       = config['max_iter'];               # We stop training if restart_iter goes above this number. 
        self.max_greedy_iter        : int       = config['max_greedy_iter'];        # We stop performing greedy sampling if restart_iter goes above this number.
        self.loss_weights           : dict      = config['loss_weights'];           # A dictionary housing the weights of the various parts of the loss function.

        LOGGER.debug("  - n_samples = %d, lr = %f, n_iter = %d, LD_weight = %f, coef_weight = %f" \
                     % (self.n_samples, self.lr, self.n_iter, self.loss_weights['LD'], self.loss_weights['coef']));

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

        # Make sure we have at least one training data point.
        assert(len(self.X_Train) > 0);
        assert(len(self.X_Train) == self.param_space.n_train());

        # Fetch parameters.
        n_train             : int               = self.param_space.n_train();
        n_IC                : int               = self.latent_dynamics.n_IC;
        n_rollout           : int               = self.n_rollout_init + self.rollout_increase_amt*(self.restart_iter//self.iter_rollout_increase);
        LD                  : LatentDynamics    = self.latent_dynamics;
        best_loss           : float             = numpy.inf;                    # Stores the lowest loss we get in this round of training.

        # Map everything to self's device.
        device              : str                       = self.device;
        model_device        : torch.nn.Module           = self.model.to(device);
        X_Train_device      : list[list[torch.Tensor]]  = [];
        t_Train_device      : list[torch.Tensor]        = [];
        for i in range(n_train):
            t_Train_device.append(self.t_Train[i].to(device));
            
            ith_X_Train_device  : list[torch.Tensor] = [];
            for j in range(n_IC):
                ith_X_Train_device.append(self.X_Train[i][j].to(device));
            X_Train_device.append(ith_X_Train_device);

        # Make sure the checkpoints and results directories exist.
        from pathlib import Path
        Path(self.path_checkpoint).mkdir(   parents = True, exist_ok = True);
        Path(self.path_results).mkdir(      parents = True, exist_ok = True);
        
        # Set up the t_Grid for the rollout. This should result in an array whose j, k element is
        # the k'th time step we want to solve for when we use the j'th rollout initial condition 
        # when we use the i'th combination of parameter values. Thus, the i,j element should be 
        # t_Train_device[j + k].
        t_Grid_rollout : list[torch.Tensor] = [];
        for i in range(n_train):
            n_t_i   : int   = t_Train_device[i].shape[0];
            n_i     : int   = n_t_i - n_rollout;

            t_Grid_i_Rollout = torch.empty((n_i, n_rollout), dtype = torch.float32);
            for k in range(n_rollout):
                t_Grid_i_Rollout[:, k]  = t_Train_device[k:(k + n_i)];
            t_Grid_rollout.append(t_Grid_i_Rollout);

        # Determine number of iterations we should run in this round of training.
        next_iter   : int = min(self.restart_iter + self.n_iter, self.max_iter);
        
        # Run the iterations!
        LOGGER.info("Training for %d epochs (starting at %d, going to %d) with %d parameters" % (next_iter - self.restart_iter, self.restart_iter, next_iter, n_train));
        for iter in range(self.restart_iter, next_iter):
            # Begin timing the current training step.            
            self.timer.start("train_step");

            # Check if we need to update n_rollout. If so, then we also need to update 
            # t_Grid_rollout.
            if(iter > 0 and ((iter % self.iter_rollout_increase) == 0)):
                n_rollout += self.rollout_increase_amt;

                t_Grid_rollout : list[torch.Tensor] = [];
                for i in range(n_train):
                    n_t_i   : int   = t_Train_device[i].shape[0];
                    n_i     : int   = n_t_i - n_rollout;

                    t_Grid_i_Rollout = torch.empty((n_i, n_rollout), dtype = torch.float32);
                    for k in range(n_rollout):
                        t_Grid_i_Rollout[:, k]  = t_Train_device[k:(k + n_i)];
                    t_Grid_rollout.append(t_Grid_i_Rollout);
            
                LOGGER.info("n_rollout is now %d" % n_rollout);


            # Zero out the gradients. 
            self.optimizer.zero_grad();
            

            # -------------------------------------------------------------------------------------
            # Compute losses

            # Different kinds of models have different losses.
            if(isinstance(model_device, Autoencoder)):
                # Compute the reconstructions.
                Z       : list[list[torch.Tensor]]  = [];
                X_Pred  : list[list[torch.Tensor]]  = [];

                for i in range(n_train):
                    Z.append([model_device.Encoder(*(X_Train_device[i]))]);
                    X_Pred.append([model_device.Decoder(*(Z[i]))]);

                # Compute the reconstruction loss. 
                loss_recon      : torch.Tensor          = torch.zeros(1, dtype = torch.float32);
                for i in range(n_train):
                    loss_recon  += self.MSE(X_Train_device[i][0], X_Pred[i][0]);

                # Compute the latent dynamics and coefficient losses. Also fetch the current latent
                # dynamics coefficients for each training point. The latter is stored in a 3d array 
                # called "coefs" of shape (n_train, N_z, N_l), where N_{\mu} = n_train = number of 
                # training parameter combinations, N_z = latent space dimension, and N_l = number of 
                # terms in the SINDy library.
                coefs, loss_LD, loss_coef       = LD.calibrate(Latent_States    = Z,
                                                               t_Grid           = t_Train_device);

                # Compute the final loss.
                loss = (self.loss_weights['recon']  * loss_recon + 
                        self.loss_weights['LD']     * loss_LD + 
                        self.loss_weights['coef']   * loss_coef);


            elif(isinstance(model_device, Autoencoder_Pair)):
                # Setup. 
                Latent_States       : list[list[torch.Tensor]]  = [];       # len = n_train. i'th element is 2 element list of (n_t_i, n_z) arrays.
                
                Z_Rollout_IC        : list[list[torch.Tensor]]  = [];       # len = n_train. i'th element is 2 element list of (n_t - n_rollout, n_z) arrays.
                Z_Rollout_Targets   : list[list[torch.Tensor]]  = [];       # len = n_train. i'th element is 2 element list of (n_t - n_rollout, n_z) arrays.
                X_Rollout_Targets   : list[list[torch.Tensor]]  = [];       # len = n_train. i'th element is 2 element list of (n_t - n_rollout, n_x[0], ... n_x[-1]) arrays.

                loss_recon_D        : torch.Tensor              = torch.zeros(1, dtype = torch.float32);
                loss_recon_V        : torch.Tensor              = torch.zeros(1, dtype = torch.float32);

                loss_consistency_Z  : torch.Tensor              = torch.zeros(1, dtype = torch.float32);
                loss_consistency_X  : torch.Tensor              = torch.zeros(1, dtype = torch.float32);

                loss_chain_rule_X   : torch.Tensor              = torch.zeros(1, dtype = torch.float32);
                loss_chain_rule_Z   : torch.Tensor              = torch.zeros(1, dtype = torch.float32);


                # Cycle through the combinations of parameter values.
                for i in range(n_train):
                    # Setup. 
                    D_i         : torch.Tensor  = X_Train_device[i][0];
                    V_i         : torch.Tensor  = X_Train_device[i][1];

                    t_Grid_i    : torch.Tensor  = t_Train_device[i];
                    n_t_i       : int           = t_Grid_i.shape[0];


                    # -----------------------------------------------------------------------------
                    # Forward pass

                    # Run the forward pass. This results in an n_train element list whose i'th 
                    # element is a 2 element list whose j'th element is a tensor of shape 
                    # (n_t(i), n_x[0], ... , n_x[-1]) whose [k, ...] slice holds our prediction for 
                    # the j'th time derivative of the FOM solution at time t_Grid[i][k] when we use 
                    # the i'th combination of parameter values. Here, n_t(i) is the number of time 
                    # steps in the solution for the i'th combination of parameter values. 
                    Z_i     : list[torch.Tensor]        = list(model_device.Encode(*X_Train_device[i]));
                    Z_D_i   : torch.Tensor              = Z_i[0];       # shape (n_t(i), n_z)
                    Z_V_i   : torch.Tensor              = Z_i[1];       # shape (n_t(i), )
                    Latent_States.append(Z_i);

                    X_Pred_i    : list[torch.Tensor]    = list(model_device.Decode(*Z_i));
                    D_Pred_i    : torch.Tensor          = X_Pred_i[0];
                    V_Pred_i    : torch.Tensor          = X_Pred_i[1];


                    # ----------------------------------------------------------------------------
                    # Reconstruction loss

                    # Compute the reconstruction loss. 
                    loss_recon_D += self.MSE(D_i, D_Pred_i);
                    loss_recon_V += self.MSE(V_i, V_Pred_i);


                    # --------------------------------------------------------------------------------
                    # Consistency losses

                    # Make sure Z_V actually looks like the time derivative of Z_X. 
                    if(self.physics.Uniform_t_Grid == True):
                        h               : float             = t_Grid_i[1] - t_Grid_i[0];
                        dZ_Di_dt        : torch.Tensor      = Derivative1_Order4(X = Z_D_i, h = h);
                    else:
                        dZ_Di_dt        : torch.Tensor      = Derivative2_Order2_NonUniform(X = Z_D_i, t_Grid = t_Grid_i);
                    
                    loss_consistency_Z  : torch.Tensor      = loss_consistency_Z + self.MSE(dZ_Di_dt, Z_V_i);

                    # Next, make sure that V_Pred actually looks like the derivative of D_Pred. 
                    if(self.physics.Uniform_t_Grid  == True):
                        h               : float             = t_Grid_i[1] - t_Grid_i[0];
                        dD_Pred_i_dt    : torch.Tensor      = Derivative1_Order4(X = D_Pred_i, h = h);
                    else:
                        dZ_Di_dt        : torch.Tensor      = Derivative2_Order2_NonUniform(X = D_Pred_i, t_Grid = t_Grid_i);

                    loss_consistency_X  : torch.Tensor      = loss_consistency_X + self.MSE(dD_Pred_i_dt, V_Pred_i);

            
                    # ----------------------------------------------------------------------------
                    # Chain Rule Losses

                    # First, we compute the X portion of the chain rule loss. This stems from the 
                    # fact that 
                    #       (d/dt)X(t) \approx (d/dt)\phi_D,D(Z_D(t)) 
                    #                   = (d/dz)\phi_D,D(Z_D(t)) Z_V(t)
                    # Here, \phi_D,D is the displacement portion of the decoder. We can use torch 
                    # to compute jacobian-vector products. Note that the jvp function expects a 
                    # function as its first arguments (to define the forward pass). It passes the 
                    # inputs through func, then computes the jacobian-vector product (using 
                    # reverse mode AD) of inputs with v. It returns the result of the forward pass 
                    # and the associated jacobian-vector product. We only keep the latter.
                    d_dz_D_Pred__Z_V_i  : torch.Tensor  = torch.autograd.functional.jvp(
                                                                func    = lambda Z_D : model_device.Displacement_Autoencoder.Decode(Z_D), 
                                                                inputs  = Z_D_i, 
                                                                v       = Z_V_i)[1];
                    loss_chain_rule_X += self.MSE(V_i, d_dz_D_Pred__Z_V_i);

                    # Next, we compute the Z portion of the chain rule loss:
                    #       (d/dt)Z(t) \approx (d/dt)\phi_E,D(D(t))
                    #                   = (d/dX)\phi_E,D(D(t)) V(t)
                    # Here, \phi_E,D is the displacement portion of the encoder.
                    d_dx_Z_D__V         : torch.Tensor  = torch.autograd.functional.jvp(
                                                                func    = lambda D : model_device.Displacement_Autoencoder.Encode(D),
                                                                inputs  = D_i, 
                                                                v       = V_i)[1];
                    loss_chain_rule_Z += self.MSE(Z_V_i, d_dx_Z_D__V);


                    # ----------------------------------------------------------------------------
                    # Setup Rollout losses.

                    # Select the latent states we want to use as initial conditions for the i'th 
                    # combination of parameter values. This should be any latent state other than
                    # the final n_rollout (so we can simulate forward n_rollout steps and still get
                    # valid target values). Each element of Z_Rollout_IC is a 2 element list of 
                    # numpy.ndarray objects of shape (n_t - n_rollout, n_z).
                    Z_Rollout_IC.append([[Z_D_i[:(-n_rollout), :], Z_V_i[:(-n_rollout), :]]]);

                    # Fetch the corresponding targets (the actual latent states n_rollout steps
                    # in the future).
                    Z_Rollout_Targets.append([Z_D_i[n_rollout:, :], Z_V_i[n_rollout:, :]]);
                
                    # Finally, fetch the FOM targets (the FOM frames n_rollout steps in the future)
                    X_Rollout_Targets.append(D_i[n_rollout:, ...], V_i[n_rollout:, ...]);


                # --------------------------------------------------------------------------------
                # Latent Dynamics, Coefficient losses

                # Compute the latent dynamics and coefficient losses. Also fetch the current latent
                # dynamics coefficients for each training point. The latter is stored in a 2d array 
                # called "coefs" of shape (n_train, n_coefs), where n_train = number of training 
                # parameter parameters and n_coefs denotes the number of coefficients in the latent
                # dynamics model. 
                coefs, loss_LD, loss_coef       = LD.calibrate(Latent_States = Latent_States, t_Grid    = t_Train_device);


                # ---------------------------------------------------------------------------------
                # Rollout loss. Note that we need the coefficients before we can compute this.

                # Setup
                loss_rollout_Z_D    : torch.Tensor              = torch.zeros(1, dtype = torch.float32);
                loss_rollout_Z_V    : torch.Tensor              = torch.zeros(1, dtype = torch.float32);
                loss_rollout_D      : torch.Tensor              = torch.zeros(1, dtype = torch.float32);
                loss_rollout_V      : torch.Tensor              = torch.zeros(1, dtype = torch.float32);

                # Simulate the frames forward in time. This should return an n_param element list
                # whose i'th element is a 2 element list whose j'th element has shape (n_rollout, 
                # n_t(i) - n_rollout, n_z). The p, q, r element of the j'th element should hold the 
                # r'th component of the p'th frame of the j'th time derivative of the solution when 
                # we use the p'th initial condition for the i'th combination of parameter values.
                Z_Rollout           : list[list[torch.Tensor]]  = self.latent_dynamics.simulate(coefs = coefs, IC = Z_Rollout_IC, t_Grid = t_Grid_rollout);            

                # Now cycle through the training examples.
                for i in range(n_train):
                    # Fetch the latent displacement/velocity for the i'th combination of parameter
                    # values. 
                    Z_Rollout_i             : list[torch.Tensor]    = Z_Rollout[i];
                    Z_D_Rollout_i           : torch.Tensor          = Z_Rollout_i[0];               # shape = (n_t(i), n_t(i) - n_rollout, n_z)
                    Z_V_Rollout_i           : torch.Tensor          = Z_Rollout_i[1];               # shape = (n_t(i), n_t(i) - n_rollout, n_z)

                    # The final frame from each simulation is the prediction. 
                    Z_D_Rollout_Predict_i   : torch.Tensor          = Z_D_Rollout_i[-1, :, :];      # shape = (n_t(i) - n_rollout, n_z)
                    Z_V_Rollout_Predict_i   : torch.Tensor          = Z_V_Rollout_i[-1, :, :];      # shape = (n_t(i) - n_rollout, n_z)

                    # Now fetch the corresponding targets.
                    Z_Rollout_Targets_i     : list[torch.Tensor]    = Z_Rollout_Targets[i];
                    Z_D_Rollout_Target_i    : torch.Tensor          = Z_Rollout_Targets_i[0];       # shape = (n_t(i) - n_rollout, n_z)
                    Z_V_Rollout_Target_i    : torch.Tensor          = Z_Rollout_Targets_i[1];       # shape = (n_t(i) - n_rollout, n_z)

                    # Decode the latent predictions to get FOM predictions.
                    D_Rollout_Predict_i, V_Rollout_Predict_i = model_device.Decode(Z_D_Rollout_Predict_i, Z_V_Rollout_Predict_i);
                
                    # Get the corresponding FOM targets.
                    X_Rollout_Target_i      : list[torch.Tensor]    = X_Rollout_Targets[i];
                    D_Rollout_Target_i      : torch.Tensor          = X_Rollout_Target_i[0];
                    V_Rollout_Target_i      : torch.Tensor          = X_Rollout_Target_i[1];
                
                    # Compute the losses for the i'th combination of parameter values!
                    loss_rollout_Z_D +=  self.MSE(Z_D_Rollout_Target_i, Z_D_Rollout_Predict_i);
                    loss_rollout_Z_V +=  self.MSE(Z_V_Rollout_Target_i, Z_V_Rollout_Predict_i);
                    loss_rollout_D   +=  self.MSE(D_Rollout_Target_i,   D_Rollout_Predict_i);
                    loss_rollout_V   +=  self.MSE(V_Rollout_Target_i,   V_Rollout_Predict_i);


                # --------------------------------------------------------------------------------
                # Total loss

                loss_recon          : torch.Tensor  = loss_recon_D + loss_recon_V;
                loss_consistency    : torch.Tensor  = loss_consistency_Z + loss_consistency_X;
                loss_chain_rule     : torch.Tensor  = loss_chain_rule_X + loss_chain_rule_Z;
                loss_rollout        : torch.Tensor  = loss_rollout_Z_D + loss_rollout_Z_V + loss_rollout_D + loss_rollout_V;

                # Compute the final loss.
                loss = (self.loss_weights['recon']          * loss_recon + 
                        self.loss_weights['consistency']    * loss_consistency +
                        self.loss_weights['chain_rule']     * loss_chain_rule + 
                        self.loss_weights['rollout']        * loss_rollout +
                        self.loss_weights['LD']             * loss_LD + 
                        self.loss_weights['coef']           * loss_coef);


            # Convert coefs to numpy and find the maximum element.
            coefs           : numpy.ndarray = coefs.numpy();                # Shape = (n_train, n_coefs).
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
                            % (iter + 1, self.max_iter, loss.item(), loss_recon.item(), loss_LD.item(), loss_coef.item(), max_coef));
            elif(isinstance(model_device, Autoencoder_Pair)):
                LOGGER.info("Iter: %05d/%d, Total: %3.6f, Recon D: %3.6f, Recon V: %3.6f, CR X: %3.6f, CR Z: %3.6f, Cons Z: %3.6f, Cons X: %3.6f, Roll D: %3.6f, Roll V: %3.6f, Roll ZD: %3.6f, Roll ZV: %3.6f, LD: %3.6f, Coef: %3.6f, max|c|: %04.1f, "
                            % (iter + 1, self.max_iter, loss.item(), loss_recon_D.item(), loss_recon_V.item(), loss_chain_rule_X.item(), loss_chain_rule_Z.item(), loss_consistency_Z.item(), loss_consistency_X.item(), loss_rollout_D.item(), loss_rollout_V.item(), loss_rollout_Z_D.item(), loss_rollout_Z_V.item(), loss_LD.item(), loss_coef.item(), max_coef)); 

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
        assert(len(self.X_Test)             >  0);
        assert(len(self.X_Test)             == self.param_space.n_test());
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
        # Yields an n_test element list whose i'th element is an n_IC element list whose j'th
        # element is an numpy.ndarray of shape (n_z) whose k'th element holds the k'th component
        # of the encoding of the initial condition for the j'th derivative of the latent dynamics 
        # corresponding to the i'th combination of parameter values.
        Z0 : list[list[numpy.ndarray]]  = model.latent_initial_conditions(self.param_space.test_space, self.physics);

        # Train the GPs on the training data, get one GP per latent space coefficient.
        gp_list : list[GaussianProcessRegressor] = fit_gps(self.param_space.train_space, coefs);

        # For each combination of parameter values in the testing set, for each coefficient, 
        # draw a set of samples from the posterior distribution for that coefficient evaluated at
        # the testing parameters. We store the samples for a particular combination of parameter 
        # values in a 2d numpy.ndarray of shape (n_sample, n_coef), whose i, j element holds the 
        # i'th sample of the j'th coefficient. We store the arrays for different parameter values 
        # in a list of length n_test. 
        coef_samples : list[numpy.ndarray] = [sample_coefs(gp_list, self.param_space.test_space[i], self.n_samples) for i in range(n_test)];

        # Now, solve the latent dynamics forward in time for each set of coefficients in 
        # coef_samples. There are n_test combinations of parameter values, and we have n_samples 
        # sets of coefficients for each combination of parameter values. For the i'th one of those,
        # we want to solve the latent dynamics for n_t(i) times steps. Each solution frame consists
        # of n_IC elements of \marthbb{R}^{n_z}.
        # 
        # Thus, we store the latent states in an n_test element list whose i'th element is an n_IC
        # element list whose j'th element is an array of shape (n_samples, n_t(i), n_z) whose
        # p, q, r element holds the r'th component of j'th derivative of the latent state at the 
        # q'th time step when we use the p'th set of coefficient values sampled from the posterior
        # distribution for the i'th combination of testing parameter values.
        LatentStates    : list[list[numpy.ndarray]]     = [];
        n_z             : int                           = self.latent_dynamics.n_z;
        for i in range(n_test):
            LatentStates_i  : list[numpy.ndarray]    = [];
            for j in range(n_IC):
                LatentStates_i.append(numpy.ndarray([self.n_samples, len(self.t_Test[j]), n_z]));
            LatentStates.append(LatentStates_i);
        
        for i in range(n_test):
            # Fetch the t_Grid for the i'th combination of parameter values.
            t_Grid  : numpy.ndarray = self.t_Test[i];
            n_t_j   : int           = len(t_Grid);

            # Reshape each element of the IC to have shape (1, n_z), which is what simulate expects
            Z0_i     = Z0[i];
            for d in range(n_IC):
                Z0_i[d] = Z0_i[d].reshape(1, -1);

            # Simulate one sample at a time; store the resulting frames.           
            for j in range(self.n_samples):
                LatentState_ij : list[numpy.ndarray] = self.latent_dynamics.simulate(   coefs   = coef_samples[i][j:(j + 1), :], 
                                                                                        IC      = [Z0_i], 
                                                                                        t_Grid  = [t_Grid]);
                for k in range(n_IC):
                    LatentStates[i][k][j, :, :] = LatentState_ij[k].reshape(n_t_j, n_z);

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
                 't_Train'          : self.t_Train,
                 't_Test'           : self.t_Test,
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
        self.X_Train        : list[list[torch.Tensor]]  = dict_['X_Train'];
        self.X_Test         : list[list[torch.Tensor]]  = dict_['X_Test'];

        self.t_Train        : list[numpy.ndarray]       = dict_['t_Train'];
        self.t_Test         : list[numpy.ndarray]       = dict_['t_Test'];

        self.best_coefs     : numpy.ndarray             = dict_['best_coefs'];
        self.restart_iter   : int                       = dict_['restart_iter'];

        # Load the timer / optimizer. 
        self.timer.load(dict_['timer']);
        self.optimizer.load_state_dict(dict_['optimizer']);
        if (self.device != 'cpu'):
            optimizer_to(self.optimizer, self.device);

        # All done!
        return;
    