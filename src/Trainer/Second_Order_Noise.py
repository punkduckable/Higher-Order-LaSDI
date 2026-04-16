# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

import  sys;
import  os;
# Add sibling (src/*) directories to the search path. This file lives in src/Trainer/.
src_path            : str   = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir));
Physics_Path        : str   = os.path.join(src_path, "Physics");
LD_Path             : str   = os.path.join(src_path, "LatentDynamics");
EncoderDecoder_Path : str   = os.path.join(src_path, "EncoderDecoder");
Utils_Path          : str   = os.path.join(src_path, "Utilities");
sys.path.append(Physics_Path);
sys.path.append(LD_Path);
sys.path.append(EncoderDecoder_Path);
sys.path.append(Utils_Path);

import  logging;
from    copy                        import  deepcopy;

import  torch;

from    EncoderDecoder              import  EncoderDecoder;
from    ParameterSpace              import  ParameterSpace;
from    Physics                     import  Physics;
from    LatentDynamics              import  LatentDynamics;
from    Second_Order_Rollout        import  Second_Order_Rollout;

# Setup Logger
LOGGER : logging.Logger = logging.getLogger(__name__);



# -------------------------------------------------------------------------------------------------
# Trainer class
# -------------------------------------------------------------------------------------------------

class Second_Order_Noise(Second_Order_Rollout):
    # Noise ratio for corrupting training data (0.0 = no noise).
    noise_ratio : float;

    # Clean (noise-free) training data, stored when noise_ratio > 0.
    # Same structure as U_Train.
    U_Train_clean : list[list[torch.Tensor]];

    def __init__(self, 
                 physics            : Physics, 
                 encoder_decoder    : EncoderDecoder, 
                 latent_dynamics    : LatentDynamics, 
                 param_space        : ParameterSpace, 
                 config             : dict):
        """
        This defines a variant of Second_Order_Rollout that can add noise to the data. It assumes
        the latent dynamics have two initial conditions (n_IC = 2). Besides being able to add noise,
        it is otherwise identical to the base Rollout class.

        It can only be paired with Latent_Dynamics, Physics, and EncoderDecoder sub-classes which 
        also have n_IC = 2.

        **Configuration format**

        - `config['trainer']` contains base trainer settings such as `n_iter`, `max_iter`,
          `max_greedy_iter`, `normalize`, and `device`.
        - Subclass-specific hyperparameters live under `config['trainer']['Second_Order_Noise']`
          (This must contain the same setting as Second_Order_Rollout, but can optionally include 
           a "noise_ratio" attribute to add noise.).


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        physics : Physics
            Encodes the FOM. It allows us to fetch the FOM solution and/or initial conditions 
            for a particular combination of parameters. We use this object to generate FOM 
            solutions which we then use to train the encoder_decoder and latent dynamics.
         
        encoder_decoder : EncoderDecoder
            use to compress the FOM state to a reduced, latent state.

        latent_dynamics : LatentDynamics
            A LatentDynamics object which describes how we specify the dynamics in the 
            EncoderDecoder's latent space.

        param_space: ParameterSpace
            holds the set of testing and training parameters. 

        config: dict
            houses the Trainer settings. This should contain a 'trainer' sub-dictionary.

        
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Nothing!
        """
        
        # Make sure the Config is set up for `Second_Order_Noise`
        assert 'trainer' in config,                                 "config must contain a 'trainer' sub-dictionary";
        assert 'type' in config['trainer'],                         "trainer dictionary must contain a 'type' attribute";
        assert config['trainer']['type'] == "Second_Order_Noise",   "config['trainer']['type'] = %s, should be Second_Order_Noise" % config['trainer']['type'];
        assert "Second_Order_Noise" in config['trainer'],           "Second_Order_Noise must be in config['trainer']";

        LOGGER.info("Initializing a Second_Order_Noise object"); 


        # Extract the noise_ratio: the ratio of noise std to signal RMS.
        self.noise_ratio        : float = float(config['trainer']['Second_Order_Noise'].get('noise_ratio', 0.0));
        if self.noise_ratio > 0.0:
            LOGGER.info("Noise injection enabled: noise_ratio = %f" % self.noise_ratio);
        else:
            LOGGER.info("Noise injection disabled (noise_ratio = 0.0)");

        # Now, set everything up using the `Second_Order_Rollout` initializer. To do this, 
        # we need to trick the sub-class into thinking we were configured for a 
        # `Second_Order_Rollout` class. To do this, we simply change the 'type' argument 
        # and add a "Second_Order_Rollout" sub-dictionary.
        rollout_config : dict = deepcopy(config);
        rollout_config['trainer']['type']                   = "Second_Order_Rollout";
        del rollout_config['trainer']['Second_Order_Noise'];
        rollout_config['trainer']['Second_Order_Rollout']   = config['trainer']["Second_Order_Noise"];

        # Set up U_Train_clean
        self.U_Train_clean = [];

        # Now call the super-class initialize to finish the set up.
        super().__init__(   physics             = physics, 
                            encoder_decoder     = encoder_decoder, 
                            latent_dynamics     = latent_dynamics, 
                            param_space         = param_space, 
                            config              = rollout_config);

        # All done!
        return;



    # ---------------------------------------------------------------------------------------------
    # Method to add noise
    # ---------------------------------------------------------------------------------------------

    @staticmethod
    def addNoise(x : torch.Tensor, noise_ratio : float) -> torch.Tensor:
        """
        Add Gaussian noise to a tensor, scaled by the signal's RMS power.

        sigma = noise_ratio * sqrt(mean(x^2))
        noise ~ N(0, sigma)

        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        x : torch.Tensor
            The clean signal to corrupt.

        noise_ratio : float
            The ratio of the noise standard deviation to the signal RMS.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        x_noisy : torch.Tensor
            The corrupted signal (same shape and dtype as x).
        """

        if noise_ratio <= 0.0:
            return x;
        
        signal_power    : float         = float(torch.sqrt(torch.mean(x**2)).item());
        sigma           : float         = noise_ratio * signal_power;
        noise           : torch.Tensor  = torch.normal(mean = 0.0, std = sigma, size = x.shape).to(dtype = x.dtype, device = x.device);
        return x + noise;



    def apply_noise_to_U_Train(self) -> None:
        """
        Apply Gaussian noise to the current training data (self.U_Train).

        Before corrupting the data, a deep copy of the clean training data is saved in
        self.U_Train_clean so that noise-free references remain available (e.g., for
        initial conditions). Note that the first frame (IC) of every trajectory is left
        untouched because we assume perfect initial conditions.
        """

        if self.noise_ratio <= 0.0:
            return;

        LOGGER.info("Applying noise (ratio = %f) to %d training trajectories" % (self.noise_ratio, len(self.U_Train)));

        # Deep-copy clean data before corruption. Notably, if U_Train is longer than 
        # U_Train_clean, then the extra elements of U_Train were added by the sampler and do not
        # yet have any noise; we need to back them up in U_Train_clean.
        for i in range( len(self.U_Train_clean), len(self.U_Train) ):
            self.U_Train_clean.append([u.clone() for u in self.U_Train[i]]);

        # Corrupt each trajectory, each IC derivative, but preserve the first frame (IC).
        for i in range(len(self.U_Train)):
            for j in range(len(self.U_Train[i])):
                clean_IC    : torch.Tensor  = self.U_Train_clean[i][j][0:1, ...].clone();     # shape (1, ...)
                noisy_data  : torch.Tensor  = self.addNoise(self.U_Train_clean[i][j].clone(), self.noise_ratio);
                noisy_data[0:1, ...]        = clean_IC;                                  # restore perfect IC
                self.U_Train[i][j]          = noisy_data;
                
                LOGGER.debug("  Trajectory %d, IC %d: signal_rms = %.6e, noise_std = %.6e" % (
                    i, j,
                    float(torch.sqrt(torch.mean(self.U_Train_clean[i][j]**2)).item()),
                    float(self.noise_ratio * torch.sqrt(torch.mean(self.U_Train_clean[i][j]**2)).item())));

        LOGGER.info("Noise injection complete. Clean data saved in U_Train_clean.");
        return;





    # ---------------------------------------------------------------------------------------------
    # Iterate
    # ---------------------------------------------------------------------------------------------

    def Iterate(self, 
                start_iter      : int, 
                end_iter        : int) -> None:
        """
        Same as Second_Order_Rollout.Iterate(), except it applies noise to the training data 
        before launching the iterations. 

        Run one training round for a second-order system (`n_IC = 2`).

        This trainer is designed for higher-order physics where the state is represented via
        multiple time derivatives (e.g., displacement and velocity). Concretely, each training
        trajectory provides two streams `U_D(t)` and `U_V(t)` and the EncoderDecoder is expected
        to encode/decode these jointly (see `Autoencoder_Pair`).

        Each epoch in `[start_iter, end_iter)` typically performs:

        - Forward passes to obtain latent trajectories and reconstructions
        - Latent dynamics calibration/loss evaluation via `latent_dynamics.calibrate(...)`
        - Higher-order consistency losses (e.g., chain-rule and consistency penalties)
        - Optional rollout and IC-rollout losses (curriculum-controlled)
        - Backpropagation + gradient clipping + optimizer step

        **Checkpointing (important)**

        When a new best epoch is found within the round, this method calls
        `Trainer._Save_Checkpoint(...)` to snapshot:

        - EncoderDecoder weights
        - the per-training-point coefficient matrix (`train_coefs`)
        - the full test-space coefficient matrix (`self.test_coefs`)

        This ensures `Trainer.train()` can restore the model and coefficients from the best epoch
        of the round, which is what greedy sampling should use.

        **Loss logging**

        This method records both per-parameter losses and totals using the base-class helpers
        `_store_loss_by_param(...)` and `_store_total_loss(...)`.


        -------------------------------------------------------------------------------------------
        Arguments:
        -------------------------------------------------------------------------------------------

        start_iter : int
            The index of the first training iteration. Must have start_iter <= end_iter.

        end_iter : int 
            The index of the last training iteration. Must have start_iter <= end_iter.

            
        -------------------------------------------------------------------------------------------
        Returns:
        -------------------------------------------------------------------------------------------

        None! 
        """
        
        # First, we need to add noise to the data.
        if self.noise_ratio > 0:
            self.apply_noise_to_U_Train();

        # Next, check if we are using any loss functions which do not play well with noise.
        if(self.noise_ratio > 0):
            if self.loss_weights.get('consistency', 0.0) > 0.0:
                LOGGER.warning(
                    "noise_ratio = %f but consistency weight = %f and weak form is DISABLED. "
                    "Finite-difference consistency losses are unreliable with noisy data; "
                    "consider enabling the weak form or setting consistency weight to 0." % (self.noise_ratio, self.loss_weights['consistency']));
            if self.loss_weights.get('chain_rule', 0.0) > 0.0:
                LOGGER.warning(
                    "noise_ratio = %f but chain_rule weight = %f and weak form is DISABLED. "
                    "Strong-form chain-rule losses compare against noisy FOM velocity and use FD of noisy "
                    "latent states; consider enabling the weak form or setting chain_rule weight to 0." % (self.noise_ratio, self.loss_weights['chain_rule']));

        # We can use the super-class' Iterate method to do the rest.
        super().Iterate(start_iter = start_iter, end_iter = end_iter);

    



    # ---------------------------------------------------------------------------------------------
    # Save, Load
    # ---------------------------------------------------------------------------------------------

    def export(self) -> dict:
        """
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        dict_ : dict
            A dictionary housing most of the internal variables in self. You can pass this 
            dictionary to self (after initializing it using ParameterSpace, encoder_decoder, and 
            LatentDynamics objects) to make a GLaSDI object whose internal state matches that of 
            self.
        """
        # Fetch the super class' save dict.
        dict_ : dict = super().export();

        # Add some noise-specific attributes.
        dict_['noise_ratio']    = self.noise_ratio;
        dict_['U_Train_clean']  = self.U_Train_clean;
        
        return dict_;



    def load(self, dict_ : dict) -> None:
        """
        Modifies self's internal state to match the one whose export method generated the dict_ 
        dictionary.


        -------------------------------------------------------------------------------------------
        Arguments 
        -------------------------------------------------------------------------------------------

        dict_ : dict 
            This should be a dictionary returned by calling the export method on another 
            Second_Order_Noise object. We use this to make self hav the same internal state as the object that 
            generated dict_. 
            

        -------------------------------------------------------------------------------------------
        Returns  
        -------------------------------------------------------------------------------------------
        
        Nothing!
        """

        # Extract class-specific atrributes
        self.noise_ratio    = dict_['noise_ratio']
        self.U_Train_clean  = dict_['U_Train_clean'];

        # Extract the rest of the contents via the base class's load method.
        super().load(dict_);