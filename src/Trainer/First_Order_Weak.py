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

import  torch;

from    EncoderDecoder              import  EncoderDecoder;
from    ParameterSpace              import  ParameterSpace;
from    Physics                     import  Physics;
from    LatentDynamics              import  LatentDynamics;
from    First_Order_Rollout         import  First_Order_Rollout;

# Setup Logger
LOGGER : logging.Logger = logging.getLogger(__name__);



# -------------------------------------------------------------------------------------------------
# Trainer class
# -------------------------------------------------------------------------------------------------

class First_Order_Weak(First_Order_Rollout):
    def __init__(self,
                 physics            : Physics,
                 encoder_decoder    : EncoderDecoder,
                 latent_dynamics    : LatentDynamics,
                 param_space        : ParameterSpace,
                 config             : dict):
        """
        First-order rollout trainer for weak-form latent dynamics.

        This class uses the same per-epoch training logic as `First_Order_Rollout`. The only extra
        setup is generating weak-form test functions before each training round. Optional data noise
        is controlled by the base `Trainer` through top-level `trainer.noise_ratio`.
        """

        assert 'trainer' in config,                                 "config must contain a 'trainer' sub-dictionary";
        assert 'type' in config['trainer'],                         "trainer dictionary must contain a 'type' attribute";
        assert config['trainer']['type'] == "First_Order_Weak",     "config['trainer']['type'] = %s, should be First_Order_Weak" % config['trainer']['type'];
        assert "First_Order_Weak" in config['trainer'],             "First_Order_Weak must be in config['trainer']";

        LOGGER.info("Initializing a First_Order_Weak object");

        # Make sure we are set up to work with a weak-form latent dynamics object.
        assert getattr(latent_dynamics, "type", None) == "weak",        "First_Order_Weak requires latent_dynamics.type == 'weak'";
        assert hasattr(latent_dynamics, "add_weight_functions"),        "latent dynamics must have an `add_weight_functions` method";
        assert hasattr(latent_dynamics, "get_test_functions"),          "latent dynamics must have a `get_test_functions` method";

        super().__init__(   physics             = physics,
                            encoder_decoder     = encoder_decoder,
                            latent_dynamics     = latent_dynamics,
                            param_space         = param_space,
                            config              = config);

        return;



    # ---------------------------------------------------------------------------------------------
    # Test function methods
    # ---------------------------------------------------------------------------------------------

    def _prepare_weak_form_data(self) -> None:
        r"""
        Build weak-form test functions for every testing parameter value.

        The latent-dynamics object owns the generated tensors. This trainer only supplies the
        parameter value and its time grid.
        """

        assert len(self.t_Test) == self.param_space.n_test(), "t_Test is not initialized or has wrong length";

        # Build weights for the *entire* test space once. Training parameters are a subset of the
        # test space, so this covers all calibrations and avoids needing sampler-specific logic.
        for i in range(self.param_space.n_test()):
            params_i = self.param_space.test_space[i, :];
            t_i : torch.Tensor = self.t_Test[i].to(self.device);
            self.latent_dynamics.add_weight_functions(params_i, t_i);

        LOGGER.info("Prepared weak-form test functions for %d test trajectories" % self.param_space.n_test());
        return;



    # ---------------------------------------------------------------------------------------------
    # Iterate
    # ---------------------------------------------------------------------------------------------

    def Iterate(self,
                start_iter      : int,
                end_iter        : int) -> None:
        """
        Prepare weak-form test functions, then use `First_Order_Rollout.Iterate` unchanged.

        Once weak-form latent-dynamics calibration receives the latent trajectories, the first-order
        rollout/reconstruction/IC-rollout iterations are identical to the strong-form trainer.
        """

        self._prepare_weak_form_data();
        super().Iterate(start_iter = start_iter, end_iter = end_iter);
        return;
