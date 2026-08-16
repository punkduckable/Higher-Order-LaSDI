# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

import  logging;

import  torch;

from    HLaSDI.EncoderDecoder                  import  EncoderDecoder;
from    HLaSDI.ParameterSpace                  import  ParameterSpace;
from    HLaSDI.Physics                         import  Physics;
from    HLaSDI.LatentDynamics                  import  LatentDynamics, WeakLatentDynamics;
from    HLaSDI.Trainer.First_Order_Rollout     import  First_Order_Rollout;
from    HLaSDI.Schemas                         import  ExperimentConfig;

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
                 config             : ExperimentConfig):
        """
        First-order rollout trainer for weak-form latent dynamics.

        This class uses the same per-epoch training logic as `First_Order_Rollout`. The only extra
        setup is generating weak-form test functions before each training round. Optional data noise
        is controlled by the base `Trainer` through top-level `trainer.noise_ratio`.
        """

        assert isinstance(config, ExperimentConfig), "config must be an ExperimentConfig, got %s" % str(type(config));
        assert config.trainer.type == "First_Order_Weak", "config.trainer.type = %s, should be First_Order_Weak" % config.trainer.type;

        LOGGER.info("Initializing a First_Order_Weak object");

        # Make sure we are set up to work with a weak-form latent dynamics object.
        assert isinstance(latent_dynamics, WeakLatentDynamics),         "First_Order_Weak a weak latent dynamics object";
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
        # test space, so this covers compute_losses and avoids needing sampler-specific logic.
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
                end_iter        : int,
                profiler        : torch.profiler.profile | None = None) -> None:
        """
        Prepare weak-form test functions, then use `First_Order_Rollout.Iterate` unchanged.

        Once weak-form latent-dynamics losses are computed from the latent trajectories, the
        first-order rollout/reconstruction/IC-rollout iterations are identical to the strong-form
        trainer.
        """

        self._prepare_weak_form_data();
        super().Iterate(start_iter = start_iter, end_iter = end_iter, profiler = profiler);
        return;
