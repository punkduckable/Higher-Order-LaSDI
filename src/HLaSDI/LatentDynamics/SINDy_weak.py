# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

import  logging;

import  numpy;
import  torch;

from    HLaSDI.LatentDynamics.Weak             import  WeakLatentDynamics;
from    HLaSDI.LatentDynamics.Interpolatable   import  InterpolatableLatentDynamics;
from    HLaSDI.LatentDynamics.LatentDynamics   import  LD_Loss_Container;
from    HLaSDI.LatentDynamics.SINDy            import  SINDy;
from    HLaSDI.Schemas                         import  SINDyWeakLatentDynamicsConfig;
from    HLaSDI.Utilities.Statistics            import  tensor_statistics;

LOGGER  : logging.Logger    = logging.getLogger(__name__);



# -------------------------------------------------------------------------------------------------
# SINDy_weak class
# -------------------------------------------------------------------------------------------------

class SINDy_weak(WeakLatentDynamics, SINDy):
    def __init__(   self,
                    n_z             : int,
                    Uniform_t_Grid  : bool,
                    n_p             : int,
                    config          : SINDyWeakLatentDynamicsConfig) -> None:
        r"""
        Initializes a SINDy_weak latent-dynamics object.

        This class is the weak-form version of the affine SINDy latent dynamics

            z'(t) = A z(t) + b.

        Here, z is the latent state, A is an n_z x n_z matrix, and b is an n_z-vector. There is a
        separate set of coefficients for each combination of parameter values. We store the native
        coefficient tensors in `self.train_coefs` as

            {"A": A, "b": b}.

        The weak-form latent-dynamics residual is based on

            - \int phi'(t) z(t) dt = \int phi(t) (A z(t) + b) dt,

        where phi is one of the compactly supported test functions owned by the base class.

        
        Note: This class inherits `parameters`, `RHS`, and `simulate` from SINDy.

        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        n_z : int
            The number of dimensions in the latent space.

        Uniform_t_Grid : bool
            Whether each trajectory has uniform time spacing. This argument is kept for API
            consistency with other latent-dynamics classes; weak compute_losses uses stored test
            functions rather than finite differences.

        n_p : int 
            The number of (scalar) parameters in the parameter space.
            
        config : dict
            The latent-dynamics configuration dictionary. It must three keys: `type`, `trainable`,
            and `sindy_w`. It must have `config["type"] == "sindy_w"` and `config["sindy_w"]` 
            should be a weak-form sub-dictionary containing the following keys:
                - test_func_type: Specifies the kind of bump function. Either "bump" or "PC-poly".
                - test_func_width: The width of each bump.
                - overlap: The amount of overlap between successive bumps.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Nothing!
        """

        assert isinstance(config, SINDyWeakLatentDynamicsConfig), "config must be a SINDyWeakLatentDynamicsConfig, got %s" % str(type(config));

        # Run the base class initializer. Since A has n_z^2 entries and b has n_z entries, there
        # are n_z*(n_z + 1) scalar coefficients.
        InterpolatableLatentDynamics.__init__(
            self,
            n_z            = n_z,
            n_coefs        = n_z*(n_z + 1),
            n_IC           = 1,
            n_p            = n_p,
            Uniform_t_Grid = Uniform_t_Grid,
            trainable      = config.trainable,
            config         = config);

        WeakLatentDynamics.__init__(
            self,
            n_z            = n_z,
            n_IC           = 1,
            n_p            = n_p,
            Uniform_t_Grid = Uniform_t_Grid,
            trainable      = config.trainable,
            config         = config);

        LOGGER.info("Initializing a SINDy_weak object with n_z = %d, Uniform_t_Grid = %s" % (
            self.n_z,
            str(self.Uniform_t_Grid),
        ));

        # Setup the loss functions used by compute_losses.
        self.MSE = torch.nn.MSELoss(reduction = 'mean');
        self.MAE = torch.nn.L1Loss(reduction = 'mean');
        return;



    # ---------------------------------------------------------------------------------------------
    # initialize_coefficients
    # ---------------------------------------------------------------------------------------------

    def initialize_coefficients(
            self,
            Latent_States   : list[list[torch.Tensor]],
            t_Grid          : list[torch.Tensor],
            device          : torch.device,
            params          : numpy.ndarray) -> None:
        r"""
        Initialize weak-form SINDy coefficients to zero.

        This method intentionally does not solve a weak-form least-squares system. For weak
        training, coefficients computed from a randomly initialized encoder can be a poor starting
        point. Instead, each requested parameter receives trainable zero tensors for `A` and `b`;
        the optimizer learns them jointly with the encoder/decoder.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        Latent_States : list[list[torch.Tensor]], len = n_param
            The i'th list element contains one latent trajectory tensor with shape (n_t(i), n_z).
            This method uses the tensor dtype to initialize coefficients with matching precision.

        t_Grid : list[torch.Tensor], len = n_param
            Time grids corresponding to the latent trajectories. These are checked for length
            consistency but are not otherwise used because weak coefficients are zero-initialized.

        device : torch.device
            Device on which the new coefficient tensors should be stored.

        params : numpy.ndarray, shape = (n_param, n_p)
            Parameter rows used as keys in `self.train_coefs`.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        None. Zero coefficient dictionaries are stored in `self.train_coefs`, and the interpolator
        is updated from the full training-coefficient dictionary.
        """

        assert params is not None, "SINDy_weak.initialize_coefficients requires `params`";
        assert isinstance(t_Grid, list);
        assert isinstance(Latent_States, list);
        assert len(Latent_States) == len(t_Grid) == params.shape[0];

        for i in range(params.shape[0]):
            assert isinstance(Latent_States[i], list);
            assert len(Latent_States[i]) == self.n_IC;
            assert isinstance(Latent_States[i][0], torch.Tensor);
            dtype  = Latent_States[i][0].dtype;

            A : torch.Tensor = torch.zeros((self.n_z, self.n_z), device = device, dtype = dtype, requires_grad = True);
            b : torch.Tensor = torch.zeros((self.n_z,),          device = device, dtype = dtype, requires_grad = True);
            self.set_train_coefs(params[i, :], {"A": A, "b": b}, device);

        # Finally, update the interpolator using the new training coefficients!
        self.update_interpolator();

        # All done :) 
        return None;



    # ---------------------------------------------------------------------------------------------
    # compute_losses
    # ---------------------------------------------------------------------------------------------

    def compute_losses(  
        self,
        Latent_States   : list[list[torch.Tensor]],
        t_Grid          : list[torch.Tensor],
        step            : int,
        params          : numpy.ndarray | None = None
    ) -> LD_Loss_Container:
        r"""
        Compute weak-form SINDy latent-dynamics, coefficient, and stability losses.

        For each parameter combination, this method fetches the native coefficient dictionary from
        `self.train_coefs` and the weak-form test functions from the base class. Missing
        coefficients or test functions are intentional hard errors because the sampler/training
        path should initialize both before optimization starts.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        Latent_States : list[list[torch.Tensor]], len = n_param
            The i'th list element contains one latent trajectory tensor of shape (n_t(i), n_z).

        t_Grid : list[torch.Tensor], len = n_param
            Time grids corresponding to the latent trajectories.

        params : numpy.ndarray, shape = (n_param, n_p)
            Parameter rows used to fetch weak-form test functions and coefficient dictionaries.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        losses : LD_Loss_Container
            Container housing scalar total losses, matching loss weights, parameter rows, and
            scalar diagnostic metrics. Its `losses` dictionary has keys `LD`, `coef`, and `stab`;
            each value is a scalar tensor summed over parameter rows. Per-parameter diagnostics are
            available in `losses.metrics` under metric keys such as `loss/LD/<param>`.
        """

        # Checks.
        assert params is not None, "SINDy_weak.compute_losses requires params";
        assert isinstance(t_Grid, list);
        assert isinstance(Latent_States, list);
        assert len(Latent_States) == len(t_Grid) == params.shape[0];

        loss_LD_list            : list[torch.Tensor]        = [];
        loss_coef_list          : list[torch.Tensor]        = [];
        loss_stab_list          : list[torch.Tensor]        = [];
        coef_A_fro_list         : list[torch.Tensor]        = [];
        coef_b_l2_list          : list[torch.Tensor]        = [];
        lambda_max_list         : list[torch.Tensor]        = [];
        weight_fun_residuals    : list[torch.Tensor]        = [];
        metrics                 : dict[str, torch.Tensor]   = {};

        # -----------------------------------------------------------------------------------------
        # Loop over parameter combinations.
        # -----------------------------------------------------------------------------------------

        for i in range(len(t_Grid)):
            assert isinstance(Latent_States[i], list);
            assert len(Latent_States[i]) == self.n_IC;

            # Fetch this parameter's latent trajectory.
            Z : torch.Tensor = Latent_States[i][0];
            assert isinstance(Z, torch.Tensor);
            assert len(Z.shape) == 2;
            assert Z.shape[-1] == self.n_z;

            # Fetch weak test functions and match their device/dtype to Z.
            Phis0, dPhis0 = self.get_test_functions(params[i, :]);
            Phis   : torch.Tensor = Phis0.to(device = Z.device, dtype = Z.dtype);
            dPhis  : torch.Tensor = dPhis0.to(device = Z.device, dtype = Z.dtype);

            # Fetch native trainable coefficients for this parameter.
            coef_dict = self.get_train_coefs(params[i, :]);
            A = coef_dict["A"].to(device = Z.device, dtype = Z.dtype);
            b = coef_dict["b"].to(device = Z.device, dtype = Z.dtype);

            # Compute the weak residual. We follow the existing weak-form convention in this repo:
            # matrix multiplication with the sampled test functions approximates the time integral.
            weak_LHS : torch.Tensor = -torch.matmul(dPhis, Z);
            RHS      : torch.Tensor = torch.matmul(Z, A.T) + b.reshape(1, -1);
            weak_RHS : torch.Tensor = torch.matmul(Phis, RHS);

            # Normalize each test-function residual by the norm of phi' to keep losses comparable
            # across support locations and widths.
            scale : torch.Tensor = torch.linalg.norm(dPhis, dim = 1, keepdim = True).clamp(min = 1.0e-10);
            loss_LD = self.MSE(weak_LHS / scale, weak_RHS / scale);

            # Approximate the L2 (integral) norm of phi_h'(t) z(t) - phi_h(t)  f(z(t), t, theta)
            normalized_residual : torch.Tensor = (weak_LHS - weak_RHS) / scale;
            weight_fun_residuals.append(torch.sqrt(torch.mean(normalized_residual**2, dim = 1)));

            # Compute regularization terms.
            coef_A_fro = torch.norm(A, 'fro');
            coef_b_l2  = torch.norm(b);
            lambda_max = torch.linalg.eigvalsh(0.5*(A + A.T)).max();
            loss_coef = coef_A_fro + coef_b_l2;
            loss_stab = torch.nn.functional.softplus(lambda_max + 0.1);

            loss_LD_list.append(loss_LD);
            loss_coef_list.append(loss_coef);
            loss_stab_list.append(loss_stab);
            coef_A_fro_list.append(coef_A_fro);
            coef_b_l2_list.append(coef_b_l2);
            lambda_max_list.append(lambda_max);
            metrics[f"loss/LD/{str(params[i, :])}"]     = loss_LD.detach();
            metrics[f"loss/coef/{str(params[i, :])}"]   = loss_coef.detach();
            metrics[f"loss/stab/{str(params[i, :])}"]   = loss_stab.detach();

        loss_LD   : torch.Tensor    = torch.sum(torch.stack(loss_LD_list));
        loss_coef : torch.Tensor    = torch.sum(torch.stack(loss_coef_list));
        loss_stab : torch.Tensor    = torch.sum(torch.stack(loss_stab_list));
        metrics["loss/LD/total"]    = loss_LD.detach();
        metrics["loss/coef/total"]  = loss_coef.detach();
        metrics["loss/stab/total"]  = loss_stab.detach();
        metrics.update(tensor_statistics(prefix = "coef/A/fro", values = torch.stack(coef_A_fro_list)));
        metrics.update(tensor_statistics(prefix = "coef/b/l2",  values = torch.stack(coef_b_l2_list)));
        metrics["stability/lambda_max/mean"] = torch.mean(torch.stack(lambda_max_list)).detach();
        metrics["stability/lambda_max/max"]  = torch.max(torch.stack(lambda_max_list)).detach();
        metrics.update(tensor_statistics(prefix = "weak/weight_fun_residuals", values = torch.cat(weight_fun_residuals, dim = 0)));

        losses_dict = {'LD' : loss_LD, 'coef' : loss_coef, 'stab' : loss_stab};

        return LD_Loss_Container(losses = losses_dict, weights = self.loss_weights, params = params, metrics = metrics);
