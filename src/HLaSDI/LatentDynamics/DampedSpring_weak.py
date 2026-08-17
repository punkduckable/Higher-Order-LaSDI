# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

import  logging;

import  numpy;
import  torch;

from    HLaSDI.LatentDynamics.Weak             import  WeakLatentDynamics;
from    HLaSDI.LatentDynamics.Interpolatable   import  InterpolatableLatentDynamics;
from    HLaSDI.LatentDynamics.LatentDynamics   import  LD_Loss_Container;
from    HLaSDI.LatentDynamics.DampedSpring     import  DampedSpring;
from    HLaSDI.Schemas                         import  DampedSpringWeakLatentDynamicsConfig;


# Setup Logger.
LOGGER : logging.Logger = logging.getLogger(__name__);



# -------------------------------------------------------------------------------------------------
# DampedSpring class
# -------------------------------------------------------------------------------------------------

class DampedSpring_weak(WeakLatentDynamics, DampedSpring):
    def __init__(   self,
                    n_z             :   int,
                    Uniform_t_Grid  :   bool,
                    n_p             :   int,
                    config          :   dict) -> None:
        r"""
        Initializes a DampedSpring_weak object. This is a subclass of the LatentDynamics class which
        implements the following latent dynamics

                z''(t) = K z(t) + C z'(t) + b

        Here, z is the latent state. K \in \mathbb{R}^{n x n} represents a generalized spring
        matrix, C represents a damping matrix, and b is an offset/constant forcing function.
        In this expression, K, C, and b are the model's coefficients. There is a separate set of
        coefficients for each combination of parameter values.

        Note that this class inherits `parameters`, `simulate`, and `RHS` from DampedSpring.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        n_z : int
            The number of dimensions in the latent space, where the latent dynamics takes place.
            frame corresponds to time t0, the second to t0 + h, the k'th to t0 + (k - 1)h, etc
            (note that h may depend on the parameter value, but it needs to be constant for a
            specific parameter value). The value of this setting determines which finite difference
            method we use to compute time derivatives.

        n_p : int 
            The number of (scalar) parameters in the parameter space.

        config : dict
            The latent-dynamics configuration dictionary. It must three keys: `type`, `trainable`,
            and `spring_w`. It must have `config["type"] == "spring_w"` and `config["spring_w"]`
            should be a weak-form sub-dictionary containing the following keys:
                - test_func_type: Specifies the kind of bump function. Either "bump" or "PC-poly".
                - test_func_width: The width of each bump.
                - overlap: The amount of overlap between successive bumps.

        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Nothing!
        """

        assert isinstance(config, DampedSpringWeakLatentDynamicsConfig), "config must be a DampedSpringWeakLatentDynamicsConfig, got %s" % str(type(config));

        # Run the base class initializer. This does not set the n_t attribute.
        # Because K and C are n_z x n_z matrices, and b is in \mathbb{R}^n_z, there are
        # n_z*(2*n_z + 1) coefficients in the latent dynamics.
        InterpolatableLatentDynamics.__init__(
            self,
            n_z             = n_z,
            n_coefs         = n_z*(2*n_z + 1),
            n_IC            = 2,
            n_p             = n_p,
            Uniform_t_Grid  = Uniform_t_Grid,
            trainable       = config.trainable,
            config          = config);

        WeakLatentDynamics.__init__(
            self,
            n_z             = n_z,
            n_IC            = 2,
            n_p             = n_p,
            Uniform_t_Grid  = Uniform_t_Grid,
            trainable       = config.trainable,
            config          = config);


        LOGGER.info("Initializing a DampedSpring_weak object with n_z = %d, Uniform_t_Grid = %s" % (
            self.n_z,
            str(self.Uniform_t_Grid),
        ));

        # Setup the loss function.
        self.MSE = torch.nn.MSELoss(reduction = 'mean');
        self.MAE = torch.nn.L1Loss(reduction = 'mean');

        return;


    # ---------------------------------------------------------------------------------------------
    # initialize_coefficients
    # ---------------------------------------------------------------------------------------------

    def initialize_coefficients(
            self,
            Latent_States : list[list[torch.Tensor]],
            t_Grid        : list[torch.Tensor],
            device        : torch.device,
            params        : numpy.ndarray) -> None:
        r"""
        Initialize weak-form damped-spring coefficients to zero.

        This method intentionally does not solve a weak-form least-squares system. For noisy weak
        training, least-squares coefficients computed from a randomly initialized encoder can be a
        poor starting point. Instead, each requested parameter receives trainable zero tensors for
        `K`, `C`, and `b`; the optimizer learns them jointly with the encoder/decoder.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        Latent_States : list[list[torch.Tensor]], len = n_param
            The i'th list element contains two tensors: latent displacement and latent velocity,
            each with shape (n_t(i), n_z). This method uses the displacement dtype to initialize
            coefficients with matching precision.

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
        assert params is not None, "DampedSpring_weak.initialize_coefficients requires `params`";
        assert isinstance(t_Grid, list) and isinstance(Latent_States, list);
        assert len(Latent_States) == len(t_Grid) == params.shape[0];

        for i in range(params.shape[0]):
            assert isinstance(Latent_States[i], list);
            assert len(Latent_States[i]) == self.n_IC;
            assert isinstance(Latent_States[i][0], torch.Tensor);
            dtype  = Latent_States[i][0].dtype;

            K : torch.Tensor = torch.zeros((self.n_z, self.n_z), device = device, dtype = dtype, requires_grad = True);
            C : torch.Tensor = torch.zeros((self.n_z, self.n_z), device = device, dtype = dtype, requires_grad = True);
            b : torch.Tensor = torch.zeros((self.n_z,),          device = device, dtype = dtype, requires_grad = True);
            self.set_train_coefs(params[i, :], {"K": K, "C": C, "b": b}, device);

        # Finally, update the interpolator using the new training coefficients!
        self.update_interpolator();

        return None;





    # ---------------------------------------------------------------------------------------------
    # Compute losses
    # ---------------------------------------------------------------------------------------------

    def compute_losses(
        self,
        Latent_States : list[list[torch.Tensor]],
        t_Grid        : list[torch.Tensor],
        step          : int,
        params        : numpy.ndarray | None = None
    ) -> LD_Loss_Container:
        r"""
        For each combination of parameter values, this function computes the weak-form
        latent-dynamics loss using the K, C, and b coefficients stored in `self.train_coefs`.

        Specifically, let us consider the case when Z has two axes (the case when it has three is
        identical, just with different coefficients for each instance of the leading dimension of
        Z). In this case, we assume the i'th row of Z holds the latent state t_0 + i*dt. We use
        We assume that the latent state is governed by an ODE of the form

                z''(t) = K z(t) + C z'(t) + b

        Coefficients are initialized by `initialize_coefficients(...)` and then looked up directly
        from `self.train_coefs` using `params`. Missing entries are intentional hard errors because
        they indicate that the sampler/training-data path failed to initialize a training
        parameter.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        Latent_States : list[list[torch.Tensor]], len = n_param
            The i'th list element is a 2 element list whose j'th element is a 2d numpy array of
            shape (n_t(i), n_z) whose p, q element holds the q'th component of the j'th derivative
            of the latent state during the p'th time step (whose time value corresponds to the p'th
            element of t_Grid) when we use the i'th combination of parameter values.

        t_Grid : list[torch.Tensor], len = n_param
            i'th element should be a 1d tensor of shape (n_t(i)) whose j'th element holds the time
            value corresponding to the j'th frame when we use the i'th combination of parameter
            values.

        step : int
            The optimizer step number.

        params: numpy.ndarray, shape = (n_param, n_p)
            The i'th row holds the i'th combination of parameter values. These rows are used to
            fetch weak-form test functions and the corresponding native coefficient dictionaries.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        losses : LD_Loss_Container
            Container housing scalar total losses, matching loss weights, parameter rows, and
            scalar diagnostic metrics. Its `losses` dictionary has keys `LD`, `coef`, and `stab`;
            each value is a scalar tensor summed over parameter rows. Per-parameter diagnostics are
            available in `losses.metrics` under metric keys such as `loss/LD/<param>`.
        """

        # Run checks.
        assert(isinstance(t_Grid, list));
        assert(isinstance(Latent_States, list));
        assert params is not None, "DampedSpring_weak.compute_losses requires `params` so it can look up weight functions by parameter tuple.";
        assert len(Latent_States) == len(t_Grid) == params.shape[0];

        # Setup
        loss_LD_list   : list[torch.Tensor]         = [];
        loss_coef_list : list[torch.Tensor]         = [];
        loss_stab_list : list[torch.Tensor]         = [];
        metrics        : dict[str, torch.Tensor]    = {};

        # -----------------------------------------------------------------------------------------
        # Loop over parameter combinations.
        # -----------------------------------------------------------------------------------------

        for i in range(len(t_Grid)):
            assert isinstance(Latent_States[i], list);
            assert len(Latent_States[i]) == self.n_IC;
            for j in range(self.n_IC):
                assert(isinstance(Latent_States[i][j], torch.Tensor));
                assert(len(Latent_States[i][j].shape)   == 2);
                assert(Latent_States[i][j].shape[-1]    == self.n_z);

            params_i = params[i, :].reshape(1, -1);

            # -------------------------------------------------------------------------------------
            # Concatenate the latent displacement and velocity.

            Z       : list[torch.Tensor]  = Latent_States[i];   # len = n_IC, j'th element has shape (n_t, n_z)

            Z_D     : torch.Tensor  = Z[0];                     # shape = (n_t, n_z)
            Z_V     : torch.Tensor  = Z[1];                     # shape = (n_t, n_z)

            Phis0, dPhis0, d2Phis0 = self.get_test_functions(params_i[0, :]);
            Phis    : torch.Tensor  = Phis0.to(device = Z_D.device, dtype = Z_D.dtype);
            dPhis   : torch.Tensor  = dPhis0.to(device = Z_D.device, dtype = Z_D.dtype);
            d2Phis  : torch.Tensor  = d2Phis0.to(device = Z_D.device, dtype = Z_D.dtype);

            # Concatenate Z_D, Z_V and a column of 1's to evaluate the weak-form RHS
            # Phis @ cat[Z_D, Z_V, 1] @ E, where E^T = [K, C, b].
            ones      : torch.Tensor = torch.ones((Z_D.shape[0], 1), device = Z_D.device, dtype = Z_D.dtype);
            ZD_ZV_1   : torch.Tensor = torch.cat([Z_D, Z_V, ones], dim = 1);          # shape = (n_t, 2*n_z + 1)

            # -------------------------------------------------------------------------------------
            # Set up coefs using the provided coefficients.

            # Fetch native trainable coefficients for this parameter. Missing entries intentionally
            # raise KeyError because coefficient initialization should have happened in the sampler.
            coef_dict = self.get_train_coefs(params_i[0, :]);
            K = coef_dict["K"].to(device = Z_D.device, dtype = Z_D.dtype);
            C = coef_dict["C"].to(device = Z_D.device, dtype = Z_D.dtype);
            b   = coef_dict["b"].to(device = Z_D.device, dtype = Z_D.dtype);
            coefs = torch.cat([K.T, C.T, b.reshape(1, self.n_z)], dim = 0);

            # Compute the weak residual used for the latent-dynamics loss.
            lhs_D = torch.matmul(d2Phis, Z_D)
            lhs_V = -torch.matmul(dPhis, Z_V)
            weak_RHS    : torch.Tensor = torch.matmul(torch.matmul(Phis, ZD_ZV_1), coefs);

            # -------------------------------------------------------------------------------------
            # Compute the stability losses and return.

            scale_D = torch.linalg.norm(d2Phis, dim=1, keepdim=True).clamp(min = 1.0e-10);
            scale_V = torch.linalg.norm(dPhis,  dim=1, keepdim=True).clamp(min = 1.0e-10);

            loss_D = self.MSE(lhs_D / scale_D, weak_RHS / scale_D);
            loss_V = self.MSE(lhs_V / scale_V, weak_RHS / scale_V);
            

            Loss_LD_i = 0.5 * loss_D + 0.5 * loss_V;

            # Stability penalty on the equivalent first-order system y' = A y (+ f).
            # For z'' = K z + C z' + b, define y = [z, z'] so A = [[0, I], [K, C]].
            Z0  : torch.Tensor  = torch.zeros((self.n_z, self.n_z), device = coefs.device, dtype = coefs.dtype);
            I   : torch.Tensor  = torch.eye(self.n_z, device = coefs.device, dtype = coefs.dtype);
            A_top    = torch.cat([Z0, I], dim = 1);
            A_bottom = torch.cat([K, C], dim = 1);
            A = torch.cat([A_top, A_bottom], dim = 0);
            Loss_Stab_i = self.stability_penalty(A);

            # Compute coefficient loss.
            Loss_coef_i = torch.norm(K, 'fro') + torch.norm(C, 'fro') + torch.norm(b);

            # Package the results from this combination of parameter values.
            loss_LD_list.append(Loss_LD_i);
            loss_stab_list.append(Loss_Stab_i);
            loss_coef_list.append(Loss_coef_i);
            metrics[f"loss/LD/{str(params[i, :])}"]     = Loss_LD_i.detach();
            metrics[f"loss/coef/{str(params[i, :])}"]   = Loss_coef_i.detach();
            metrics[f"loss/stab/{str(params[i, :])}"]   = Loss_Stab_i.detach();

        loss_LD   : torch.Tensor    = torch.sum(torch.stack(loss_LD_list));
        loss_coef : torch.Tensor    = torch.sum(torch.stack(loss_coef_list));
        loss_stab : torch.Tensor    = torch.sum(torch.stack(loss_stab_list));
        metrics["loss/LD/total"]    = loss_LD.detach();
        metrics["loss/coef/total"]  = loss_coef.detach();
        metrics["loss/stab/total"]  = loss_stab.detach();

        losses_dict = {'LD' : loss_LD, 'coef' : loss_coef, 'stab' : loss_stab};

        return LD_Loss_Container(losses = losses_dict, weights = self.loss_weights, params = params, metrics = metrics);
